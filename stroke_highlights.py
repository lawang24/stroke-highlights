"""Local, experimental pose-based tennis stroke detection and video compilation."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from pose_utils import extract_pose_detections, load_yolo, yolo_device_arg


FIELDS = ("id", "time", "start", "end", "stroke", "score", "include", "reason")
LABELS = {"forehand", "backhand", "unknown", "ignore"}


def number(value, minimum=0, strict=False):
    value = float(value)
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise argparse.ArgumentTypeError(f"must be finite and {'greater than' if strict else 'at least'} {minimum}")
    return value


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Video has invalid frame rate or no frames")
        return {"fps": fps, "frames": count, "duration": count / fps,
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    finally:
        cap.release()


def select_player(detections, previous, shape, player, roi=None):
    """Track a box by continuity; never bridge a large spatial jump."""
    height, width = shape[:2]
    candidates = []
    for detection in detections:
        box = detection["box"]
        if box is None:
            continue
        x1, y1, x2, y2 = box
        foot = ((x1 + x2) / (2 * width), y2 / height)
        if roi and not (roi[0] <= foot[0] <= roi[2] and roi[1] <= foot[1] <= roi[3]):
            continue
        if roi is None and player == "far" and foot[1] > 0.65:
            continue
        if roi is None and player == "near" and foot[1] < 0.55:
            continue
        candidates.append(detection)
    if not candidates:
        return None
    if previous is None:
        # Largest person in the selected court region, not the smallest spectator.
        return max(candidates, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    old = np.asarray(previous)
    old_center = (old[:2] + old[2:]) / 2
    old_height = max(old[3] - old[1], 1)
    def distance(detection):
        box = detection["box"]
        center = (box[:2] + box[2:]) / 2
        ratio = max(box[3] - box[1], 1) / old_height
        return np.linalg.norm(center - old_center) / old_height + abs(math.log(ratio))
    best = min(candidates, key=distance)
    return best if distance(best) < 0.9 else None


def collect_poses(source, metadata, args, cache):
    identity = {"version": 1, "source": str(source), "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns, "model": args.model,
                "sample_fps": args.sample_fps, "imgsz": args.imgsz,
                "player": args.player, "roi": args.roi, "limit": args.limit}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("identity") == identity:
            print(f"Reusing poses: {cache}", flush=True)
            return saved["samples"]
        raise ValueError(f"Pose cache settings or input differ: {cache}. Use a new output directory or remove the cache.")
    model = load_yolo(args.model)
    cap = cv2.VideoCapture(str(source))
    stride = max(1, round(metadata["fps"] / args.sample_fps))
    stop = min(metadata["frames"], math.ceil(args.limit * metadata["fps"])) if args.limit else metadata["frames"]
    samples, previous, missing_since = [], None, None
    segment = 0
    try:
        from tqdm import tqdm
        for frame_idx in tqdm(range(stop), desc="Finding player poses", unit="frame"):
            if not cap.grab():
                raise ValueError(f"Video decoding stopped early at frame {frame_idx}")
            if frame_idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {frame_idx}")
            time = frame_idx / metadata["fps"]
            result = model.predict(frame, imgsz=args.imgsz, conf=0.3,
                                   device=yolo_device_arg(args.device), verbose=False)[0]
            detections = extract_pose_detections(result, 20)
            detection = select_player(detections, previous, frame.shape, args.player, args.roi)
            if detection is None:
                if missing_since is None:
                    missing_since = time
                if previous is not None and time - missing_since > 0.6:
                    previous = None
                    segment += 1
                samples.append({"time": time, "segment": segment, "keypoints": None})
                continue
            previous = detection["box"]
            missing_since = None
            samples.append({"time": time, "segment": segment,
                            "box": previous.tolist(), "keypoints": detection["keypoints"].tolist()})
    finally:
        cap.release()
    cache.write_text(json.dumps({"identity": identity, "samples": samples}))
    return samples


def pose_features(sample, handedness):
    if sample["keypoints"] is None:
        return None
    points = np.asarray(sample["keypoints"], dtype=float)
    wrist_idx = 10 if handedness == "right" else 9
    needed = [5, 6, 11, 12, wrist_idx]
    if points.shape != (17, 3) or not np.isfinite(points[needed]).all() or np.min(points[needed, 2]) < 0.4:
        return None
    shoulders = points[[5, 6], :2].mean(axis=0)
    hips = points[[11, 12], :2].mean(axis=0)
    scale = np.linalg.norm(shoulders - hips)
    if scale < 12:
        return None
    wrist = (points[wrist_idx, :2] - hips) / scale
    # Hip-relative coordinates remove most running / camera translation.
    return np.array([wrist[0], wrist[1], (shoulders[1] - hips[1]) / scale])


def detect_strokes(samples, duration, handedness="right", player="near", mirrored=False,
                   before=1.2, after=0.9, min_speed=2.8, min_gap=1.8,
                   frame_height=None, min_player_height=0.2):
    """Propose swings from wrist motion. Scores are heuristics, not probabilities.

    Assumes a fixed camera behind a baseline. Forehands move from the dominant
    side across the torso; backhands move the other way. Follow-through alone
    is insufficient: require matching preparation and forward-swing direction.
    """
    # Fill only brief keypoint occlusions, and only while the same tracked player
    # remains visible. Never interpolate off-screen players or identity changes.
    features = []
    for sample in samples:
        box = sample.get("box")
        too_small = (frame_height and box is not None and
                     (box[3] - box[1]) / frame_height < min_player_height)
        features.append(None if too_small else pose_features(sample, handedness))
    valid = [i for i, feature in enumerate(features) if feature is not None]
    for left, right in zip(valid, valid[1:]):
        span = samples[right]["time"] - samples[left]["time"]
        if right - left <= 1 or span > 0.34:
            continue
        if any(s["keypoints"] is None or s["segment"] != samples[left]["segment"] or
               (frame_height and s.get("box") is not None and
                (s["box"][3] - s["box"][1]) / frame_height < min_player_height)
               for s in samples[left:right + 1]):
            continue
        for index in range(left + 1, right):
            alpha = (samples[index]["time"] - samples[left]["time"]) / span
            features[index] = features[left] * (1 - alpha) + features[right] * alpha
    groups, group = [], []
    last_segment = None
    for sample, feature in zip(samples, features):
        if feature is None or (group and (sample["segment"] != last_segment or sample["time"] - group[-1][0] > 0.2)):
            if group:
                groups.append(group)
            group = []
        if feature is not None:
            group.append((sample["time"], feature))
        last_segment = sample["segment"]
    if group:
        groups.append(group)
    dominant_side = (1 if handedness == "right" else -1) * (1 if player == "near" else -1) * (-1 if mirrored else 1)
    events = []
    for group in groups:
        if len(group) < 7:
            continue
        times = np.array([item[0] for item in group])
        dt = float(np.median(np.diff(times)))
        coords = median_filter(np.array([item[1] for item in group]), size=(3, 1), mode="nearest")
        velocity = np.gradient(coords[:, :2], times, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        peaks, _ = find_peaks(speed, height=min_speed, prominence=min_speed * 0.4,
                              distance=max(1, math.ceil(min_gap / dt)))
        for peak in peaks:
            t = times[peak]
            prep = (times >= t - 0.45) & (times <= t - 0.12)
            finish = (times >= t + 0.08) & (times <= t + 0.3)
            if prep.sum() < 2 or finish.sum() < 2:
                continue
            start_x = float(np.median(coords[prep, 0])) * dominant_side
            end_x = float(np.median(coords[finish, 0])) * dominant_side
            displacement = end_x - start_x
            if abs(displacement) < 0.45:
                continue  # Mostly vertical / walking arm motion.
            low_arm = coords[peak, 1] > coords[peak, 2] - 0.2
            lowering_arm = np.median(coords[finish, 1]) - np.median(coords[prep, 1]) > 0.4
            stroke, reason, score = "unknown", "ambiguous preparation or swing direction", 0.0
            if not low_arm:
                reason = "overhead motion; possible serve or smash"
            elif lowering_arm:
                reason = "lowering arm; possible recovery after a stroke"
            elif start_x > 0.2 and displacement < -0.45:
                stroke, reason = "forehand", "dominant-side preparation and cross-body swing"
            elif start_x < -0.2 and displacement > 0.45:
                stroke, reason = "backhand", "opposite-side preparation and outward swing"
            if stroke != "unknown":
                score = min(1.0, 0.4 + 0.2 * min(abs(start_x), 1) + 0.2 * min(abs(displacement), 2))
            events.append({"time": round(t, 3), "start": round(max(0, t - before), 3),
                           "end": round(min(duration, t + after), 3), "stroke": stroke,
                           "score": round(score, 3), "include": stroke != "unknown", "reason": reason,
                           "_speed": float(speed[peak])})
    # Gaps in keypoints split groups; suppress duplicate proposals across them.
    selected = []
    for event in sorted(events, key=lambda e: e["_speed"], reverse=True):
        if all(abs(event["time"] - e["time"]) >= min_gap for e in selected):
            selected.append(event)
    selected.sort(key=lambda e: e["time"])
    for index, event in enumerate(selected, 1):
        event["id"] = index
        del event["_speed"]
    return selected


def write_events(path, events):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(events)


def read_events(path, duration):
    events = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"start", "end", "stroke", "include"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires start, end, stroke, and include columns")
        for line, row in enumerate(reader, 2):
            try:
                if any(row.get(field) is None for field in ("start", "end", "stroke", "include")):
                    raise ValueError("missing required value")
                start, end = float(row["start"]), float(row["end"])
                if (not all(math.isfinite(v) for v in [start, end]) or
                        not 0 <= start < min(end, duration) or end > duration + 0.05):
                    raise ValueError("invalid time range")
                label = row["stroke"].strip().lower()
                if label not in LABELS:
                    raise ValueError(f"stroke must be one of {sorted(LABELS)}")
                include = row["include"].strip().lower()
                if include not in {"true", "false", "1", "0", "yes", "no"}:
                    raise ValueError("include must be true or false")
                events.append({**row, "start": start, "end": min(end, duration), "stroke": label,
                               "include": include in {"true", "1", "yes"}})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
    return events


def compilation_ranges(events, stroke):
    ranges = sorted((e["start"], e["end"]) for e in events if e["include"] and e["stroke"] == stroke)
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def run_ffmpeg(arguments):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.strip()}")


def export_compilation(source, events, stroke, destination, overwrite=False):
    if destination.resolve() == source.resolve():
        raise ValueError("Output must not overwrite the input video")
    if destination.exists() and not overwrite:
        raise ValueError(f"Output exists: {destination}. Pass --overwrite to replace it.")
    ranges = compilation_ranges(events, stroke)
    if not ranges:
        if destination.exists() and overwrite:
            destination.unlink()  # Do not leave a stale compilation after review.
        print(f"No included {stroke} strokes; no video created.", flush=True)
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fps = probe_video(source)["fps"]
    output_fps = round(source_fps) if abs(source_fps - round(source_fps)) < 0.01 else source_fps
    # Encode each bounded excerpt, then concatenate identical streams. This keeps
    # memory and command length bounded for long recordings and preserves audio.
    with tempfile.TemporaryDirectory(prefix="tennis-", dir=destination.parent) as temp:
        temp = Path(temp)
        parts = []
        for index, (start, end) in enumerate(ranges):
            part = temp / f"clip-{index:05d}.mp4"
            print(f"Exporting {stroke} {index + 1}/{len(ranges)}: {start:.2f}–{end:.2f}s", flush=True)
            run_ffmpeg(["-ss", str(start), "-i", str(source), "-t", str(end - start),
                        "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-af", "asetpts=PTS-STARTPTS", "-avoid_negative_ts", "make_zero", str(part)])
            parts.append(f"file '{part.name}'\n")
        listing = temp / "concat.txt"
        listing.write_text("".join(parts))
        completed = temp / "completed.mp4"
        # AAC packet boundaries can differ slightly from video frame boundaries.
        # Normalize the joined timeline to CFR so those offsets do not create
        # irregular frame timestamps (or playback warnings) at every cut.
        run_ffmpeg(["-f", "concat", "-safe", "1", "-i", str(listing),
                    "-vf", f"fps={output_fps}:start_time=0", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    "-movflags", "+faststart", str(completed)])
        completed.replace(destination)
    print(f"Created {destination}", flush=True)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stroke", choices=["forehand", "backhand", "both"], default="both")
    parser.add_argument("--events-csv", type=Path, help="Export an edited CSV, skipping pose inference")
    parser.add_argument("--analyze-only", action="store_true", help="Write proposals without exporting video")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing events/compilations; retain compatible pose cache")
    parser.add_argument("--player", choices=["near", "far"], default="near")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument("--mirrored", action="store_true", help="Source video is horizontally mirrored")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Optional normalized player foot-position region (0–1), excluding adjacent courts")
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--sample-fps", type=lambda x: number(x, 8), default=15,
                        help="Pose samples per second (minimum 8; source frame rate is the upper limit)")
    parser.add_argument("--before", type=lambda x: number(x), default=1.2, help="Seconds before each swing")
    parser.add_argument("--after", type=lambda x: number(x, strict=True), default=0.9, help="Seconds after each swing")
    parser.add_argument("--min-speed", type=lambda x: number(x, strict=True), default=2.8,
                        help="Minimum wrist speed in torso lengths/second")
    parser.add_argument("--min-gap", type=lambda x: number(x, 0.4), default=1.8, help="Minimum seconds between strokes")
    parser.add_argument("--min-player-height", type=lambda x: number(x), default=None,
                        help="Minimum person height as fraction of frame (default: near 0.2, far 0.05)")
    parser.add_argument("--limit", type=lambda x: number(x, strict=True), help="Analyze only this many seconds")
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.min_player_height is not None and args.min_player_height > 1:
        parser.error("--min-player-height must be between 0 and 1")
    if args.min_player_height is None:
        args.min_player_height = 0.2 if args.player == "near" else 0.05
    if args.roi and not (all(math.isfinite(v) for v in args.roi) and
                         0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error("--roi must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    if args.events_csv and args.analyze_only:
        parser.error("--events-csv cannot be combined with --analyze-only")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        source = args.input.resolve()
        metadata = probe_video(source)
        if not args.analyze_only and not shutil.which("ffmpeg"):
            raise ValueError("FFmpeg is required for export. Install it (macOS: brew install ffmpeg).")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "strokes.csv"
        labels = ["forehand", "backhand"] if args.stroke == "both" else [args.stroke]
        if not args.analyze_only:
            for label in labels:
                output = args.output_dir / f"{label}s.mp4"
                if output.resolve() == source or (output.exists() and not args.overwrite):
                    raise ValueError(f"Cannot replace {output}; choose another directory or pass --overwrite.")
        if args.events_csv:
            events = read_events(args.events_csv, metadata["duration"])
        else:
            if csv_path.exists() and not args.overwrite:
                raise ValueError(f"{csv_path} exists. Use --events-csv to export it, or --overwrite to analyze again.")
            print(f"Experimental pose classifier: {args.player} player, {args.handedness}-handed. Review strokes.csv.", flush=True)
            samples = collect_poses(source, metadata, args, args.output_dir / "poses.json")
            duration = min(metadata["duration"], args.limit) if args.limit else metadata["duration"]
            events = detect_strokes(samples, duration, args.handedness, args.player, args.mirrored,
                                    args.before, args.after, args.min_speed, args.min_gap,
                                    metadata["height"], args.min_player_height)
            write_events(csv_path, events)
            print(f"Saved {csv_path}", flush=True)
        for label in ["forehand", "backhand", "unknown"]:
            print(f"{label}: {sum(e['stroke'] == label for e in events)}", flush=True)
        if not args.analyze_only:
            for label in labels:
                export_compilation(source, events, label, args.output_dir / f"{label}s.mp4", args.overwrite)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
