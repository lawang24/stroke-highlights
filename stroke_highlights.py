"""Local Tennis RNN stroke detection and video compilation."""

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


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
    parser.add_argument("--classifier", choices=["tennis-rnn"], default="tennis-rnn",
                        help="Tennis RNN is the only classifier (also the default)")
    parser.add_argument("--tennis-rnn-threshold", type=lambda x: number(x, strict=True), default=.98,
                        help="Minimum class probability; not calibrated accuracy")
    parser.add_argument("--tennis-rnn-min-gap", type=lambda x: number(x, strict=True), default=2.0)
    parser.add_argument("--events-csv", type=Path, help="Export an edited CSV, skipping inference")
    parser.add_argument("--analyze-only", action="store_true", help="Write proposals without exporting video")
    parser.add_argument("--overwrite", action="store_true", help="Replace events/compilations; retain compatible score cache")
    parser.add_argument("--player", choices=["near"], default="near", help="Only the near player is supported")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument("--mirrored", action="store_true", help="Source video is horizontally mirrored")
    parser.add_argument("--before", type=lambda x: number(x), default=1.2)
    parser.add_argument("--after", type=lambda x: number(x, strict=True), default=.9)
    parser.add_argument("--min-player-height", type=lambda x: number(x), default=.2)
    parser.add_argument("--limit", type=lambda x: number(x, strict=True), help="Analyze only this many seconds")
    args = parser.parse_args(argv)
    if args.tennis_rnn_threshold > 1:
        parser.error("--tennis-rnn-threshold must be in (0, 1]")
    if args.min_player_height > 1:
        parser.error("--min-player-height must be between 0 and 1")
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
        labels = ["forehand", "backhand"] if args.stroke == "both" else [args.stroke]
        outputs = [] if args.events_csv else [args.output_dir / "strokes.csv", args.output_dir / "stroke-review.json"]
        if not args.analyze_only:
            outputs.extend(args.output_dir / f"{label}s.mp4" for label in labels)
        for output in outputs:
            if output.resolve() == source or (output.exists() and not args.overwrite):
                raise ValueError(f"Cannot replace {output}; choose another directory or pass --overwrite.")
        if args.events_csv:
            events = read_events(args.events_csv, metadata["duration"])
        else:
            from tennis_rnn_backend import analyze_tennis_rnn
            from review_format import write_browser_review
            print("Classifier: Tennis RNN; near player. Review strokes.csv.", flush=True)
            events = analyze_tennis_rnn(source, metadata, args, args.output_dir)
            write_events(args.output_dir / "strokes.csv", events)
            settings = {key: value for key, value in vars(args).items()
                        if not isinstance(value, Path) and key not in {"overwrite", "analyze_only"}}
            write_browser_review(args.output_dir / "stroke-review.json", source, metadata, events, "tennis-rnn", settings)
            print(f"Saved shot lists in {args.output_dir}", flush=True)
        for label in ["forehand", "backhand", "unknown"]:
            print(f"{label}: {sum(e['stroke'] == label for e in events)}", flush=True)
        if not args.analyze_only:
            for label in labels:
                export_compilation(source, events, label, args.output_dir / f"{label}s.mp4", args.overwrite)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
