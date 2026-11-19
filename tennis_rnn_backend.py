"""MoveNet Lightning + the pretrained Antoine Keller 30-frame GRU classifier.

Input compatibility matters: upstream used OpenCV BGR, 192px letterboxing,
crop-local (y,x) coordinates, no eyes/ears, and x reflection for left-handers.
This adapter keeps those conventions. It timestamps windows at their center
and consolidates confident runs, rather than reporting the trailing frame.
"""

import json
import math

import cv2
import numpy as np

from scripts.setup_tennis_rnn import ROOT, ARTIFACTS, verify_models

JOINTS = [0, *range(5, 17)]
ARM_JOINTS = (7, 8, 9, 10)
CLASSES = ("backhand", "forehand", "neutral", "serve")
WINDOW = 30
FPS = 30.0
SUSTAINED_MIN_FRAMES = 8
SUSTAINED_THRESHOLD = 0.80


def pose_features(points, left_handed=False):
    """Fixed 26 columns, even when a joint has low confidence."""
    features = np.asarray(points, dtype=np.float32)[JOINTS, :2].copy()
    if left_handed:
        features[:, 1] = 1 - features[:, 1]
    return features.reshape(26)


def smooth_arm_occlusions(points_list, valid_list, min_conf=0.35, low_conf=0.22, max_jump=0.08, max_gap=6):
    """Interpolate occluded elbow/wrist keypoints across short self-occlusion bursts."""
    n = len(points_list)
    out = [np.asarray(p, dtype=np.float32).copy() if p is not None else None for p in points_list]
    i = 0
    while i < n:
        if not valid_list[i] or out[i] is None:
            i += 1
            continue
        j = i
        while j < n and valid_list[j] and out[j] is not None:
            j += 1
        seg_len = j - i
        if seg_len >= 5:
            seg = np.stack(out[i:j])
            idx = np.arange(seg_len)
            for joint in ARM_JOINTS:
                confs = seg[:, joint, 2]
                xy = seg[:, joint, :2]
                jump = np.zeros(seg_len, dtype=np.float32)
                jump[1:] = np.maximum(jump[1:], np.linalg.norm(xy[1:] - xy[:-1], axis=1))
                jump[:-1] = np.maximum(jump[:-1], np.linalg.norm(xy[:-1] - xy[1:], axis=1))
                bad = (confs < min_conf) & ((confs < low_conf) | (jump > max_jump))
                good = np.where(~bad)[0]
                if 2 <= len(good) < seg_len:
                    dist = np.min(np.abs(idx[:, None] - good[None, :]), axis=1)
                    mask = bad & (dist <= max_gap)
                    if mask.any():
                        for coord in (0, 1):
                            seg[mask, joint, coord] = np.interp(idx, good, seg[good, joint, coord])[mask]
            for k in range(seg_len):
                out[i + k] = seg[k]
        i = j
    return out


def suppress_post_backhand_recovery(times, scores, bh_trigger=0.35, neutral_reset=0.75,
                                    reset_frames=4, max_recovery_sec=1.5):
    """Reclassify post-backhand return-to-ready-position forehand spikes as neutral.

    After finishing a backhand follow-through (hands on the dominant side), returning
    the racquet to the center ready position moves the wrists in the forehand direction.
    Until neutral ready position is re-established (>= reset_frames consecutive windows
    with neutral >= neutral_reset, or max_recovery_sec elapsed), fold spurious forehand
    probability into neutral.
    """
    times = np.asarray(times, dtype=np.float64)
    out = np.asarray(scores, dtype=np.float32).copy()
    if len(times) == 0:
        return out
    in_bh_recovery = False
    last_bh_time = -1e9
    neutral_streak = 0
    prev_t = None
    for i, (t, row) in enumerate(zip(times, out)):
        if prev_t is not None and t - prev_t > 0.25:
            in_bh_recovery = False
            neutral_streak = 0
        prev_t = float(t)
        bh, fh, ne, _ = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        label = int(np.argmax(row))
        if label == 0 and bh >= bh_trigger:
            in_bh_recovery = True
            last_bh_time = float(t)
            neutral_streak = 0
            if fh > 0.0:
                out[i, 2] = ne + fh
                out[i, 1] = 0.0
            continue
        if in_bh_recovery:
            if float(t) - last_bh_time > max_recovery_sec:
                in_bh_recovery = False
                neutral_streak = 0
                continue
            if ne >= neutral_reset and label == 2:
                neutral_streak += 1
                if neutral_streak >= reset_frames:
                    in_bh_recovery = False
                    neutral_streak = 0
                    continue
            else:
                neutral_streak = 0
            if fh > 0.0:
                out[i, 2] = ne + fh
                out[i, 1] = 0.0
    return out


def load_classifier(tf):
    # Rebuild only the inspected architecture, then load numeric weights. The
    # old H5 config contains `time_major`, removed from modern Keras GRU.
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW, 26)),
        tf.keras.layers.GRU(24, dropout=.1, reset_after=True),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.load_weights(ROOT / "tennis_rnn.h5")
    return model


class MoveNetTracker:
    def __init__(self, tf, min_height):
        self.tf = tf
        self.interpreter = tf.lite.Interpreter(model_path=str(ROOT / "movenet.tflite"), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.crop = None
        self.min_height = min_height

    def extract(self, frame):
        height, width = frame.shape[:2]
        x, y, w, h = self.crop or (0, 0, width, height)
        image = self.tf.image.resize_with_pad(frame[None, y:y+h, x:x+w], 192, 192)
        self.interpreter.set_tensor(self.input["index"], np.asarray(image, dtype=self.input["dtype"]))
        self.interpreter.invoke()
        points = self.interpreter.get_tensor(self.output["index"]).reshape(17, 3)
        # Invert the letterbox to track in original image pixels. GRU features
        # stay in the normalized letterboxed crop coordinates it was trained on.
        pixels = points[:, :2] * max(w, h) + [y - (max(w, h)-h)/2, x - (max(w, h)-w)/2]
        body = pixels[JOINTS]
        low, high = body.min(axis=0), body.max(axis=0)
        valid = (np.isfinite(points).all() and points[JOINTS, 2].mean() >= .3
                 and high[0] / height >= .55
                 and (high[0] - low[0]) / height >= self.min_height)
        if not valid:
            self.crop = None
            return None
        # The training extractor used a 1.3× square around all keypoints.
        extent = pixels.max(axis=0) - pixels.min(axis=0)
        side = int(max(extent) * 1.3)
        if side < 10 or side >= min(width, height):
            self.crop = None
        else:
            center = (pixels.max(axis=0) + pixels.min(axis=0)) / 2
            x = int(np.clip(center[1] - side / 2, 0, width - side))
            y = int(np.clip(center[0] - side / 2, 0, height - side))
            self.crop = (x, y, side, side)
        return points


def iter_samples(source, metadata, duration):
    """Sample exactly 30Hz (repeat frames for low FPS), retaining source time."""
    cap = cv2.VideoCapture(str(source))
    decoded = -1
    count = min(math.ceil(duration * FPS), math.ceil(metadata["frames"] * FPS / metadata["fps"]))
    try:
        for sample in range(count):
            target = min(metadata["frames"] - 1, math.floor(sample * metadata["fps"] / FPS))
            while decoded < target:
                if not cap.grab():
                    raise ValueError(f"Video decoding stopped at frame {decoded + 1}; normalize to constant FPS first")
                decoded += 1
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {decoded}")
            yield sample / FPS, frame
    finally:
        cap.release()


def collect_scores(source, metadata, args, cache):
    duration = min(metadata["duration"], args.limit) if args.limit else metadata["duration"]
    identity = {"version": 3, "source": str(source), "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns, "duration": duration,
                "fps": metadata["fps"], "mirrored": args.mirrored,
                "handedness": args.handedness, "player": args.player,
                "min_player_height": args.min_player_height,
                "models": {name: pair[1] for name, pair in ARTIFACTS.items()}}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError(f"Tennis RNN cache settings/input differ: {cache}. Choose a new output directory.")
            print(f"Reusing Tennis RNN scores: {cache}", flush=True)
            return saved["times"], saved["scores"]
    verify_models()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("Install Tennis RNN dependencies: uv sync") from exc
    # Keep this small optional model on CPU; do not compete with the video UI.
    tf.config.set_visible_devices([], "GPU")
    tracker = MoveNetTracker(tf, args.min_player_height)
    raw_points, valid, times = [], [], []
    for t, frame in iter_samples(source, metadata, duration):
        if args.mirrored:
            frame = cv2.flip(frame, 1)
        points = tracker.extract(frame)
        raw_points.append(points)
        valid.append(points is not None)
        times.append(t)
        if len(times) % 300 == 0:
            print(f"Tennis RNN pose extraction: {t:.1f} / {duration:.1f} seconds", flush=True)
    raw_valid_count = sum(valid)
    total_frames = len(valid)
    smoothed_points = smooth_arm_occlusions(raw_points, valid)
    left_handed = args.handedness == "left"
    features = [pose_features(p, left_handed) if p is not None else np.zeros(26, dtype=np.float32)
                for p in smoothed_points]
    # Complete trailing 30-frame windows at the clip boundary so strokes in the
    # final 0.5s are not cut off mid-swing before follow-through frames arrive.
    if len(features) >= WINDOW and all(valid[-WINDOW:]):
        tail_pad = WINDOW // 2
        for _ in range(tail_pad):
            features.append(features[-1].copy())
            valid.append(True)
            times.append(times[-1] + 1.0 / FPS)
    model = load_classifier(tf)
    features = np.asarray(features, dtype=np.float32)
    times = np.asarray(times)
    centers, scores = [], []
    # Batch the lightweight GRU, keeping only windows with a full tracked pose.
    starts = [i for i in range(max(0, len(features) - WINDOW + 1)) if all(valid[i:i+WINDOW])]
    predict = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, WINDOW, 26], tf.float32)])
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset:offset+128]
        batch = np.stack([features[i:i+WINDOW] for i in batch_starts])
        scores.extend(predict(batch).numpy())
        centers.extend(min(duration, (times[i] + times[i+WINDOW-1]) / 2) for i in batch_starts)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 4)
    centers = np.asarray(centers)
    scores = suppress_post_backhand_recovery(centers, scores)
    np.savez_compressed(cache, identity=json.dumps(identity), times=centers, scores=scores)
    print(f"Tennis RNN: {raw_valid_count}/{total_frames} valid poses, {len(centers)} classified windows", flush=True)
    return centers, scores


def decode_scores(times, scores, duration, threshold=.98, min_gap=2.0, before=1.2, after=.9,
                  sustained_min_frames=SUSTAINED_MIN_FRAMES, sustained_threshold=SUSTAINED_THRESHOLD):
    """One peak per confident class run (or sustained multi-frame run), followed by cross-class suppression."""
    if not 0 < threshold <= 1 or not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("Invalid Tennis RNN threshold or minimum gap")
    times, scores = np.asarray(times), np.asarray(scores)
    if scores.shape != (len(times), 4) or not np.isfinite(scores).all():
        raise ValueError("Expected four finite probabilities per Tennis RNN window")
    scores = suppress_post_backhand_recovery(times, scores)
    eff_gate = min(threshold, sustained_threshold)
    candidates, run = [], []
    def finish():
        if run:
            peak = max(run, key=lambda p: p[2])
            if peak[2] > threshold or (len(run) >= sustained_min_frames and peak[2] > eff_gate):
                candidates.append(peak)
            run.clear()
    for t, probabilities in zip(times, scores):
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        if label == 2 or confidence <= eff_gate:
            finish()
            continue
        if run and (label != run[-1][1] or t - run[-1][0] > 1.5/FPS):
            finish()
        run.append((float(t), label, confidence))
    finish()
    selected = []
    for candidate in sorted(candidates, key=lambda p: p[2], reverse=True):
        if all(abs(candidate[0] - other[0]) >= min_gap for other in selected):
            selected.append(candidate)
    events = []
    for index, (t, label, score) in enumerate(sorted(selected), 1):
        stroke = CLASSES[label] if label != 3 else "unknown"
        events.append({"id": index, "time": round(t, 3), "start": round(max(0, t-before), 3),
                       "end": round(min(duration, t+after), 3), "stroke": stroke,
                       "score": round(score, 4), "include": stroke != "unknown",
                       "reason": f"Tennis RNN: {CLASSES[label]}; 30-frame MoveNet sequence"})
    return events


def analyze_tennis_rnn(source, metadata, args, directory):
    if args.player != "near":
        raise ValueError("Tennis RNN currently supports only --player near")
    times, scores = collect_scores(source, metadata, args, directory / "tennis-rnn-scores.npz")
    duration = min(metadata["duration"], args.limit) if args.limit else metadata["duration"]
    return decode_scores(times, scores, duration, args.tennis_rnn_threshold,
                         args.tennis_rnn_min_gap, args.before, args.after)
