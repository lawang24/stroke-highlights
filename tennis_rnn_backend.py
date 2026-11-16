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
CLASSES = ("backhand", "forehand", "neutral", "serve")
WINDOW = 30
FPS = 30.0


def pose_features(points, left_handed=False):
    """Fixed 26 columns, even when a joint has low confidence."""
    features = np.asarray(points, dtype=np.float32)[JOINTS, :2].copy()
    if left_handed:
        features[:, 1] = 1 - features[:, 1]
    return features.reshape(26)


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
    identity = {"version": 1, "source": str(source), "size": source.stat().st_size,
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
    features, valid, times = [], [], []
    for t, frame in iter_samples(source, metadata, duration):
        if args.mirrored:
            frame = cv2.flip(frame, 1)
        points = tracker.extract(frame)
        valid.append(points is not None)
        features.append(pose_features(points, args.handedness == "left") if points is not None else np.zeros(26))
        times.append(t)
        if len(times) % 300 == 0:
            print(f"Tennis RNN pose extraction: {t:.1f} / {duration:.1f} seconds", flush=True)
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
        centers.extend((times[i] + times[i+WINDOW-1]) / 2 for i in batch_starts)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 4)
    centers = np.asarray(centers)
    np.savez_compressed(cache, identity=json.dumps(identity), times=centers, scores=scores)
    print(f"Tennis RNN: {sum(valid)}/{len(valid)} valid poses, {len(centers)} classified windows", flush=True)
    return centers, scores


def decode_scores(times, scores, duration, threshold=.98, min_gap=2.0, before=1.2, after=.9):
    """One peak per confident class run, followed by cross-class suppression."""
    if not 0 < threshold <= 1 or not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("Invalid Tennis RNN threshold or minimum gap")
    times, scores = np.asarray(times), np.asarray(scores)
    if scores.shape != (len(times), 4) or not np.isfinite(scores).all():
        raise ValueError("Expected four finite probabilities per Tennis RNN window")
    candidates, run = [], []
    def finish():
        if run:
            candidates.append(max(run, key=lambda p: p[2]))
            run.clear()
    for t, probabilities in zip(times, scores):
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        if label == 2 or confidence <= threshold:
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
