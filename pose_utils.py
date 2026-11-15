"""YOLO pose helpers used by the stroke highlights pipeline."""

import numpy as np


def load_yolo(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for stroke detection. Install the project requirements "
            "or run: pip install ultralytics"
        ) from exc

    return YOLO(model_path)



def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)



def extract_pose_detections(result, max_poses):
    if result.keypoints is None:
        return []

    keypoints = to_numpy(result.keypoints.data)
    if keypoints is None or len(keypoints) == 0:
        return []

    boxes = to_numpy(result.boxes.xyxy) if result.boxes is not None else None
    scores = to_numpy(result.boxes.conf) if result.boxes is not None else None

    detections = []
    for idx, person_keypoints in enumerate(keypoints):
        score = float(scores[idx]) if scores is not None and idx < len(scores) else mean_keypoint_conf(person_keypoints)
        box = boxes[idx] if boxes is not None and idx < len(boxes) else None
        detections.append(
            {
                "box": box,
                "keypoints": person_keypoints,
                "score": score,
            }
        )

    detections.sort(key=lambda detection: detection["score"], reverse=True)
    return detections[:max_poses]



def mean_keypoint_conf(keypoints):
    if keypoints.shape[1] < 3:
        return 1.0
    valid = keypoints[:, 2] > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(keypoints[valid, 2]))



def yolo_device_arg(device):
    if device == "auto":
        return None
    return device
