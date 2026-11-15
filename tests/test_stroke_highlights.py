import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stroke_highlights import (
    compilation_ranges, detect_strokes, export_compilation, read_events, select_player,
)


def synthetic_swing(direction=-1, overhead=False, walking=False):
    samples = []
    for t in np.arange(0, 4, 1 / 15):
        points = np.zeros((17, 3))
        points[:, 2] = 0.99
        points[5, :2], points[6, :2] = (280, 200), (320, 200)
        points[11, :2], points[12, :2] = (285, 300), (315, 300)
        x = 0.8 * direction * np.tanh((t - 2) * 8) if not walking else 0.5
        points[9, :2] = points[10, :2] = (300 + x * 100, 150 if overhead else 270)
        if walking:
            points[:, 0] += t * 150
        samples.append({"time": float(t), "segment": 0, "keypoints": points.tolist()})
    return samples


class DetectionTests(unittest.TestCase):
    def test_forehand_and_backhand(self):
        for direction, label in [(-1, "forehand"), (1, "backhand")]:
            events = detect_strokes(synthetic_swing(direction), 4)
            self.assertEqual([e["stroke"] for e in events], [label])
            self.assertAlmostEqual(events[0]["time"], 2, places=1)

    def test_handedness_view_and_mirror(self):
        for settings in [{"handedness": "left"}, {"player": "far"}, {"mirrored": True}]:
            events = detect_strokes(synthetic_swing(1), 4, **settings)
            self.assertEqual([e["stroke"] for e in events], ["forehand"])

    def test_overhead_excluded_and_walking_removed(self):
        events = detect_strokes(synthetic_swing(overhead=True), 4)
        self.assertEqual([e["stroke"] for e in events], ["unknown"])
        self.assertFalse(events[0]["include"])
        self.assertEqual(detect_strokes(synthetic_swing(walking=True), 4), [])

    def test_lowering_arm_is_not_a_second_stroke(self):
        samples = synthetic_swing(1)
        for sample in samples:
            sample["keypoints"][10][1] = 220 + 60 * np.tanh((sample["time"] - 2) * 8)
        events = detect_strokes(samples, 4)
        self.assertEqual([e["stroke"] for e in events], ["unknown"])
        self.assertFalse(events[0]["include"])

    def test_small_distant_player_is_excluded(self):
        samples = synthetic_swing()
        for sample in samples:
            sample["box"] = [200, 200, 300, 300]
        self.assertEqual(detect_strokes(samples, 4, frame_height=1080), [])

    def test_missing_keypoints_do_not_create_a_swing(self):
        samples = synthetic_swing()
        for sample in samples:
            if 1.7 < sample["time"] < 2.3:
                sample["keypoints"] = None
        self.assertEqual(detect_strokes(samples, 4), [])

    def test_brief_wrist_occlusion_preserves_swing(self):
        samples = synthetic_swing()
        for sample in samples:
            if 1.9 < sample["time"] < 2.1:
                sample["keypoints"][10][2] = 0.05
        self.assertEqual([e["stroke"] for e in detect_strokes(samples, 4)], ["forehand"])

    def test_identity_change_does_not_create_a_swing(self):
        samples = synthetic_swing()
        for sample in samples:
            if sample["time"] >= 2:
                sample["segment"] = 1
        self.assertEqual(detect_strokes(samples, 4), [])

    def test_player_tracking_rejects_switch_to_distant_person(self):
        old = np.array([20, 30, 100, 200])
        bystander = {"box": np.array([800, 100, 900, 220])}
        self.assertIsNone(select_player([bystander], old, (720, 1280), "near", [0, 0, 1, 1]))

    def test_roi_excludes_neighboring_courts(self):
        player = {"box": np.array([20, 30, 100, 200])}
        bystander = {"box": np.array([800, 100, 1100, 700])}
        self.assertIs(select_player([player, bystander], None, (720, 1280), "near", [0, 0, .5, 1]), player)


class ExportTests(unittest.TestCase):
    def test_ranges_filter_sort_and_merge_without_duplicate_footage(self):
        events = [{"start": a, "end": b, "stroke": label, "include": include}
                  for a, b, label, include in [(3, 5, "forehand", True), (1, 4, "forehand", True),
                                              (7, 8, "backhand", True), (9, 10, "forehand", False)]]
        self.assertEqual(compilation_ranges(events, "forehand"), [(1, 5)])

    def test_csv_validates_rows_and_reads_false_as_false(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.csv"
            path.write_text("start,end,stroke,include\n0,1,forehand,false\n")
            self.assertFalse(read_events(path, 2)[0]["include"])
            for row in ["nan,1,forehand,true", "0,9,forehand,true", "0,1,typo,true", "0,1,forehand,maybe",
                        "0,1", "2.01,2.02,forehand,true"]:
                path.write_text("start,end,stroke,include\n" + row + "\n")
                with self.assertRaisesRegex(ValueError, "CSV line 2"):
                    read_events(path, 2)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_export_audio_silent_input_and_empty_selection(self):
        with tempfile.TemporaryDirectory(prefix="tennis test '") as temp:
            root = Path(temp)
            for with_audio in [True, False]:
                source = root / f"source-{with_audio}.mp4"
                cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=30:d=4"]
                if with_audio:
                    cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac"]
                subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
                events = [{"start": a, "end": b, "stroke": "forehand", "include": True}
                          for a, b in [(0.5, 1), (2, 3)]]
                dest = root / f"out-{with_audio}.mp4"
                export_compilation(source, events, "forehand", dest)
                probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                                           "-show_format", "-of", "json", str(dest)]))
                self.assertAlmostEqual(float(probe["format"]["duration"]), 1.5, delta=0.25)
                self.assertEqual(any(s["codec_type"] == "audio" for s in probe["streams"]), with_audio)
                video = next(s for s in probe["streams"] if s["codec_type"] == "video")
                self.assertEqual(video["r_frame_rate"], "30/1")
                decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dest), "-f", "null", "-"],
                                         capture_output=True, text=True)
                self.assertEqual(decoded.returncode, 0)
                self.assertEqual(decoded.stderr, "")
                with self.assertRaises(ValueError):
                    export_compilation(source, events, "forehand", dest)
                self.assertIsNone(export_compilation(source, events, "backhand", root / "empty.mp4"))
                self.assertFalse((root / "empty.mp4").exists())
                export_compilation(source, [], "forehand", dest, overwrite=True)
                self.assertFalse(dest.exists())
                with self.assertRaises(ValueError):
                    export_compilation(source, [], "forehand", source, overwrite=True)


if __name__ == "__main__":
    unittest.main()
