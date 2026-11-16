import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stroke_highlights import compilation_ranges, export_compilation, read_events


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
