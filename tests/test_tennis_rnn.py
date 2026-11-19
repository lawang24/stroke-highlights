import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stroke_highlights import main, parse_args
from tennis_rnn_backend import (
    pose_features,
    decode_scores,
    iter_samples,
    collect_scores,
    smooth_arm_occlusions,
    suppress_post_backhand_recovery,
)


class TennisRNNTests(unittest.TestCase):
    def test_feature_order_and_left_hand_transform(self):
        points = np.arange(51, dtype=np.float32).reshape(17, 3) / 51
        points[:, 2] = 0  # Low confidence must not change the feature dimension.
        expected = points[[0, *range(5, 17)], :2]
        np.testing.assert_array_equal(pose_features(points), expected.reshape(26))
        reflected = pose_features(points, True).reshape(13, 2)
        np.testing.assert_array_equal(reflected[:, 0], expected[:, 0])
        np.testing.assert_allclose(reflected[:, 1], 1 - expected[:, 1])

    def test_neutral_serve_and_confident_runs(self):
        times = np.arange(8) / 30 + 1
        scores = np.array([[.001, .997, .001, .001], [.001, .995, .003, .001],
                           [.001, .001, .997, .001], [.001, .001, .001, .997],
                           [.001, .001, .001, .997], [.997, .001, .001, .001],
                           [.2, .3, .3, .2], [.2, .3, .3, .2]])
        events = decode_scores(times, scores, 2, min_gap=.01)
        self.assertEqual([e['stroke'] for e in events], ['forehand', 'unknown', 'backhand'])
        self.assertFalse(events[1]['include'])
        self.assertIn('serve', events[1]['reason'])
        self.assertEqual(events[0]['time'], 1)
        self.assertEqual(events[-1]['end'], 2)
        self.assertEqual(len(decode_scores(times, scores, 2, min_gap=2)), 1)

    def test_missing_pose_gap_splits_runs(self):
        events = decode_scores([1, 4], np.array([[.999, 0, .001, 0]] * 2), 6)
        self.assertEqual([e['time'] for e in events], [1, 4])
        self.assertEqual(decode_scores([], np.empty((0, 4)), 1), [])

    def test_sampling_is_30hz_at_low_and_high_source_fps(self):
        class Capture:
            def __init__(self): self.index, self.closed = -1, False
            def grab(self): self.index += 1; return True
            def retrieve(self): return True, self.index
            def release(self): self.closed = True
        for fps in (15, 30, 60):
            cap = Capture()
            with patch('tennis_rnn_backend.cv2.VideoCapture', return_value=cap):
                samples = list(iter_samples('test', {'fps': fps, 'frames': fps * 2}, 2))
            self.assertEqual(len(samples), 60)
            self.assertAlmostEqual(samples[-1][0], 59 / 30)
            self.assertEqual([s[1] for s in samples], [int(i * fps / 30) for i in range(60)])
            self.assertTrue(cap.closed)

    def test_cache_rejects_different_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'video.mp4'; source.write_bytes(b'video')
            cache = Path(folder) / 'scores.npz'
            np.savez(cache, identity=json.dumps({'version': -1}))
            args = SimpleNamespace(limit=None, mirrored=False, handedness='right', player='near', min_player_height=.2)
            with self.assertRaisesRegex(ValueError, 'cache settings/input differ'):
                collect_scores(source, {'duration': 2, 'fps': 30}, args, cache)

    def test_default_cli_and_output_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'video.mp4'; source.write_bytes(b'video')
            output = Path(folder) / 'results'
            event = dict(id=1, time=1, start=0, end=2, stroke='forehand', score=.999, include=True, reason='test')
            argv = [str(source), '--output-dir', str(output), '--analyze-only']
            self.assertEqual(parse_args(argv).classifier, 'tennis-rnn')
            with patch('stroke_highlights.probe_video', return_value={'duration': 4, 'height': 1080}),                     patch('tennis_rnn_backend.analyze_tennis_rnn', return_value=[event]) as rnn,                     contextlib.redirect_stdout(io.StringIO()):
                main(argv)
                review = json.loads((output / 'stroke-review.json').read_text())
                self.assertEqual(review['classifier'], 'tennis-rnn')
                self.assertEqual(review['events'][0]['stroke'], 'forehand')
                with self.assertRaisesRegex(SystemExit, 'Cannot replace'): main(argv)
                self.assertEqual(rnn.call_count, 1)

    def test_cli_rejects_unsupported_settings(self):
        base = ['video.mp4', '--output-dir', 'test', '--classifier', 'tennis-rnn']
        for extra in (['--player', 'far'], ['--tennis-rnn-threshold', '1.1'], ['--tennis-rnn-min-gap', '0'], ['--classifier', 'heuristic'], ['--classifier', 'adaspot'], ['--classifier', 'compare']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): parse_args(base + extra)

    def test_sustained_backhand_run_and_arm_occlusion_smoothing(self):
        times = np.arange(15) / 30 + 1.0
        scores = np.array([[.88, .01, .10, .01]] * 9 + [[.01, .01, .97, .01]] * 4 + [[.88, .01, .10, .01]] * 2)
        events = decode_scores(times, scores, 5.0, min_gap=0.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['stroke'], 'backhand')

        base_pose = np.ones((17, 3), dtype=np.float32) * 0.5
        seq = [base_pose.copy() for _ in range(7)]
        for idx, val in enumerate(np.linspace(0.2, 0.8, 7)):
            seq[idx][9, 0] = val
            seq[idx][9, 2] = 0.8
        seq[3][9, 0] = 0.95
        seq[3][9, 2] = 0.15
        smoothed = smooth_arm_occlusions(seq, [True] * 7)
        self.assertAlmostEqual(float(smoothed[3][9, 0]), 0.5, places=4)

    def test_post_backhand_ready_position_recovery_is_reclassified_as_neutral(self):
        # Backhand (9 frames @ 0.88) followed directly by ready-position recovery (10 frames of fake forehand @ 0.95)
        # without an intervening >=4-frame neutral reset, then 5 neutral frames, then a real forehand.
        times = np.arange(34) / 30 + 1.0
        scores = np.array(
            [[.88, .02, .09, .01]] * 9 +      # True backhand
            [[.05, .92, .02, .01]] * 10 +     # Post-backhand recovery into ready position (spurious FH)
            [[.01, .01, .97, .01]] * 5 +      # Stable neutral ready state (>= 4 frames)
            [[.01, .99, .00, .00]] * 10       # Subsequent real forehand
        )
        cleaned = suppress_post_backhand_recovery(times, scores)
        # Spurious forehand during frames 9..18 is folded into neutral (class 2)
        self.assertTrue(np.all(np.argmax(cleaned[9:19], axis=1) == 2))
        self.assertTrue(np.allclose(cleaned[9:19, 1], 0.0))
        # Even with a very small min_gap (0.1s), only the true backhand and real forehand are emitted
        events = decode_scores(times, scores, 5.0, min_gap=0.1)
        self.assertEqual([e['stroke'] for e in events], ['backhand', 'forehand'])
        self.assertAlmostEqual(events[0]['time'], 1.0, places=2)
        self.assertGreater(events[1]['time'], 1.7)


if __name__ == '__main__': unittest.main()
