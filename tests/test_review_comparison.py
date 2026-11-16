import csv
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from review_comparison import byte_range, handler_for, load_comparison, ordered_matches


class ReviewTests(unittest.TestCase):
    def test_video_ranges(self):
        for header, expected in [(None, (0, 99, False)), ('bytes=10-19', (10, 19, True)),
                                 ('bytes=90-', (90, 99, True)), ('bytes=-20', (80, 99, True)),
                                 ('bytes=90-200', (90, 99, True))]:
            self.assertEqual(byte_range(header, 100), expected)
        for header in ('bytes=-0', 'bytes=100-', 'bytes=20-10', 'bytes=0-1,3-4', 'invalid'):
            with self.assertRaises(ValueError): byte_range(header, 100)

    def test_server_seeks_without_exposing_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / 'video.mp4'; video.write_bytes(b'0123456789')
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(video, {'rows': []}))
            thread = Thread(target=server.serve_forever, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(Request(url + '/video', headers={'Range': 'bytes=3-6'})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers['Content-Range'], 'bytes 3-6/10')
                    self.assertEqual(response.read(), b'3456')
                with urlopen(Request(url + '/video', method='HEAD')) as response:
                    self.assertEqual(response.headers['Content-Length'], '10')
                    self.assertEqual(response.read(), b'')
                with urlopen(url + '/api/comparison') as response:
                    self.assertEqual(json.load(response), {'rows': []})
                for path in ('/../pyproject.toml', '/outputs/', '/video/../video.mp4'):
                    with self.assertRaises(HTTPError) as error: urlopen(url + path)
                    self.assertEqual(error.exception.code, 404)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url + '/video', headers={'Range': 'bytes=100-'}))
                self.assertEqual(error.exception.code, 416)
            finally:
                server.shutdown(); server.server_close(); thread.join()

    def test_candidate_keeps_existing_review_identity_and_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); video = root / 'video.mp4'; video.write_bytes(b'video')
            directory = root / 'comparison'; directory.mkdir()
            source = {'name': video.name, 'size': 5, 'duration': 10}
            def save(name, events):
                dest = directory / name; dest.mkdir(exist_ok=True)
                (dest / 'stroke-review.json').write_text(json.dumps({'source': source, 'events': events, 'settings': {}}))
            save('heuristic', [{'time': 1, 'stroke': 'backhand'}])
            save('adaspot', [{'time': 1.1, 'stroke': 'forehand'}])
            (directory / 'comparison.csv').write_text('status,heuristic_time,adaspot_time\ndifferent_label,1,1.1\n')
            _, base = load_comparison(directory)
            save('tennis-rnn', [{'time': 1.2, 'stroke': 'forehand'}, {'time': 8, 'stroke': 'backhand'}])
            _, updated = load_comparison(directory)
            self.assertEqual(base['reviewId'], updated['reviewId'])
            self.assertEqual(updated['rows'][0]['id'], base['rows'][0]['id'])
            self.assertEqual(updated['rows'][0]['tennis-rnn']['stroke'], 'forehand')
            self.assertEqual(updated['rows'][1]['status'], 'tennis_rnn_only')
            source['size'] = 6; save('tennis-rnn', [])
            with self.assertRaisesRegex(ValueError, 'different video'): load_comparison(directory)

    def test_matching_prioritizes_number_of_matches(self):
        rows = [{'time': 1}, {'time': 1.6}]
        events = [{'time': 1.3}, {'time': .6}]
        self.assertEqual(set(ordered_matches(rows, events, .5)), {(0, 1), (1, 0)})
        self.assertEqual(ordered_matches([], events, .5), [])


if __name__ == '__main__': unittest.main()
