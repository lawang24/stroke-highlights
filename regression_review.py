"""Build and review a local, shuffled tennis regression corpus (never auto-labels)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import re
import subprocess
import threading
from urllib.parse import urlsplit

from review_comparison import byte_range

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'web/dataset-review'
DEFAULT = ROOT / 'outputs/regression-review-v1'
LABELS = {'forehand', 'backhand', 'no_shot', 'unsure'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def probe(path):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    num, den = map(int, stream['avg_frame_rate'].split('/'))
    return {'duration': float(stream.get('duration', info['format']['duration'])), 'fps': num/den,
            'frames': int(stream.get('nb_frames') or round(float(info['format']['duration'])*num/den)),
            'width': stream['width'], 'height': stream['height'], 'codec': stream['codec_name']}


def build_dataset(directory, seed=20261115):
    """Sample existing sources. Prior labels select clips but never become truth."""
    if directory.exists():
        raise ValueError(f'{directory} already exists; choose a new directory to preserve reviews')
    rng = random.Random(seed)
    source_paths = [ROOT/'outputs/adaspot-comparison-input.mp4', ROOT/'inputs/long-demo-last-5-minutes.mp4']
    sources = [{**probe(p), 'id': f'source-{i+1}', 'path': str(p), 'name': p.name,
                'sha256': digest(p), 'size': p.stat().st_size,
                'split_group': 'existing-local-footage', 'independent_recording_verified': False}
               for i, p in enumerate(source_paths)]
    prior_path = ROOT/'outputs/manual-review-backups/manual-review-20261115T212246864689Z.json'
    prior = json.loads(prior_path.read_text())
    if prior['source']['size'] != sources[0]['size'] or prior['source']['name'] != sources[0]['name']:
        raise ValueError('Prior review does not match the selected source')
    candidates = []

    def add(source_index, start, length, kind, evidence=None):
        src = sources[source_index]
        start = max(0, round(start*src['fps'])/src['fps'])
        end = min(src['duration'], round((start+length)*src['fps'])/src['fps'])
        candidates.append({'source_id': src['id'], 'source_start': start, 'source_end': end,
                           'selection': kind, 'selection_evidence': evidence})

    # Deduplicate nearby proposals, including the two calls for the same final forehand.
    seen = []
    for moment in sorted(prior['moments'], key=lambda m: m['time']):
        verdict = moment.get('review', {}).get('verdict')
        if verdict not in {'forehand', 'backhand'} or any(abs(moment['time']-t) < 1.5 for t in seen):
            continue
        # Prefer the later, full-stroke proposal when overlapping calls exist.
        if any(0 < later['time']-moment['time'] < 1.5 and later.get('review', {}).get('verdict') == verdict for later in prior['moments']):
            continue
        seen.append(moment['time'])
        add(0, moment['time']-2, 4, 'previously_reviewed_shot', {'prior_verdict': verdict, 'source_time': moment['time']})
    for moment in prior['moments']:
        if moment.get('review', {}).get('verdict') == 'no_shot':
            add(0, moment['time']-1.5, 3, 'previously_reviewed_negative_candidate', moment['review'])
    # Stratified random samples span the entire longer recording, without model selection.
    for i in range(8):
        start = rng.uniform(i*sources[1]['duration']/8, (i+1)*sources[1]['duration']/8-6)
        add(1, start, 6, 'random_section')
    # Low motion is only a sampling hint, never evidence that a clip contains no shot.
    import numpy as np
    print('Finding low-motion candidate sections…', flush=True)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(source_paths[1]),
                                  '-vf', 'fps=1,scale=160:90,format=gray', '-f', 'rawvideo', '-'])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 90, 160).astype(np.float32)
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    ranked = sorted(range(2, len(frames)-6), key=lambda s: float(motion[s:s+4].mean()))
    taken = [(c['source_start'], c['source_end']) for c in candidates if c['source_id'] == sources[1]['id']]
    picked = 0
    for start in ranked:
        if any(start < b+2 and start+4 > a-2 for a,b in taken):
            continue
        add(1, start, 4, 'low_motion_negative_candidate', {'motion_score': float(motion[start:start+4].mean())})
        taken.append((start, start+4)); picked += 1
        if picked == 4:
            break
    add(0, 0, sources[0]['duration'], 'continuous_context')
    rng.shuffle(candidates)
    (directory/'clips').mkdir(parents=True)
    for index, clip in enumerate(candidates, 1):
        clip['id'] = f'clip-{index:03d}'
        source = next(s for s in sources if s['id'] == clip['source_id'])
        target = directory/'clips'/f"{clip['id']}.mp4"
        print(f"Exporting {clip['id']} ({clip['source_end']-clip['source_start']:.1f}s)", flush=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(clip['source_start']), '-i', source['path'],
                        '-t', str(clip['source_end']-clip['source_start']), '-map', '0:v:0', '-map', '0:a:0?',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
                        '-vf', 'setsar=1', '-fps_mode', 'passthrough', '-c:a', 'aac', '-movflags', '+faststart', str(target)], check=True)
        clip.update(probe(target))
        clip.update(file=str(target.relative_to(directory)), sha256=digest(target))
    manifest = {'version': 1, 'type': 'tennis-regression-review-set', 'created_at': now(), 'seed': seed,
                'id': f'local-review-{seed}', 'status': 'awaiting_human_review', 'player': 'near',
                'sources': sources, 'clips': candidates, 'prior_review_sha256': digest(prior_path),
                'selection_counts': dict(Counter(c['selection'] for c in candidates)),
                'limitations': ['Existing local footage only; source independence is unverified.',
                               'Clips overlap in source time; keep the entire source group in the same split.',
                               'Review MP4s are re-encoded; use original sources and mapped times to test decoding.',
                               'Prior labels and low-motion scores are sampling hints, not ground truth.']}
    write_json(directory/'manifest.json', manifest)
    write_json(directory/'annotations.json', {'version': 1, 'dataset_id': manifest['id'], 'revision': 0,
                                             'updated_at': now(), 'reviews': {}})
    print(json.dumps({'directory': str(directory), 'clips': len(candidates), 'selection_counts': manifest['selection_counts']}, indent=2))


def review_clips(manifest):
    return [c for c in manifest['clips'] if c.get('selection') != 'continuous_context']


def validate_review(review, clip=None, complete=False):
    if not isinstance(review, dict) or review.get('label') not in LABELS | {None}:
        raise ValueError('Choose forehand, backhand, no shot, or unsure')
    if complete and review.get('label') is None:
        raise ValueError('Every review clip needs a human decision before freezing')
    return {'label': review.get('label'), 'decision': review.get('decision', 'manual'),
            'reviewed_at': review.get('reviewed_at')}


def propose_label(events, window_count):
    """Aggregate the current detector to one clip label, without timing comparison."""
    labels = {e['stroke'] for e in events}
    if not window_count:
        return {'label': 'unsure', 'reason': 'Not enough tracked poses to classify this clip.'}
    if not labels:
        return {'label': 'no_shot', 'reason': 'No confident shot detected in this clip.'}
    if labels == {'forehand'} or labels == {'backhand'}:
        return {'label': next(iter(labels)), 'reason': 'The detector found only this shot type.'}
    return {'label': 'unsure', 'reason': 'Mixed shot types or a serve detected.'}


def prepare_proposals(directory):
    from types import SimpleNamespace
    from tennis_rnn_backend import collect_scores, decode_scores, ARTIFACTS
    dataset = Dataset(directory)
    args = SimpleNamespace(limit=None, mirrored=False, handedness='right', player='near', min_player_height=.2)
    backend_hash = digest(ROOT/'tennis_rnn_backend.py')
    report = {'version': 1, 'dataset_id': dataset.manifest['id'], 'created_at': now(),
              'architecture': 'MoveNet Lightning + 30-frame Tennis RNN GRU', 'runtime': 'Python TensorFlow CPU',
              'backend_sha256': backend_hash, 'models': {name: pair[1] for name, pair in ARTIFACTS.items()},
              'settings': {**vars(args), 'threshold': .98, 'min_gap': 2.0}, 'clips': {}}
    cache_dir = directory/'proposal-cache'/backend_hash[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for clip in review_clips(dataset.manifest):
        source = (directory/clip['file']).resolve()
        if digest(source) != clip['sha256']:
            raise ValueError(f"Clip changed: {clip['id']}")
        print(f"Proposing {clip['id']}…", flush=True)
        metadata = probe(source)
        times, scores = collect_scores(source, metadata, args, cache_dir/f"{clip['id']}.npz")
        events = decode_scores(times, scores, metadata['duration'])
        report['clips'][clip['id']] = {**propose_label(events, len(scores)), 'clip_sha256': clip['sha256'],
                                      'classified_windows': len(scores), 'detected_labels': [e['stroke'] for e in events]}
        write_json(directory/'proposals.json', report)
    print(json.dumps(dict(Counter(p['label'] for p in report['clips'].values())), indent=2))


class Dataset:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.manifest = json.loads((directory/'manifest.json').read_text())
        self.manifest_hash = digest(directory/'manifest.json')
        self.clips = {c['id']: c for c in self.manifest['clips']}
        self.lock = threading.Lock()
        # Preserve all earlier human work before simplifying its schema.
        path = self.directory/'annotations.json'
        legacy = json.loads(path.read_text())
        if legacy.get('version', 1) < 2:
            backup = self.directory/'annotations-before-simple-review.json'
            if not backup.exists():
                with backup.open('x') as handle:
                    handle.write(path.read_text())
            converted = {}
            for key, item in legacy['reviews'].items():
                verdict = item.get('verdict')
                labels = {s['label'] for s in item.get('shots', [])}
                label = None
                if verdict == 'no_shot': label = 'no_shot'
                elif verdict == 'exclude': label = 'unsure'
                elif verdict == 'shots': label = next(iter(labels)) if len(labels) == 1 and labels <= {'forehand', 'backhand'} else 'unsure'
                converted[key] = {'label': label, 'decision': 'migrated_human_review', 'reviewed_at': item.get('reviewed_at')}
            write_json(path, {**legacy, 'version': 2, 'revision': legacy['revision']+1, 'reviews': converted})

    def annotations(self):
        data = json.loads((self.directory/'annotations.json').read_text())
        if data['dataset_id'] != self.manifest['id']:
            raise ValueError('Annotations belong to a different dataset')
        return data

    def proposals(self):
        path = self.directory/'proposals.json'
        if not path.exists(): return {}
        data = json.loads(path.read_text())
        if data['dataset_id'] != self.manifest['id']: return {}
        if data.get('backend_sha256') != digest(ROOT/'tennis_rnn_backend.py'): return {}
        return {key: item for key, item in data['clips'].items()
                if key in self.clips and item['clip_sha256'] == self.clips[key]['sha256'] and item['label'] in LABELS}

    def save(self, clip_id, review, revision):
        with self.lock:
            data = self.annotations()
            if revision != data['revision']:
                raise ValueError('Review changed in another tab. Reload before saving.')
            validated = validate_review(review, self.clips[clip_id])
            if clip_id not in {c['id'] for c in review_clips(self.manifest)}:
                raise ValueError('This context segment is outside the single-label review queue')
            proposal = self.proposals().get(clip_id)
            if review.get('action') == 'approve':
                if not proposal or validated['label'] != proposal['label']:
                    raise ValueError('Suggestion changed. Reload before approving.')
                validated['decision'] = 'approved'
            else:
                validated['decision'] = 'corrected' if proposal else 'manual'
            validated['suggested_label'] = proposal['label'] if proposal else None
            validated['reviewed_at'] = now()
            data['reviews'][clip_id] = validated
            data['revision'] += 1; data['updated_at'] = now()
            write_json(self.directory/'annotations.json', data)
            return data

    def freeze(self, revision):
        with self.lock:
            data = self.annotations()
            if revision != data['revision']:
                raise ValueError('Review changed in another tab. Reload before freezing.')
            if digest(self.directory/'manifest.json') != self.manifest_hash:
                raise ValueError('Dataset manifest changed; restart the reviewer')
            reviews = {}
            for clip in review_clips(self.manifest):
                review = data['reviews'].get(clip['id'], {'label': None})
                reviews[clip['id']] = validate_review(review, clip, complete=True)
                if digest(self.directory/clip['file']) != clip['sha256']:
                    raise ValueError(f"Clip changed: {clip['id']}")
            if all(r['label'] == 'unsure' for r in reviews.values()):
                raise ValueError('At least one clip must be included')
            for source in self.manifest['sources']:
                if digest(source['path']) != source['sha256']:
                    raise ValueError(f"Original source changed: {source['name']}")
            truth = []
            for clip in review_clips(self.manifest):
                review = reviews[clip['id']]
                truth.append({'clip_id': clip['id'], 'source_id': clip['source_id'], 'source_start': clip['source_start'],
                              'source_end': clip['source_end'], 'included': True, **review})
            snapshot = {'version': 2, 'type': 'human-reviewed-tennis-regression-dataset', 'frozen_at': now(),
                        'manifest_sha256': self.manifest_hash, 'manifest': self.manifest,
                        'annotation_revision': revision, 'ground_truth': truth,
                        'evaluation': {**json.loads((ROOT/'docs/regression-evaluation-policy.json').read_text()),
                                       'labels': sorted(LABELS)}}
            body = json.dumps(snapshot, indent=2)+'\n'
            sha = hashlib.sha256(body.encode()).hexdigest()
            name = f'frozen-r{revision}-{sha[:12]}.json'
            with (self.directory/name).open('x') as handle:
                handle.write(body)
            return {'file': name, 'sha256': sha, 'included': sum(t['included'] for t in truth)}


def handler_for(dataset):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; media-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            super().end_headers()

        def json_response(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = f'http://127.0.0.1:{self.server.server_port}'
            if (origin and origin != allowed) or self.headers.get('Content-Type') != 'application/json':
                self.json_response({'error': 'Only same-origin JSON requests are accepted'}, 403); return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 100000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == '/api/review':
                    result = dataset.save(payload['clip_id'], payload['review'], payload['revision'])
                elif path == '/api/freeze':
                    result = dataset.freeze(payload['revision'])
                else:
                    self.send_error(404); return
                self.json_response(result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.json_response({'error': str(exc)}, 409)

        def do_HEAD(self):
            self.serve(True)

        def do_GET(self):
            self.serve(False)

        def serve(self, head):
            path = urlsplit(self.path).path
            if path == '/api/dataset':
                clips = [{k: c[k] for k in ('id', 'duration', 'fps', 'width', 'height')} for c in review_clips(dataset.manifest)]
                self.json_response({'id': dataset.manifest['id'], 'clips': clips, 'annotations': dataset.annotations(),
                                    'proposals': dataset.proposals()}); return
            if path == '/api/export':
                self.json_response({'manifest': dataset.manifest, 'annotations': dataset.annotations(), 'proposals': dataset.proposals()}); return
            media = re.fullmatch(r'/clips/(clip-\d{3})\.mp4', path)
            if media and media[1] in dataset.clips:
                file = dataset.directory/dataset.clips[media[1]]['file']; mime = 'video/mp4'
            elif path in ('/', '/review.js', '/review.css'):
                file = ASSETS/('index.html' if path == '/' else path[1:])
                mime = {'/': 'text/html', '/review.js': 'text/javascript', '/review.css': 'text/css'}[path]
            else:
                self.send_error(404); return
            size = file.stat().st_size
            try:
                start, end, partial = byte_range(self.headers.get('Range'), size)
            except ValueError:
                self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mime); self.send_header('Content-Length', str(end-start+1))
            self.send_header('Accept-Ranges', 'bytes')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not head:
                try:
                    with file.open('rb') as handle:
                        handle.seek(start); remaining = end-start+1
                        while remaining:
                            block = handle.read(min(262144, remaining))
                            if not block: break
                            self.wfile.write(block); remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'serve', 'propose'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    if args.command == 'build':
        build_dataset(args.directory.resolve(), args.seed)
    elif args.command == 'propose':
        prepare_proposals(args.directory.resolve())
    else:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Dataset(args.directory)))
        print(f'Review dataset: http://127.0.0.1:{server.server_port}/\nReviews save to {args.directory}/annotations.json', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
