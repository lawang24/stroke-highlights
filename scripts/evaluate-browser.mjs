// node scripts/evaluate-browser.mjs FROZEN.json OUTPUT.json [PORT]
// Open the printed loopback URL and click Run baseline. Never edits review labels.
import {readFile, writeFile, mkdir, mkdtemp, stat, rm, rename} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createServer} from 'node:http';
import {tmpdir, platform, arch, cpus} from 'node:os';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {createHash, randomUUID} from 'node:crypto';
import {evaluate, scoredTruth} from './regression-metrics.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
if (!process.argv[2] || !process.argv[3]) throw new Error('Usage: node scripts/evaluate-browser.mjs FROZEN.json OUTPUT.json [PORT]');
const snapshot = resolve(process.argv[2]), output = resolve(process.argv[3]), port = Number(process.argv[4] || 8772);
try { await stat(output); throw new Error('Output already exists; use a new filename'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const sha = data => createHash('sha256').update(data).digest('hex');
const frozenBytes = await readFile(snapshot), frozen = JSON.parse(frozenBytes);
const dataset = dirname(snapshot);
if (sha(await readFile(resolve(dataset, 'manifest.json'))) !== frozen.manifest_sha256) throw new Error('Manifest checksum mismatch');
const truth = scoredTruth(frozen);
const clips = truth.map(row => frozen.manifest.clips.find(clip => clip.id === row.clip_id));
if (!clips.length || clips.some(clip => !clip)) throw new Error('Missing scored clips');
for (const clip of clips) if (sha(await readFile(resolve(dataset, clip.file))) !== clip.sha256) throw new Error(`Clip checksum mismatch: ${clip.id}`);
const files = ['web/media.js', 'web/tennis.worker.js', 'web/tennis-rnn.js', 'package-lock.json',
  'scripts/evaluate-browser.mjs', 'scripts/regression-metrics.mjs',
  'public/runtime/tennis-rnn/weights.json', 'public/runtime/tennis-rnn/movenet.tflite'];
const hashes = Object.fromEntries(await Promise.all(files.map(async file => [file, sha(await readFile(resolve(root, file)))])));
const temp = await mkdtemp(resolve(tmpdir(), 'tennis-regression-'));
const bundle = (contents, name) => build({stdin: {contents, resolveDir: resolve(root, 'web'), loader: 'js'}, outfile: resolve(temp, name), bundle: true, minify: true, format: 'iife'});
await bundle(await readFile(resolve(root, 'web/tennis.worker.js'), 'utf8'), 'worker.js');
const replace = (source, from, to) => { if (!source.includes(from)) throw new Error(`Source anchor missing: ${from}`); return source.replace(from, to); };
let media = await readFile(resolve(root, 'web/media.js'), 'utf8');
if (!media.includes('export async function exportReel')) throw new Error('Media source anchor missing');
media = media.slice(0, media.indexOf('export async function exportReel'));
media = replace(media, 'import.meta.env.BASE_URL', "'/'");
media = replace(media, '`${base}runtime/tennis-worker.js`', '`${base}worker.js`');
media = replace(media, 'return decodeScores(samples, video.duration, options);', 'return {events: decodeScores(samples, video.duration, options), classifiedWindows: samples.length, samples, duration: video.duration, frames: count};');
const options = {handedness: 'right', mirrored: false, threshold: .98, minGap: 2, before: 1.2, after: .9};
const token = randomUUID();
await bundle(media + `
const status = document.querySelector('#status'), results = document.querySelector('#results');
document.querySelector('button').onclick = async event => {
  event.target.disabled = true;
  const report = {date: new Date().toISOString(), userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, crossOriginIsolated, runs: []};
  try {
    for (const id of ${JSON.stringify(clips.map(clip => clip.id))}) {
      const response = await fetch('/clips/' + id);
      if (!response.ok) throw new Error('Clip fetch failed: ' + id);
      const url = URL.createObjectURL(await response.blob());
      const start = performance.now();
      try {
        const run = await analyzeVideo(url, ${JSON.stringify(options)}, new AbortController().signal,
          (message, percent) => {status.textContent = (report.runs.length+1) + '/${clips.length} ' + id + ': ' + message + ' (' + percent.toFixed(0) + '%)';});
        report.runs.push({...run, clipId: id, elapsedMs: performance.now()-start});
      } finally { URL.revokeObjectURL(url); }
      const saved = await fetch('/results', {method: 'POST', headers: {'Content-Type':'application/json', 'X-Run-Token':${JSON.stringify(token)}}, body: JSON.stringify(report)});
      if (!saved.ok) throw new Error(await saved.text());
      const summary = await saved.json();
      results.textContent = summary.metrics ? JSON.stringify(summary.metrics, null, 2) : report.runs.length + ' clips saved';
    }
    status.textContent = 'Baseline complete. ${clips.length} clips evaluated and saved.';
  } catch(error) { status.textContent = 'Baseline failed: ' + error.stack; }
};`, 'evaluate.js');
await writeFile(resolve(temp, 'index.html'), '<!doctype html><meta charset="utf-8"><title>Tennis regression baseline</title><h1>Tennis regression baseline</h1><p>Fresh browser inference on frozen human labels. No timing comparisons.</p><button>Run baseline</button><p id="status">Ready</p><pre id="results"></pre><script src="/evaluate.js"></script>');
const identity = {snapshot, snapshotSHA256: sha(frozenBytes), annotationRevision: frozen.annotation_revision, hashes,
  options, platform: platform(), arch: arch(), cpu: cpus()[0]?.model,
  clips: clips.map(({id, sha256}) => ({id, sha256})),
  methodology: 'Fresh worker per clip; actual browser decode, MoveNet CPU/WASM and JS GRU; 30Hz; no cached predictions. Current preprocessing reorder included. Runtime is descriptive, not a controlled speed benchmark.'};
const mime = {html:'text/html', js:'text/javascript', wasm:'application/wasm', json:'application/json', mp4:'video/mp4', tflite:'application/octet-stream'};
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, `http://127.0.0.1:${port}`).pathname;
    if (req.method === 'POST' && path === '/results') {
      if (req.headers['x-run-token'] !== token) {res.writeHead(403);res.end();return;}
      const chunks = []; let size = 0;
      for await (const chunk of req) {size += chunk.length; if (size > 10_000_000) throw new Error('Result too large'); chunks.push(chunk);}
      const report = JSON.parse(Buffer.concat(chunks));
      const complete = report.runs.length === clips.length;
      const metrics = complete ? evaluate(frozen, report.runs) : null;
      const saved = {...report, identity, complete, metrics};
      await mkdir(dirname(output), {recursive: true});
      await writeFile(output + '.tmp', JSON.stringify(saved, null, 2));
      await rename(output + '.tmp', output);
      console.log(JSON.stringify({completed: report.runs.length, total: clips.length, metrics: metrics && {correct: metrics.correct, accuracy: metrics.accuracy}, output}));
      res.writeHead(200, {'Content-Type':'application/json'}); res.end(JSON.stringify({metrics})); return;
    }
    let file;
    const clip = clips.find(clip => path === '/clips/' + clip.id);
    if (clip) file = resolve(dataset, clip.file);
    else if (['/', '/evaluate.js', '/worker.js'].includes(path)) file = resolve(temp, path === '/' ? 'index.html' : path.slice(1));
    else if (/^\/runtime\/(tflite|tennis-rnn)\/[a-zA-Z0-9_.-]+$/.test(path)) file = resolve(root, 'public', path.slice(1));
    else {res.writeHead(404);res.end();return;}
    res.writeHead(200, {'Content-Type':mime[file.split('.').at(-1)] || 'application/octet-stream', 'Content-Length':(await stat(file)).size, 'Cache-Control':'no-store'});
    createReadStream(file).pipe(res);
  } catch(error) {res.writeHead(500);res.end(String(error));console.error(error);}
});
server.listen(port, '127.0.0.1', () => console.log(`Baseline: http://127.0.0.1:${port}/`));
process.on('SIGINT', () => server.close(async () => {await rm(temp, {recursive:true, force:true});process.exit(0);}));
