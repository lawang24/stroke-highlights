// Local A/B benchmark. Run after npm run setup; open the printed URL.
// Serves only the selected video, generated benchmark assets, and runtime assets.
import {readFile, writeFile, mkdir, mkdtemp, stat, rm} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createServer} from 'node:http';
import {tmpdir} from 'node:os';
import {dirname, resolve, basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const video = resolve(process.argv[2] || resolve(root, 'outputs/adaspot-comparison-input.mp4'));
const output = resolve(process.argv[3] || resolve(root, 'experiments/2026-11-15-preprocessing-benchmark.json'));
const port = Number(process.argv[4] || 8771);
const temp = await mkdtemp(resolve(tmpdir(), 'tennis-browser-benchmark-'));
const workerSource = await readFile(resolve(root, 'web/tennis.worker.js'), 'utf8');
const mediaSource = await readFile(resolve(root, 'web/media.js'), 'utf8');
const replace = (source, from, to) => {
  if (!source.includes(from)) throw new Error(`Benchmark source anchor missing: ${from}`);
  return source.replace(from, to);
};
const bundle = (contents, outfile) => build({stdin: {contents, resolveDir: resolve(root, 'web'), loader: 'js'}, outfile: resolve(temp, outfile), bundle: true, minify: true, format: 'iife'});
for (const variant of ['before', 'after']) {
  let source = workerSource;
  if (variant === 'before') {
    source = replace(source, 'const rgb = tf.browser.fromPixels(pixels);', 'const rgb = tf.gather(tf.browser.fromPixels(pixels), [2,1,0], 2);');
    source = replace(source, 'const bgr = tf.gather(resized, [2,1,0], 2);', 'const bgr = resized;');
  }
  source = replace(source, 'const points = tf.tidy(() => {', 'const prepStart = performance.now(); let prepMs, inferenceMs;\nconst points = tf.tidy(() => {');
  source = replace(source, 'const output = model.predict(input);', 'prepMs = performance.now() - prepStart; const inferenceStart = performance.now();\nconst output = model.predict(input);');
  source = replace(source, 'return Float32Array.from(tensor.dataSync());', 'const result = Float32Array.from(tensor.dataSync()); inferenceMs = performance.now() - inferenceStart; return result;');
  source = replace(source, "self.postMessage({type: 'prediction', prediction});", "self.postMessage({type: 'prediction', prediction, timing: {prepMs, inferenceMs, cropWidth: w, cropHeight: h}});");
  await bundle(source, `${variant}.js`);
}
let media = mediaSource.slice(0, mediaSource.indexOf('export async function exportReel'));
media = replace(media, 'import.meta.env.BASE_URL', "'/'");
media = replace(media, '`${base}runtime/tennis-worker.js`', '`${base}${options.variant}.js`');
media = replace(media, "progress('Loading Tennis RNN…', 0);", "const started = performance.now(); const timings = []; progress('Loading Tennis RNN…', 0);");
media = replace(media, 'const count = Math.ceil(video.duration * 30);', 'const count = Math.ceil(Math.min(video.duration, options.limit || Infinity) * 30); const setupMs = performance.now() - started;');
media = replace(media, 'const time = Math.min(i / 30, video.duration - .001);', 'const frameStart = performance.now(); const time = Math.min(i / 30, video.duration - .001);');
media = replace(media, 'ctx.drawImage(video, 0, 0, canvas.width, canvas.height);', 'const seekEnd = performance.now(); ctx.drawImage(video, 0, 0, canvas.width, canvas.height);');
media = replace(media, "const { prediction } = await ask({ type: 'frame', bitmap, time }, [bitmap]);", "const bitmapEnd = performance.now(); const { prediction, timing } = await ask({ type: 'frame', bitmap, time }, [bitmap]); timings.push({...timing, seekMs: seekEnd-frameStart, bitmapMs: bitmapEnd-seekEnd, workerMs: performance.now()-bitmapEnd});");
media = replace(media, 'return decodeScores(samples, video.duration, options);', 'const events = decodeScores(samples, video.duration, options); return {elapsedMs: performance.now()-started, setupMs, frames: count, duration: video.duration, width: canvas.width, height: canvas.height, timings, samples, events};');
await bundle(media + `
const status = document.querySelector('#status'), results = document.querySelector('#results');
document.querySelector('button').onclick = async event => {
  event.target.disabled = true;
  try {
    const blob = await (await fetch('/video')).blob(), url = URL.createObjectURL(blob);
    const report = {date: new Date().toISOString(), userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, crossOriginIsolated, visibility: document.visibilityState, methodology: 'Two 2-second warmups, then full-clip ABBA; fresh worker each run; cached local video Blob; detection only, no export. Same minified source except preprocessing order and identical timing probes.', warmups: [], runs: []};
    const order = ['before', 'after', 'before', 'after', 'after', 'before'];
    for (let index=0; index<order.length; index++) {
      const variant = order[index], warmup = index < 2;
      const name = warmup ? 'Warmup ' + variant : 'Run ' + (index-1) + '/4 ' + variant;
      const run = await analyzeVideo(url, {variant, limit: warmup ? 2 : null, handedness:'right', mirrored:false, threshold:.98, minGap:2, before:1.2, after:.9}, new AbortController().signal,
        (message, percent) => { status.textContent = name + ': ' + message + ' (' + percent.toFixed(0) + '%)'; });
      run.variant = variant;
      run.visibility = document.visibilityState;
      (warmup ? report.warmups : report.runs).push(run);
      results.textContent += name + ': ' + (run.elapsedMs/1000).toFixed(2) + ' s; ' + run.events.length + ' events\\n';
      await fetch('/results', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(report)});
    }
    URL.revokeObjectURL(url);
    status.textContent = 'Benchmark complete. Four full runs saved.';
  } catch(error) { status.textContent = 'Benchmark failed: ' + error.stack; }
};
`, 'benchmark.js');
await writeFile(resolve(temp, 'index.html'), '<!doctype html><meta charset="utf-8"><title>Tennis preprocessing benchmark</title><h1>Tennis preprocessing benchmark</h1><p>Two warmups, then four full-video runs: before, after, after, before.</p><button>Start benchmark</button><p id="status">Ready</p><pre id="results"></pre><script src="/benchmark.js"></script>');
const mime = {html:'text/html', js:'text/javascript', wasm:'application/wasm', json:'application/json', mp4:'video/mp4'};
const sourceIdentity = {name: basename(video), size: (await stat(video)).size, workerSHA256: createHash('sha256').update(workerSource).digest('hex')};
const server = createServer(async (req,res) => {
  try {
    const path = new URL(req.url, `http://127.0.0.1:${port}`).pathname;
    if (req.method === 'POST' && path === '/results') {
      const chunks = []; let size = 0;
      for await (const chunk of req) { size += chunk.length; if (size > 10_000_000) throw new Error('Result too large'); chunks.push(chunk); }
      const report = {...JSON.parse(Buffer.concat(chunks).toString()), source: sourceIdentity};
      await mkdir(dirname(output), {recursive:true});
      await writeFile(output, JSON.stringify(report, null, 2));
      console.log(JSON.stringify({completedRuns: report.runs.length, latest: report.runs.at(-1)?.elapsedMs, file: output}));
      res.writeHead(200); res.end('Saved'); return;
    }
    let file;
    if (path === '/video') file = video;
    else if (['/', '/benchmark.js', '/before.js', '/after.js'].includes(path)) file = resolve(temp, path === '/' ? 'index.html' : path.slice(1));
    else if (/^\/runtime\/(tflite|tennis-rnn)\/[a-zA-Z0-9_.-]+$/.test(path)) file = resolve(root, 'public', path.slice(1));
    else { res.writeHead(404); res.end(); return; }
    const {size} = await stat(file);
    const headers = {'Content-Type': mime[file.split('.').at(-1)] || 'application/octet-stream', 'Accept-Ranges':'bytes', 'Cache-Control':'no-cache'};
    const range = req.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
    if (range) {
      const start = Number(range[1]), end = Math.min(range[2] ? Number(range[2]) : size-1, size-1);
      if (start > end) {res.writeHead(416); res.end(); return;}
      res.writeHead(206, {...headers, 'Content-Length':end-start+1, 'Content-Range': 'bytes '+start+'-'+end+'/'+size});
      createReadStream(file,{start,end}).pipe(res);
    } else {res.writeHead(200,{...headers,'Content-Length':size}); createReadStream(file).pipe(res);}
  } catch(error) {res.writeHead(500);res.end(String(error));}
});
server.listen(port, '127.0.0.1', () => console.log(`Benchmark: http://127.0.0.1:${port}/`));
process.on('SIGINT', () => server.close(async () => {await rm(temp,{recursive:true,force:true}); process.exit(0);}));
