// All inference/export assets are served locally; there are no runtime CDN calls.
import {mkdir, copyFile, readdir, rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {build} from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const runtime = resolve(root, 'public/runtime');
const prepared = spawnSync('uv', ['run', 'python', 'scripts/export_tennis_rnn.py'], {cwd: root, stdio: 'inherit'});
if (prepared.error || prepared.status !== 0) throw new Error('Model setup failed. Install uv, then run uv sync and npm run setup.');
await mkdir(runtime, {recursive: true});
await build({entryPoints: [resolve(root, 'web/tennis.worker.js')], outfile: resolve(runtime, 'tennis-worker.js'), bundle: true, format: 'iife', minify: true});
for (const [source, target] of [
  ['@tensorflow/tfjs-tflite/wasm', 'tflite'],
  ['@ffmpeg/core/dist/esm', 'ffmpeg'],
]) {
  const destination = resolve(runtime, target);
  await mkdir(destination, {recursive: true});
  for (const name of await readdir(resolve(root, 'node_modules', source))) {
    if (/\.(wasm|js)$/.test(name)) await copyFile(resolve(root, 'node_modules', source, name), resolve(destination, name));
  }
}
await mkdir(resolve(runtime, 'tennis-rnn'), {recursive: true});
for (const name of ['weights.json', 'movenet.tflite']) await copyFile(resolve(root, 'models/tennis-rnn', name), resolve(runtime, 'tennis-rnn', name));
// Remove generated assets belonging to the retired browser classifier.
for (const name of ['vision', 'pose-worker.js', 'pose_landmarker_lite.task']) await rm(resolve(runtime, name), {recursive: true, force: true});
console.log('Tennis RNN browser inference and export assets are ready.');
