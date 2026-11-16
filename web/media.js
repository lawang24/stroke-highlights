import { decodeScores } from './tennis-rnn.js';

const base = new URL(import.meta.env.BASE_URL, location.href).href;
const abortError = () => new DOMException('Cancelled', 'AbortError');

export function waitForMedia(video, event, signal, trigger) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, done);
      video.removeEventListener('error', failed);
      signal?.removeEventListener('abort', aborted);
    };
    const done = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error('This browser cannot decode the video. Try an H.264 MP4 or a WebM file.')); };
    const aborted = () => { cleanup(); reject(abortError()); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('Video decoding timed out. Try a shorter MP4 clip.')); }, 30000);
    video.addEventListener(event, done, { once: true });
    video.addEventListener('error', failed, { once: true });
    signal?.addEventListener('abort', aborted, { once: true });
    if (signal?.aborted) return aborted();
    try { trigger?.(); } catch (error) { cleanup(); reject(error); }
  });
}

export async function analyzeVideo(url, options, signal, progress) {
  const worker = new Worker(`${base}runtime/tennis-worker.js`);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const ask = (message, transfer = []) => new Promise((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); worker.onmessage = null; worker.onerror = null; signal.removeEventListener('abort', aborted); };
    const aborted = () => { cleanup(); reject(abortError()); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('Tennis RNN analysis timed out. Reload the page and try a shorter clip.')); }, 120000);
    worker.onmessage = ({ data }) => { cleanup(); data.type === 'error' ? reject(new Error(data.message)) : resolve(data); };
    worker.onerror = (event) => { cleanup(); reject(new Error(event.message || 'Could not load Tennis RNN. Run npm run setup and reload.')); };
    signal.addEventListener('abort', aborted, { once: true });
    if (signal.aborted) return aborted();
    worker.postMessage(message, transfer);
  });
  const stop = () => worker.terminate();
  signal.addEventListener('abort', stop, { once: true });
  try {
    progress('Loading Tennis RNN…', 0);
    await ask({ type: 'init', base, options });
    await waitForMedia(video, 'loadeddata', signal, () => { video.src = url; });
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    const samples = [];
    const count = Math.ceil(video.duration * 30);
    for (let i = 0; i < count; i++) {
      if (signal.aborted) throw abortError();
      const time = Math.min(i / 30, video.duration - .001);
      if (Math.abs(video.currentTime - time) > .001) await waitForMedia(video, 'seeked', signal, () => { video.currentTime = time; });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const bitmap = await createImageBitmap(canvas);
      const { prediction } = await ask({ type: 'frame', bitmap, time }, [bitmap]);
      if (prediction) samples.push(prediction);
      if (i % 5 === 0 || i === count - 1) progress(`Tennis RNN · ${Math.round(time)} of ${Math.round(video.duration)} seconds analyzed`, (i + 1) / count * 100);
    }
    return decodeScores(samples, video.duration, options);
  } finally {
    signal.removeEventListener('abort', stop);
    worker.terminate();
    video.removeAttribute('src');
    video.load();
  }
}

export async function exportReel(file, ranges, signal, progress) {
  if (!ranges.length) throw new Error('Include at least one shot before compiling.');
  const { FFmpeg } = await import('@ffmpeg/ffmpeg');
  if (signal.aborted) throw abortError();
  const ffmpeg = new FFmpeg();
  const stop = () => ffmpeg.terminate();
  signal.addEventListener('abort', stop, { once: true });
  let lastLog = '';
  ffmpeg.on('log', ({ message }) => { lastLog = message; });
  const exec = async (args) => {
    if (signal.aborted) throw abortError();
    if (await ffmpeg.exec(args) !== 0) throw new Error(`Video export failed. Try a shorter clip. ${lastLog}`);
  };
  try {
    progress('Loading local video compiler…', 0);
    await ffmpeg.load({ coreURL: `${base}runtime/ffmpeg/ffmpeg-core.js`, wasmURL: `${base}runtime/ffmpeg/ffmpeg-core.wasm` });
    if (signal.aborted) throw abortError();
    progress('Opening your video in local memory…', 2);
    // WORKERFS reads the File without duplicating the entire source in WASM memory.
    await ffmpeg.createDir('/input');
    await ffmpeg.mount('WORKERFS', { files: [new File([file], 'source')] }, '/input');
    const parts = [];
    for (const [index, [start, end]] of ranges.entries()) {
      progress(`Compiling excerpt ${index + 1} of ${ranges.length}…`, 5 + index / ranges.length * 85);
      const part = `part-${index}.mp4`;
      await exec(['-ss', String(start), '-i', '/input/source', '-t', String(end - start),
        '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,fps=30',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', '-pix_fmt', 'yuv420p', '-threads', '1',
        '-c:a', 'aac', '-af', 'asetpts=PTS-STARTPTS', '-avoid_negative_ts', 'make_zero', part]);
      parts.push(`file '${part}'`);
    }
    progress('Joining your highlights…', 92);
    await ffmpeg.writeFile('clips.txt', new TextEncoder().encode(parts.join('\n')));
    await exec(['-f', 'concat', '-safe', '1', '-i', 'clips.txt', '-vf', 'fps=30:start_time=0',
      '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '1',
      '-c:a', 'copy', '-movflags', '+faststart', 'reel.mp4']);
    const data = await ffmpeg.readFile('reel.mp4');
    return new Blob([data], { type: 'video/mp4' });
  } finally {
    signal.removeEventListener('abort', stop);
    ffmpeg.terminate();
  }
}
