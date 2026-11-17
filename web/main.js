import { compilationRanges, validateEvents } from './detection.js';
import { analyzeVideo, exportReel, waitForMedia } from './media.js';

const $ = (id) => document.getElementById(id);
const labels = ['forehand', 'backhand'];
const state = { file: null, url: null, duration: 0, events: [], reels: {}, controller: null, analyzed: false, classifier: 'manual' };
let devClipPicker;
const formatTime = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const showError = (message = '') => { $('error').textContent = message; $('error').hidden = !message; };
const report = (text, percent) => {
  $('status-panel').hidden = false;
  $('status-text').textContent = text;
  $('progress').value = percent;
};
const options = () => ({ player: 'near', handedness: $('handedness').value, mirrored: $('mirrored').checked, before: Number($('before').value), after: Number($('after').value) });

function invalidateReels() {
  for (const label of labels) {
    const video = $(`${label}-video`);
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.hidden = true;
    const link = $(`download-${label}`);
    link.removeAttribute('href');
    link.hidden = true;
    if (state.reels[label]) URL.revokeObjectURL(state.reels[label]);
  }
  state.reels = {};
}

function updateControls() {
  const busy = !!state.controller;
  devClipPicker?.setBusy(busy);
  $('results').hidden = !state.file;
  $('progress').hidden = !busy;
  let valid = true;
  try { validateEvents(state.events, state.duration); } catch { valid = false; }
  document.body.classList.toggle('busy', busy);
  for (const control of $('settings').elements) control.disabled = busy || !state.file;
  $('analyze').textContent = state.events.length ? 'Detect again (replaces edits)' : 'Detect shots';
  $('file-input').disabled = busy;
  $('replace-video').disabled = busy;
  $('cancel').hidden = !busy;
  for (const id of ['add-shot', 'load-review', 'save-review']) $(id).disabled = busy || !state.file || (id === 'save-review' && !valid);
  for (const control of $('shot-rows').querySelectorAll('input,select,button')) control.disabled = busy;
  $('compile-all').disabled = busy || !valid || !state.events.some(e => e.include && labels.includes(e.stroke));
  for (const label of labels) {
    const count = state.events.filter(e => e.include && e.stroke === label).length;
    const ranges = valid ? compilationRanges(state.events, label) : [];
    const duration = ranges.reduce((total, [start, end]) => total + end - start, 0);
    $(`${label}-count`).textContent = `${count} shot${count === 1 ? '' : 's'}`;
    $(`compile-${label}`).disabled = busy || !valid || !count;
    $(`compile-${label}`).textContent = state.reels[label] ? 'Recompile' : 'Compile';
    $(`${label}-duration`).textContent = count && valid ? (duration < 60 ? `${duration.toFixed(1)}s` : formatTime(duration)) : '';
    $(`${label}-empty`).hidden = !!state.reels[label];
    $(`${label}-empty`).querySelector('p').textContent = !valid ? 'Correct the shot times before compiling.' : count ? 'Ready to compile.' : state.file ? 'No included shots.' : 'No video selected.';
  }
}

async function operation(task) {
  if (state.controller) return;
  const controller = new AbortController();
  state.controller = controller;
  showError();
  updateControls();
  try { await task(controller.signal); }
  catch (error) {
    if (controller.signal.aborted || error?.name === 'AbortError') report('Cancelled. Video and shot list retained.', 0);
    else { report('Processing stopped. Retry or edit shots manually.', 0); showError(error?.message || String(error)); }
  } finally {
    state.controller = null;
    updateControls();
  }
}

async function compile(categoryList, signal) {
  validateEvents(state.events, state.duration);
  let created = 0;
  for (const label of categoryList) {
    const ranges = compilationRanges(state.events, label);
    if (!ranges.length) continue;
    const blob = await exportReel(state.file, ranges, signal, (text, percent) => report(`${label === 'forehand' ? 'Forehands' : 'Backhands'} · ${text}`, percent));
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    const old = state.reels[label];
    state.reels[label] = URL.createObjectURL(blob);
    const video = $(`${label}-video`);
    video.src = state.reels[label];
    video.hidden = false;
    const link = $(`download-${label}`);
    link.href = state.reels[label];
    link.hidden = false;
    if (old) URL.revokeObjectURL(old);
    created++;
    updateControls();
  }
  report(created ? 'Compilations ready.' : 'No classified shots. Add or relabel shots, then compile.', 100);
}

function renderReview() {
  $('review-section').hidden = !state.file;
  $('shot-rows').replaceChildren();
  $('no-shots').hidden = state.events.length > 0;
  state.events.forEach((event, index) => {
    const row = document.createElement('tr');
    const cell = (node) => { const td = document.createElement('td'); if (typeof node === 'string') td.textContent = node; else td.append(node); row.append(td); };
    const keep = document.createElement('input');
    keep.type = 'checkbox'; keep.checked = event.include; keep.setAttribute('aria-label', `Include shot ${index + 1}`);
    const changed = () => {
      invalidateReels();
      try { validateEvents(state.events, state.duration); showError(); } catch (error) { showError(error.message); }
      updateControls();
    };
    keep.addEventListener('change', () => { event.include = keep.checked; changed(); });
    cell(keep);
    const jump = document.createElement('button');
    jump.textContent = `Play ${index + 1}`;
    jump.setAttribute('aria-label', `Play shot ${index + 1}`);
    jump.addEventListener('click', () => { $('source-video').currentTime = Number.isFinite(event.start) ? event.start : 0; $('source-video').play().catch(() => {}); $('source-panel').scrollIntoView({ behavior: 'smooth', block: 'center' }); });
    cell(jump);
    for (const key of ['start', 'end']) {
      const input = document.createElement('input');
      input.type = 'number'; input.min = 0; input.max = state.duration; input.step = '.01'; input.value = event[key].toFixed(2);
      input.setAttribute('aria-label', `Shot ${index + 1} ${key} in seconds`);
      input.addEventListener('input', () => { event[key] = input.valueAsNumber; changed(); });
      cell(input);
    }
    const select = document.createElement('select');
    select.setAttribute('aria-label', `Shot ${index + 1} category`);
    for (const label of [...labels, 'unknown', 'ignore']) { const option = document.createElement('option'); option.value = label; option.textContent = label[0].toUpperCase() + label.slice(1); select.append(option); }
    select.value = event.stroke;
    select.addEventListener('change', () => { event.stroke = select.value; if (['unknown', 'ignore'].includes(event.stroke)) { event.include = false; keep.checked = false; } changed(); });
    cell(select);
    const reason = document.createElement('span'); reason.className = 'reason'; reason.textContent = event.reason || 'Manually reviewed'; cell(reason);
    const remove = document.createElement('button'); remove.textContent = 'Remove'; remove.setAttribute('aria-label', `Remove shot ${index + 1}`);
    remove.addEventListener('click', () => { state.events.splice(index, 1); changed(); renderReview(); }); cell(remove);
    $('shot-rows').append(row);
  });
  updateControls();
}

async function chooseVideo(file) {
  if (!file || state.controller) return;
  showError();
  if (file.size > 500 * 1024 * 1024) return showError('Choose a video under 500 MB. Browser processing works best with shorter sessions.');
  if (!file.size) return showError('This file is empty. Choose a video recording.');
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name)) return showError('Choose a video file such as MP4, MOV, or WebM.');
  const url = URL.createObjectURL(file);
  const probe = document.createElement('video'); probe.muted = true; probe.preload = 'auto';
  await operation(async (signal) => {
    let accepted = false;
    try {
      report('Opening video locally…', 0);
      await waitForMedia(probe, 'loadeddata', signal, () => { probe.src = url; });
      if (!Number.isFinite(probe.duration) || probe.duration <= 0 || !probe.videoWidth) throw new Error('Could not read this recording’s duration. Try an H.264 MP4.');
      invalidateReels();
      $('source-video').pause();
      $('source-video').src = url;
      if (state.url) URL.revokeObjectURL(state.url);
      state.file = file; state.url = url; state.duration = probe.duration; state.events = []; state.analyzed = false; state.classifier = 'manual'; accepted = true;
      $('drop-zone').hidden = true; $('source-panel').hidden = false;
      $('file-name').textContent = file.name;
      $('file-meta').textContent = `${formatTime(probe.duration)} · ${probe.videoWidth} × ${probe.videoHeight} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
      $('reader-status').textContent = '';
      report('Video ready. Detect shots or add them manually.', 0);
      renderReview();
    } finally {
      probe.removeAttribute('src'); probe.load();
      if (!accepted) URL.revokeObjectURL(url);
    }
  });
}

$('file-input').addEventListener('change', (event) => { const file = event.target.files[0]; event.target.value = ''; chooseVideo(file); });
if (import.meta.env.DEV) {
  import('./dev-clips.js').then(({ setupTestClips }) => {
    devClipPicker = setupTestClips(async clip => {
      let file;
      await operation(async signal => {
        report('Loading test clip…', 0);
        const response = await fetch(`/__dev/test-clips/${clip.id}`, { signal });
        if (!response.ok) throw new Error('Could not load this test clip. Reopen Test clips and try again.');
        const blob = await response.blob();
        signal.throwIfAborted();
        file = new File([blob], clip.name, { type: clip.type, lastModified: clip.lastModified });
      });
      if (file) await chooseVideo(file);
    });
    updateControls();
  }).catch(() => showError('The test clip picker could not load. Refresh to try again.'));
}
$('replace-video').addEventListener('click', () => $('file-input').click());
const dropZone = $('drop-zone');
for (const name of ['dragenter', 'dragover']) dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.remove('dragging'); });
dropZone.addEventListener('drop', event => chooseVideo(event.dataTransfer.files[0]));
document.addEventListener('dragover', event => event.preventDefault());
document.addEventListener('drop', event => event.preventDefault());
$('cancel').addEventListener('click', () => state.controller?.abort());
$('settings').addEventListener('submit', event => {
  event.preventDefault();
  if (!state.file || !$('settings').reportValidity()) return;
  const settings = options();
  operation(async signal => {
    $('source-video').pause();
    $('reader-status').textContent = '';
    const events = await analyzeVideo(state.url, { ...settings, blob: state.file, onReader: text => { $('reader-status').textContent = text; } }, signal, report);
    invalidateReels(); state.events = events; state.analyzed = true; state.classifier = 'tennis-rnn';
    renderReview();
    await compile(labels, signal);
  });
});
for (const label of labels) $(`compile-${label}`).addEventListener('click', () => operation(signal => compile([label], signal)));
$('compile-all').addEventListener('click', () => operation(signal => compile(labels, signal)));
$('add-shot').addEventListener('click', () => {
  const time = Math.min($('source-video').currentTime, Math.max(0, state.duration - .1));
  const { before, after } = options();
  state.events.push({ id: state.events.length + 1, time, start: Math.max(0, time - (Number.isFinite(before) ? before : 1.2)), end: Math.min(state.duration, time + (after > 0 ? after : .9)), stroke: 'forehand', include: true, reason: 'Added manually' });
  invalidateReels(); renderReview();
});
$('save-review').addEventListener('click', () => {
  const data = { version: 1, classifier: state.classifier, settings: options(), source: { name: state.file.name, size: state.file.size, lastModified: state.file.lastModified, duration: state.duration }, events: validateEvents(state.events, state.duration) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'stroke-review.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('load-review').addEventListener('click', () => $('review-file').click());
$('review-file').addEventListener('change', async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file || state.controller) return;
  try {
    if (file.size > 5 * 1024 * 1024) throw new Error('Shot lists must be smaller than 5 MB.');
    const data = JSON.parse(await file.text());
    if (data.version !== 1 || data.source?.name !== state.file.name || data.source?.size !== state.file.size || Math.abs(data.source?.duration - state.duration) > .1 || !Number.isFinite(data.source?.duration)) throw new Error('This shot list belongs to a different video. Open the original recording first.');
    const events = validateEvents(data.events, state.duration);
    invalidateReels(); state.events = events; state.analyzed = true; state.classifier = typeof data.classifier === 'string' ? data.classifier : 'imported'; showError(); renderReview(); report('Shot list loaded. Ready to compile.', 100);
  } catch (error) { showError(error.message); }
});
// Pause other players so two reels never play audio over one another.
for (const video of document.querySelectorAll('video')) video.addEventListener('play', () => { for (const other of document.querySelectorAll('video')) if (other !== video) other.pause(); });
window.addEventListener('beforeunload', event => { if (state.controller) { event.preventDefault(); event.returnValue = ''; } });
if (!window.Worker || !window.WebAssembly || !window.createImageBitmap || !window.OffscreenCanvas) showError('This browser is missing local video-processing features. Try a recent desktop Chrome, Edge, Firefox, or Safari.');
updateControls();
