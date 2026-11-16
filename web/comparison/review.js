const $ = id => document.getElementById(id);
const video = $('video');
const names = { forehand: 'Forehand', backhand: 'Backhand', unknown: 'Unknown', ignore: 'Ignore', serve_overhead: 'Serve / overhead', no_shot: 'No shot', unsure: 'Unsure' };
const statuses = { agree: 'Labels agree', different_label: 'Labels disagree', heuristic_only: 'Heuristic only', adaspot_only: 'AdaSpot only', tennis_rnn_only: 'Tennis RNN only', manual: 'Added by you' };
const timeText = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`;
let data, rows = [], selected, calls = {}, storageKey, stopAt = null;
let models = [{ id: 'heuristic', label: 'Motion heuristic' }, { id: 'adaspot', label: 'AdaSpot' }];
const make = (tag, text, className) => { const el = document.createElement(tag); if (text != null) el.textContent = text; if (className) el.className = className; return el; };
const error = message => { $('error').textContent = message; $('error').hidden = !message; };

function save() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({ calls, manual: rows.filter(r => r.status === 'manual') }));
    $('save-status').textContent = 'Saved in this browser. Download your review to keep a copy.';
  } catch { $('save-status').textContent = 'Browser storage unavailable. Download your review to keep your changes.'; }
}

function renderRows() {
  const filter = $('filter').value;
  $('moments').replaceChildren();
  let shown = 0;
  for (const row of rows) {
    if (filter === 'unreviewed' && calls[row.id]?.verdict) continue;
    if (filter === 'different_label' && row.status !== filter) continue;
    shown++;
    const button = make('button', null, `moment${selected?.id === row.id ? ' active' : ''}`);
    button.setAttribute('aria-label', `Review ${timeText(row.time)}, ${statuses[row.status]}`);
    button.setAttribute('aria-pressed', String(selected?.id === row.id));
    const top = make('div', null, 'moment-top');
    top.append(make('span', timeText(row.time), 'moment-time'), make('span', statuses[row.status], `badge${row.status === 'different_label' ? ' disagree' : ''}`));
    const predictions = make('div', null, 'moment-predictions');
    for (const { id: key, label } of models) {
      const box = make('div'); box.append(make('span', label), make('div', names[row[key]?.stroke] || 'No detection')); predictions.append(box);
    }
    button.append(top, predictions);
    if (calls[row.id]?.verdict) button.append(make('div', `Your call: ${names[calls[row.id].verdict]}`, 'your-call'));
    button.addEventListener('click', () => selectRow(row, true));
    $('moments').append(button);
  }
  $('empty').hidden = shown > 0;
  $('progress').textContent = `${rows.filter(r => calls[r.id]?.verdict).length} / ${rows.length} reviewed`;
}

function selectRow(row, play = false) {
  const changed = selected?.id !== row.id;
  selected = row; stopAt = null; video.pause();
  $('moment-label').textContent = `MOMENT ${rows.indexOf(row) + 1} OF ${rows.length}`;
  $('moment-title').textContent = `${timeText(row.time)} · ${row.status === 'different_label' ? 'Different model calls' : 'Check this swing'}`;
  $('moment-status').textContent = statuses[row.status];
  for (const { id: key } of models) {
    const event = row[key];
    $(`${key}-label`).textContent = event ? names[event.stroke] : 'No detection';
    $(`${key}-detail`).textContent = event ? `${timeText(event.time)} · ${event.reason || ''}` : 'This classifier did not propose a shot here.';
  }
  $('note').value = calls[row.id]?.note || '';
  for (const button of $('verdicts').children) { const active = calls[row.id]?.verdict === button.dataset.verdict; button.classList.toggle('selected', active); button.setAttribute('aria-pressed', String(active)); }
  $('previous').disabled = rows.indexOf(row) === 0;
  $('next').disabled = rows.indexOf(row) === rows.length - 1;
  $('clear-call').disabled = !calls[row.id]?.verdict;
  renderRows();
  if (play) replay();
  else if (changed && row.status !== 'manual') {
    if (video.readyState >= 1) video.currentTime = bounds()[0];
    else video.addEventListener('loadedmetadata', () => { if (selected?.id === row.id) video.currentTime = bounds()[0]; }, { once: true });
  }
}

function bounds() {
  const times = [selected.time, ...models.flatMap(({id}) => selected[id] ? [selected[id].time] : [])];
  return [Math.max(0, Math.min(...times) - 1.5), Math.min(data.source.duration, Math.max(...times) + 1.2)];
}

function replay() {
  if (!selected) return;
  const [start, end] = bounds();
  stopAt = end; video.currentTime = start;
  video.play().catch(() => error('Press play on the video to start playback.'));
}

video.addEventListener('timeupdate', () => {
  $('clock').textContent = timeText(video.currentTime);
  if (stopAt !== null && video.currentTime >= stopAt) {
    if ($('loop').checked) replay();
    else { video.pause(); stopAt = null; }
  }
});
// Native playback after an excerpt has paused can continue through the whole recording.
video.addEventListener('error', () => error('This browser could not play the video. Use an H.264 MP4, or check that the local review server is still running.'));
$('speed').addEventListener('change', () => { video.playbackRate = Number($('speed').value); });
for (const [id, direction] of [['back-frame', -1], ['forward-frame', 1]]) $(id).addEventListener('click', () => { if (!data) return; video.pause(); stopAt = null; video.currentTime = Math.max(0, Math.min(data.source.duration, video.currentTime + direction * .1)); });
$('replay').addEventListener('click', replay);
for (const [id, direction] of [['previous', -1], ['next', 1]]) $(id).addEventListener('click', () => { const row = rows[rows.indexOf(selected) + direction]; if (row) selectRow(row, true); });
$('filter').addEventListener('change', renderRows);
$('first-disagreement').addEventListener('click', () => { const row = rows.find(r => r.status === 'different_label'); if (row) { $('filter').value = 'all'; selectRow(row, true); } });
for (const button of $('verdicts').children) button.addEventListener('click', () => {
  if (!selected) return;
  calls[selected.id] = { ...calls[selected.id], verdict: button.dataset.verdict };
  save(); selectRow(selected);
});
$('clear-call').addEventListener('click', () => {
  if (!selected) return;
  delete calls[selected.id]?.verdict;
  save(); selectRow(selected);
});
$('note').addEventListener('input', () => { if (selected) { calls[selected.id] = { ...calls[selected.id], note: $('note').value }; save(); } });
$('add').addEventListener('click', () => {
  if (!data) return;
  video.pause(); stopAt = null;
  const row = { id: `manual-${crypto.randomUUID()}`, time: Math.min(video.currentTime, data.source.duration), status: 'manual', heuristic: null, adaspot: null };
  rows.push(row); rows.sort((a, b) => a.time - b.time); $('filter').value = 'all'; save(); selectRow(row);
});
$('download').addEventListener('click', () => {
  const payload = { version: 1, type: 'classifier-comparison-review', source: data.source, settings: data.settings, models,
    reviewedAt: new Date().toISOString(), moments: rows.map(r => ({ ...r, review: calls[r.id] || null })) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const link = make('a'); link.href = url; link.download = 'classifier-comparison-review.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});

try {
  const response = await fetch('/api/comparison');
  if (!response.ok) throw new Error('Could not load the comparison. Check the local review server.');
  data = await response.json(); rows = data.rows; storageKey = `stroke-comparison:${data.reviewId}`;
  models = data.models || models;
  $('model-summary').textContent = models.map(({id, label}) => `${label}: ${rows.filter(row => row[id]).length} proposals`).join(' · ');
  $('predictions').replaceChildren(...models.map(({id, label}) => {
    const box = make('div'); const name = make('span', label, 'model-name');
    const prediction = make('strong', '—'); prediction.id = `${id}-label`;
    const detail = make('small'); detail.id = `${id}-detail`;
    box.append(name, prediction, detail); return box;
  }));
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
    calls = saved.calls || {};
    if (Array.isArray(saved.manual)) rows.push(...saved.manual.filter(r => r.status === 'manual' && Number.isFinite(r.time) && r.time >= 0 && r.time <= data.source.duration));
  } catch { /* A missing/invalid browser draft does not prevent video review. */ }
  rows.sort((a, b) => a.time - b.time);
  video.playbackRate = .5;
  $('player').textContent = data.settings.adaspot.player || 'near';
  $('run-details').textContent = `Recording: ${data.source.name}. AdaSpot threshold: ${data.settings.adaspot.adaspot_threshold ?? 'not recorded'}; gap: ${data.settings.adaspot.adaspot_min_gap ?? 'not recorded'} s. Heuristic: ${data.settings.heuristic.model || 'YOLO pose'}.`;
  if (data.settings['tennis-rnn']) $('run-details').textContent += ` Tennis RNN threshold: ${data.settings['tennis-rnn'].tennis_rnn_threshold}; gap: ${data.settings['tennis-rnn'].tennis_rnn_min_gap} s. 30-frame MoveNet + GRU, timestamps at window centers.`;
  $('first-disagreement').disabled = !rows.some(r => r.status === 'different_label');
  $('download').disabled = false;
  if (rows.length) selectRow(rows.find(r => r.status === 'different_label') || rows[0]);
  else { $('moment-title').textContent = 'No detections. Play the video and add missed shots.'; renderRows(); }
} catch (e) { error(e.message); }
