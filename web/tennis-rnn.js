// Numeric inference for the pinned Keras reset_after=True GRU. No retraining.
export const CLASSES = ['backhand', 'forehand', 'neutral', 'serve'];
export const JOINTS = [0, ...Array.from({length: 12}, (_, i) => i + 5)];
const sigmoid = x => 1 / (1 + Math.exp(-x));

export function poseFeatures(points, leftHanded = false) {
  return Float32Array.from(JOINTS.flatMap(i => [points[i * 3], leftHanded ? 1 - points[i * 3 + 1] : points[i * 3 + 1]]));
}

export function classifyWindow(sequence, weights) {
  if (sequence.length !== 30 || sequence.some(frame => frame.length !== 26)) throw new Error('Tennis RNN needs 30 × 26 features.');
  let h = new Float32Array(24);
  for (const frame of sequence) {
    const xp = Float32Array.from(weights.bias.slice(0, 72));
    const hp = Float32Array.from(weights.bias.slice(72));
    for (let j = 0; j < 72; j++) {
      let x = xp[j], r = hp[j];
      for (let i = 0; i < 26; i++) x += frame[i] * weights.kernel[i * 72 + j];
      for (let i = 0; i < 24; i++) r += h[i] * weights.recurrent[i * 72 + j];
      xp[j] = x; hp[j] = r;
    }
    const next = new Float32Array(24);
    for (let j = 0; j < 24; j++) {
      const z = sigmoid(xp[j] + hp[j]);
      const r = sigmoid(xp[24 + j] + hp[24 + j]);
      next[j] = z * h[j] + (1 - z) * Math.tanh(xp[48 + j] + r * hp[48 + j]);
    }
    h = next;
  }
  const dense = new Float32Array(8);
  for (let j = 0; j < 8; j++) {
    let sum = weights.denseBias[j];
    for (let i = 0; i < 24; i++) sum += h[i] * weights.denseKernel[i * 8 + j];
    dense[j] = Math.max(0, sum);
  }
  const logits = weights.outputBias.map((b, j) => {
    for (let i = 0; i < 8; i++) b += dense[i] * weights.outputKernel[i * 4 + j];
    return b;
  });
  const peak = Math.max(...logits), exps = logits.map(x => Math.exp(x - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / total);
}

export function trackPose(points, crop, width, height, minHeight = .2) {
  const [x, y, w, h] = crop || [0, 0, width, height], scale = Math.max(w, h);
  const pixels = Array.from({length: 17}, (_, i) => [points[i*3] * scale + y - (scale-h)/2, points[i*3+1] * scale + x - (scale-w)/2]);
  const ys = JOINTS.map(i => pixels[i][0]);
  const confidence = JOINTS.reduce((s, i) => s + points[i*3+2], 0) / JOINTS.length;
  if (!points.every(Number.isFinite) || confidence < .3 || Math.max(...ys)/height < .55 || (Math.max(...ys)-Math.min(...ys))/height < minHeight) return { valid: false, crop: null };
  const lows = [0, 1].map(j => Math.min(...pixels.map(p => p[j])));
  const highs = [0, 1].map(j => Math.max(...pixels.map(p => p[j])));
  const side = Math.floor(Math.max(highs[0]-lows[0], highs[1]-lows[1]) * 1.3);
  if (side < 10 || side >= Math.min(width, height)) return { valid: true, crop: null };
  const clamp = (v, max) => Math.floor(Math.max(0, Math.min(max, v)));
  return { valid: true, crop: [clamp((highs[1]+lows[1]-side)/2, width-side), clamp((highs[0]+lows[0]-side)/2, height-side), side, side] };
}

export function suppressPostBackhandRecovery(samples, { bhTrigger = .35, neutralReset = .75, resetFrames = 4, maxRecoverySec = 1.5 } = {}) {
  let inBhRecovery = false, lastBhTime = -Infinity, neutralStreak = 0, prevTime = null;
  return samples.map(({ time, scores }) => {
    if (prevTime !== null && time - prevTime > .25) { inBhRecovery = false; neutralStreak = 0; }
    prevTime = time;
    const [bh, fh, ne, sv] = scores;
    const label = scores.indexOf(Math.max(...scores));
    if (label === 0 && bh >= bhTrigger) {
      inBhRecovery = true; lastBhTime = time; neutralStreak = 0;
      return fh > 0 ? { time, scores: [bh, 0, ne + fh, sv] } : { time, scores };
    }
    if (inBhRecovery) {
      if (time - lastBhTime > maxRecoverySec) { inBhRecovery = false; neutralStreak = 0; return { time, scores }; }
      if (ne >= neutralReset && label === 2) {
        neutralStreak += 1;
        if (neutralStreak >= resetFrames) { inBhRecovery = false; neutralStreak = 0; return { time, scores }; }
      } else {
        neutralStreak = 0;
      }
      if (fh > 0) return { time, scores: [bh, 0, ne + fh, sv] };
    }
    return { time, scores };
  });
}

export function decodeScores(samples, duration, { threshold = .98, minGap = 2, before = 1.2, after = .9, sustainedMinFrames = 8, sustainedThreshold = .80 } = {}) {
  const cleaned = suppressPostBackhandRecovery(samples);
  const effGate = Math.min(threshold, sustainedThreshold);
  const candidates = []; let run = null, runLength = 0, previous = null;
  const finish = () => {
    if (run && (run.score > threshold || (runLength >= sustainedMinFrames && run.score > effGate))) candidates.push(run);
    run = null; runLength = 0;
  };
  for (const {time, scores} of cleaned) {
    const score = Math.max(...scores), label = scores.indexOf(score);
    if (label === 2 || score <= effGate) { finish(); previous = null; continue; }
    if (run && (run.label !== label || time - previous > 1.5/30)) finish();
    runLength += 1;
    if (!run || score > run.score) run = {time, label, score};
    previous = time;
  }
  finish();
  const selected = [];
  for (const e of candidates.sort((a,b) => b.score-a.score)) if (selected.every(other => Math.abs(other.time-e.time) >= minGap)) selected.push(e);
  const round = t => Math.round(t * 1000) / 1000;
  return selected.sort((a,b) => a.time-b.time).map(({time, label, score}, i) => ({
    id: i+1, time: round(time), start: round(Math.max(0, time-before)), end: round(Math.min(duration, time+after)),
    stroke: label === 3 ? 'unknown' : CLASSES[label], include: label !== 3, score,
    reason: `Tennis RNN: ${CLASSES[label]}; 30-frame MoveNet sequence`,
  }));
}
