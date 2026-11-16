// Score only forehand/backhand positives. Keep tracking diagnostics separately.
const positiveLabels = ['forehand', 'backhand'];
export const scoringPolicy = 'forehand-backhand-positives-v1';
export function scoredTruth(frozen) {
  // Frozen v2 marked human unsure excluded under the previous scoring policy.
  return frozen.ground_truth.filter(row => row.included || row.label === 'unsure');
}
const positives = run => positiveLabels.filter(label => run.events.some(event => event.stroke === label));
export function clipLabel(run) {
  const found = positives(run);
  // Preserve mixed positives so they cannot pass as a negative clip.
  return found.length === 2 ? 'mixed' : found[0] || 'no_shot';
}

export function evaluate(frozen, runs) {
  const truth = scoredTruth(frozen);
  const ids = runs.map(run => run.clipId);
  if (!truth.length || new Set(ids).size !== ids.length || runs.length !== truth.length || truth.some(row => !ids.includes(row.clip_id))) {
    throw new Error('Evaluation requires exactly one result for every scored clip');
  }
  const labels = ['forehand', 'backhand', 'no_shot'];
  const confusion = Object.fromEntries(labels.map(label => [label, Object.fromEntries([...labels, 'mixed'].map(predicted => [predicted, 0]))]));
  const clips = truth.map(row => {
    const run = runs.find(run => run.clipId === row.clip_id);
    if (run.error || !Array.isArray(run.events) || !Number.isInteger(run.classifiedWindows) || run.classifiedWindows < 0) throw new Error(`Invalid inference result: ${row.clip_id}`);
    const predicted = clipLabel(run), expected = row.label === 'unsure' ? 'no_shot' : row.label;
    if (!labels.includes(expected)) throw new Error(`Unknown human label: ${row.label}`);
    confusion[expected][predicted]++;
    return {clipId: row.clip_id, humanLabel: row.label, expected, predicted, predictedPositives: positives(run), correct: predicted === expected,
      classifiedWindows: run.classifiedWindows, reason: !run.classifiedWindows ? 'insufficient_tracked_poses' : predicted === 'mixed' ? 'mixed_positive_shots' : null};
  });
  const correct = clips.filter(row => row.correct).length;
  const perClass = Object.fromEntries(labels.map(label => {
    const tp = clips.filter(row => row.expected === label && (label === 'no_shot' ? row.predicted === label : row.predictedPositives.includes(label))).length;
    const support = clips.filter(row => row.expected === label).length;
    const predicted = clips.filter(row => label === 'no_shot' ? row.predicted === label : row.predictedPositives.includes(label)).length;
    return [label, {support, correct: tp, predicted, falsePositives: predicted - tp, falseNegatives: support - tp,
      precision: predicted ? tp / predicted : null, recall: support ? tp / support : null}];
  }));
  return {scoringPolicy, total: clips.length, correct, accuracy: correct / clips.length,
    trackingFailures: clips.filter(row => !row.classifiedWindows).length,
    policy: 'Human unsure and no_shot are negative. No forehand/backhand events means negative, including tracking failures. Any forehand/backhand detection counts as a positive; mixed positives cannot pass as negative. Clip labels only; no detection/contact timing checks.',
    perClass, confusion, clips, mismatches: clips.filter(row => !row.correct)};
}
