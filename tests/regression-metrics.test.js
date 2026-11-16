import test from 'node:test';
import assert from 'node:assert/strict';
import {clipLabel, evaluate} from '../scripts/regression-metrics.mjs';

test('tracking failures and neutral clips are both negative; mixed positives stay visible', () => {
  assert.equal(clipLabel({classifiedWindows: 0, events: []}), 'no_shot');
  assert.equal(clipLabel({classifiedWindows: 10, events: []}), 'no_shot');
  assert.equal(clipLabel({classifiedWindows: 10, events: [{stroke:'unknown'}]}), 'no_shot');
  assert.equal(clipLabel({classifiedWindows: 10, events: [{stroke:'backhand'}, {stroke:'forehand'}]}), 'mixed');
});

test('unsure joins negatives; missing tracking still misses a human positive', () => {
  const frozen = {ground_truth: [
    {clip_id:'a', included:true, label:'backhand'},
    {clip_id:'b', included:true, label:'no_shot'},
    {clip_id:'c', included:false, label:'unsure'},
    {clip_id:'d', included:true, label:'forehand'},
  ]};
  const runs = [{clipId:'a', classifiedWindows:30, events:[{stroke:'backhand', time:999}]},
    ...['b', 'c', 'd'].map(clipId => ({clipId, classifiedWindows:0, events:[]}))];
  const metrics = evaluate(frozen, runs);
  assert.equal(metrics.accuracy, .75);
  assert.equal(metrics.trackingFailures, 3);
  assert.equal(metrics.confusion.no_shot.no_shot, 2);
  assert.equal(metrics.perClass.backhand.recall, 1);
  assert.equal(metrics.perClass.forehand.falseNegatives, 1);
  assert.equal(metrics.perClass.no_shot.precision, 2/3);
  assert.throws(() => evaluate(frozen, [runs[0]]), /exactly one/);
  assert.throws(() => evaluate(frozen, [runs[0], runs[0], runs[2], runs[3]]), /exactly one/);
  assert.throws(() => evaluate(frozen, [runs[0], {...runs[1], error:'failed'}, ...runs.slice(2)]), /Invalid inference/);
});

test('mixed detections count as both positives, including false positives on negatives', () => {
  const frozen = {ground_truth:[{clip_id:'a', included:true, label:'no_shot'}]};
  const metrics = evaluate(frozen, [{clipId:'a', classifiedWindows:30, events:[{stroke:'forehand'}, {stroke:'backhand'}]}]);
  assert.equal(metrics.accuracy, 0);
  assert.equal(metrics.perClass.forehand.falsePositives, 1);
  assert.equal(metrics.perClass.backhand.falsePositives, 1);
});

test('detecting the wrong stroke still fails the expected positive-class recall gate', () => {
  const frozen = {ground_truth:[{clip_id:'positive', included:true, label:'forehand'}]};
  const metrics = evaluate(frozen, [{clipId:'positive', classifiedWindows:30, events:[{stroke:'backhand'}]}]);
  assert.equal(metrics.perClass.forehand.falseNegatives, 1);
  assert.equal(metrics.perClass.forehand.recall, 0);
  assert.equal(metrics.perClass.backhand.falsePositives, 1);
  assert.equal(metrics.mismatches[0].clipId, 'positive');
});
