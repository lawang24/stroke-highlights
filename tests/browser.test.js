import test from 'node:test';
import assert from 'node:assert/strict';
import { compilationRanges, validateEvents } from '../web/detection.js';
import {classifyWindow, poseFeatures, decodeScores, trackPose} from '../web/tennis-rnn.js';
import {readFileSync} from 'node:fs';

test('browser GRU agrees with Keras for the pinned pretrained weights', () => {
  const {weights} = JSON.parse(readFileSync(new URL('../models/tennis-rnn/weights.json', import.meta.url)));
  const {seeds, probabilities} = JSON.parse(readFileSync(new URL('./fixtures/tennis-rnn-parity.json', import.meta.url)));
  seeds.forEach((seed, index) => {
    const sequence = Array.from({length:30}, (_,t) => Float32Array.from({length:26}, (_,j) => ((t*26+j)*17+seed)%101/100));
    const actual = classifyWindow(sequence, weights);
    actual.forEach((p, i) => assert.ok(Math.abs(p-probabilities[index][i]) < 2e-6));
  });
});
test('preserves 13-joint order and reflects only x for left-handers', () => {
  const points = Float32Array.from({length:51}, (_,i) => i/51);
  const original = poseFeatures(points), reflected = poseFeatures(points, true);
  assert.equal(original.length,26);
  assert.equal(original[2], points[15]);
  for(let i=0;i<26;i++) assert.ok(Math.abs(reflected[i] - (i%2 ? 1-original[i] : original[i])) < 1e-7);
});
test('neutral and tracking gaps split runs; serves stay excluded', () => {
  const samples = [{time:1,scores:[0,.999,.001,0]}, {time:1.03,scores:[0,.995,.005,0]},
    {time:1.06,scores:[0,0,1,0]}, {time:4,scores:[0,0,0,1]}, {time:7,scores:[1,0,0,0]}];
  const events = decodeScores(samples,8);
  assert.deepEqual(events.map(e=>e.stroke), ['forehand','unknown','backhand']);
  assert.equal(events[1].include,false);
  assert.deepEqual(decodeScores([],8),[]);
  assert.equal(decodeScores([{time:1,scores:[1,0,0,0]}, {time:4,scores:[1,0,0,0]}],8).length,2);
});
test('invalid or small player poses reset the crop', () => {
  const points = Float32Array.from({length:51}, (_,i) => i%3===2 ? .99 : .5);
  assert.deepEqual(trackPose(points,null,1920,1080),{valid:false,crop:null});
  points[0]=NaN;
  assert.equal(trackPose(points,null,1920,1080).valid,false);
});

test('merges overlapping excerpts and filters excluded and other categories', () => {
  const events = [[3, 5, 'forehand', true], [1, 4, 'forehand', true], [7, 8, 'backhand', true], [9, 10, 'forehand', false]].map(([start, end, stroke, include]) => ({ start, end, stroke, include }));
  assert.deepEqual(compilationRanges(events, 'forehand'), [[1, 5]]);
  assert.deepEqual(compilationRanges(events, 'backhand'), [[7, 8]]);
  assert.deepEqual(compilationRanges([], 'forehand'), []);
});
test('validates edits before export, including non-finite values and invalid ranges', () => {
  const good = { start: 0, end: 2, stroke: 'forehand', include: true };
  assert.equal(validateEvents([good], 4)[0].id, 1);
  for (const patch of [{ start: NaN }, { end: Infinity }, { start: -1 }, { end: 5 }, { end: 0 }, { stroke: 'serve' }, { include: 'false' }]) assert.throws(() => validateEvents([{ ...good, ...patch }], 4));
  assert.throws(() => validateEvents({}, 4));
});
