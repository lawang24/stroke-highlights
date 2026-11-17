import test from 'node:test';
import assert from 'node:assert/strict';
import {chromiumSeekMicroseconds,atOrBefore,seekTick,selectable} from '../web/sequential-decoder.js';

test('preserves Chromium double conversion at the observed seek boundary',()=>{
  assert.equal(Math.trunc(61/30*1e6),2033333);
  assert.equal(chromiumSeekMicroseconds(61/30),2033332);
  assert.equal(atOrBefore(61/30,61/30),false);
  assert.equal(atOrBefore(60/30,61/30),true);
  assert.equal(atOrBefore(1/30,1/30),true);
  assert.equal(atOrBefore(2/30,2/30),false);
});

test('fractional keyframe follows native demuxer timebase rounding without advancing non-key frames',()=>{
 const t=500/30,resolution=15360;
 assert.equal(chromiumSeekMicroseconds(t),16666666);
 assert.equal(seekTick(t,resolution),256000);
 assert.equal(atOrBefore(t,t),false);
 assert.equal(selectable(t,t,resolution,true),true);
 assert.equal(selectable(t,t,resolution,false),false);
 assert.equal(selectable(t,t-.001,resolution,true),false);
});
