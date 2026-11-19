import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
// The published package's unbundled entry omits its WASM client; use its ESM bundle.
import {loadTFLiteModel, setWasmPath} from '@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js';
import {classifyWindow, poseFeatures, smoothArmOcclusions, trackPose} from './tennis-rnn.js';

let model, weights, options, canvas, crop = null, sequence = [], lastPrediction = null;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      await tf.setBackend('cpu'); await tf.ready();
      setWasmPath(`${data.base}runtime/tflite/`);
      const response = await fetch(`${data.base}runtime/tennis-rnn/weights.json`);
      if (!response.ok) throw new Error('Tennis RNN weights are missing. Run npm run setup.');
      const payload = await response.json(); weights = payload.weights; options = data.options;
      model = await loadTFLiteModel(`${data.base}runtime/tennis-rnn/movenet.tflite`, {numThreads: 1});
      crop = null; sequence = []; lastPrediction = null;
      self.postMessage({type: 'ready'});
    } else if (data.type === 'frame') {
      const {bitmap, time} = data;
      try {
        const width = bitmap.width, height = bitmap.height;
        canvas ??= new OffscreenCanvas(width, height);
        const context = canvas.getContext('2d', {willReadFrequently: true});
        context.setTransform(options.mirrored ? -1 : 1, 0, 0, 1, options.mirrored ? width : 0, 0);
        context.drawImage(bitmap, 0, 0);
        const [x,y,w,h] = crop || [0,0,width,height];
        const pixels = context.getImageData(x,y,w,h);
        const points = tf.tidy(() => {
          const rgb = tf.browser.fromPixels(pixels);
          const scale = 192 / Math.max(w,h), nh = Math.floor(h*scale), nw = Math.floor(w*scale);
          const top = Math.floor((192-h*scale)/2), left = Math.floor((192-w*scale)/2);
          const resized = tf.image.resizeBilinear(rgb, [nh,nw], false, true);
          // Preserve the training extractor's BGR convention at the smaller size.
          const bgr = tf.gather(resized, [2,1,0], 2);
          const input = tf.cast(tf.expandDims(tf.pad(bgr, [[top,192-nh-top],[left,192-nw-left],[0,0]]), 0), 'int32');
          const output = model.predict(input);
          const tensor = output instanceof tf.Tensor ? output : Object.values(output)[0];
          return Float32Array.from(tensor.dataSync());
        });
        const tracked = trackPose(points, crop, width, height); crop = tracked.crop;
        let prediction = null;
        if (!tracked.valid) { sequence = []; lastPrediction = null; }
        else {
          const leftHanded = options.handedness === 'left';
          sequence.push({time, points, features: poseFeatures(points, leftHanded)});
          if (sequence.length === 30) {
            const smoothed = smoothArmOcclusions(sequence.map(s => s.points));
            const feats = smoothed.map(p => poseFeatures(p, leftHanded));
            prediction = {time: (sequence[0].time + time)/2, scores: classifyWindow(feats, weights)};
            lastPrediction = prediction;
            sequence.shift();
          }
        }
        self.postMessage({type: 'prediction', prediction});
      } finally { bitmap.close(); }
    } else if (data.type === 'flush') {
      const predictions = [];
      // Complete trailing 30-frame windows if a stroke is active at the clip boundary.
      if (sequence.length === 29 && lastPrediction && Math.max(lastPrediction.scores[0], lastPrediction.scores[1]) >= .15) {
        const leftHanded = options.handedness === 'left';
        const tail = sequence.slice();
        const last = tail[tail.length - 1];
        for (let k = 1; k <= 15; k++) {
          const t = last.time + k / 30;
          tail.push({time: t, points: last.points, features: last.features});
          const windowSeq = tail.slice(tail.length - 30);
          const smoothed = smoothArmOcclusions(windowSeq.map(s => s.points));
          const feats = smoothed.map(p => poseFeatures(p, leftHanded));
          const center = Math.min(data.duration ?? t, (windowSeq[0].time + t) / 2);
          predictions.push({time: center, scores: classifyWindow(feats, weights)});
        }
      }
      self.postMessage({type: 'flushed', predictions});
    }
  } catch (error) { console.error(error); self.postMessage({type: 'error', message: error.message || String(error)}); }
};
