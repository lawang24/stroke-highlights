import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
// The published package's unbundled entry omits its WASM client; use its ESM bundle.
import {loadTFLiteModel, setWasmPath} from '@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js';
import {classifyWindow, poseFeatures, trackPose} from './tennis-rnn.js';

let model, weights, options, canvas, crop = null, sequence = [];
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      await tf.setBackend('cpu'); await tf.ready();
      setWasmPath(`${data.base}runtime/tflite/`);
      const response = await fetch(`${data.base}runtime/tennis-rnn/weights.json`);
      if (!response.ok) throw new Error('Tennis RNN weights are missing. Run npm run setup.');
      const payload = await response.json(); weights = payload.weights; options = data.options;
      model = await loadTFLiteModel(`${data.base}runtime/tennis-rnn/movenet.tflite`, {numThreads: 1});
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
        if (!tracked.valid) sequence = [];
        else {
          sequence.push({time, features: poseFeatures(points, options.handedness === 'left')});
          if (sequence.length === 30) {
            prediction = {time: (sequence[0].time + time)/2, scores: classifyWindow(sequence.map(s => s.features), weights)};
            sequence.shift();
          }
        }
        self.postMessage({type: 'prediction', prediction});
      } finally { bitmap.close(); }
    }
  } catch (error) { console.error(error); self.postMessage({type: 'error', message: error.message || String(error)}); }
};
