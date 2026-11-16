# Classifier experiments and decisions

## 2026-11-15 — Select Tennis RNN as the only supported classifier

**Decision:** use Antoine Keller's pretrained MoveNet + GRU Tennis RNN in the
Python CLI and browser. Remove the YOLO/MediaPipe motion heuristics and AdaSpot
inference paths, their selectors, model setup scripts, and dependencies. Retain
historical prediction files and the comparison viewer for inspecting experiments.
The user accepted this choice after trying the review workflow.

This is a practical product decision, **not a measured accuracy win**. We tested
one short recording and did not finish a ground-truth benchmark. Matching a second
model's label does not establish correctness; fewer proposals do not establish
fewer false positives or better recall.

### Recording and evaluation

- Source: `outputs/adaspot-comparison-input.mp4`, normalized from `small-clip.mov`.
- Duration: 35.1667 seconds; 1,055 frames at 30 FPS; 1920×1080; 41,602,524 bytes.
- Fixed, low courtside/baseline camera; near player, right-handed, not mirrored.
- We normalized the MOV because its reported and decoded frame counts differed.
- Comparison tolerance: ±0.5 seconds, one-to-one matching.
- Raw video remains local/ignored. Small prediction/settings snapshots are stored
  in [the experiment folder](2026-11-15-classifier-selection/).

### What we tried

| Candidate / configuration | Result on this recording | Assessment and disposition |
| --- | --- | --- |
| YOLO pose + wrist-motion rules | 7 proposals: 2 forehand, 3 backhand, 2 unknown | Useful initial baseline; relies on wrist position, swing direction, and camera/handedness assumptions. Removed from active inference. |
| MediaPipe + browser motion rules | Implemented and checked using synthetic motion tests and browser smoke tests; not separately benchmarked on this recording | Different pose estimator from Python. No measured accuracy parity. Removed rather than retaining a hidden alternative browser classifier. |
| AdaSpot F3Set-small, threshold 0.30, min gap 0.5 s | 0 proposals | Default event × court-side scores were too low for useful output on this clip. |
| AdaSpot F3Set-small, exploratory threshold 0.05, min gap 0.5 s | 3 proposals: 2 forehand, 1 backhand | Limited overlap with baseline; camera/domain mismatch is a hypothesis, not proven. Removed from active inference. |
| Tennis RNN, threshold 0.98, min gap 2.0 s | 5 proposals: 3 forehand, 2 backhand | All five match the heuristic's five labeled moments; four labels agree, one differs. Selected for further use and evaluation. |

Tennis RNN proposals: **7.017 s backhand, 10.650 s forehand, 23.650 s
forehand, 27.817 s backhand, 31.017 s forehand**. At approximately 10.6 s,
Tennis RNN and AdaSpot predict forehand; the heuristic predicts backhand. This is
the key label disagreement to validate manually. The heuristic's two extra unknown
events and AdaSpot's unmatched proposals are not automatically false detections.

### Model provenance and implementation details

- [Tennis shot recognition](https://github.com/antoinekeller/tennis_shot_recognition):
  commit `6fdfd938f8d6893e2c926eb12fcc7f9bfc3e792f`, `tennis_rnn.h5` SHA-256
  `17dcbc5aba7ac4b88378bc0f8e02ed548de7c749f58c6ccb4969daa39020ade6`.
  MoveNet Lightning float16 v4 SHA-256
  `0fac2226112d0371903ca86e3853cec24ef603a0b2f96f589b180f0ebdd135ab`.
  GRU(24) → Dense(8, ReLU) → Dense(4, softmax); class order backhand,
  forehand, neutral, serve. The upstream notebook reuses training folders in its
  test data, so its reported accuracy is not independent evidence.
- Tennis RNN preprocessing retains the upstream BGR convention, letterboxing to
  192×192, crop-local `(y,x)` features for 13 joints, 1.3× square crop tracking,
  and x reflection for left-handers. Our adapter samples 30 Hz, requires complete
  valid 30-frame sequences, timestamps them at their centers, and consolidates
  confident runs with cross-class suppression. Neutral creates no proposal;
  serves become excluded `unknown` events with the reason preserved.
- [AdaSpot](https://github.com/arturxe2/AdaSpot): source commit
  `6a46bbc0f4b3120d75ebcd6179c7275449388431`; [F3Set_small-1 checkpoint](https://huggingface.co/arturxe/AdaSpot/tree/main/F3Set_small-1)
  revision `5bdac671e2a6eae860d4d097758c977149559449`, SHA-256
  `d502016ff6f12290a1e3cace9108e08866603aa19bfca0c4abed56419f5068be`.
  Adapter used 100-frame RGB windows, 50-frame overlap, 796×448 frames, and
  foreground × court-side peaks plus the hand head. This differed from upstream's
  1,108-class benchmark postprocessing. CPU inference worked; full-window MPS
  inference caused heavy memory pressure on this 8 GB Mac.
- F3ED and E2E-Spot were researched as alternatives, but were not integrated or
  run here. Published benchmark results were not treated as local measurements.

### What would change this decision

Label complete recordings (including missed shots), split evaluation by recording,
and measure per-class precision/recall, label accuracy at matched contacts, timing
error, and runtime. Include different players, handedness, and camera angles.
Revisit the default if another model wins on those held-out clips. Browser/Python
pixel decoding can differ even with the same model weights; check both runtimes.

### Default-only integration verification

After removing the alternatives, the default CLI rerun reproduced all five
original Tennis RNN events exactly. The browser worker analyzed the full same
recording and returned the same five labels, with a maximum timestamp difference
of 0.067 seconds. See [the browser verification result](2026-11-15-classifier-selection/tennis-rnn-browser-verification.json).
The browser's numeric GRU passed comparison with Keras reference probabilities;
14 Python tests, 6 browser-logic tests, and the production build passed. The full
browser inference check used a local fixture page invoking the app's video-analysis
function; the embedded browser file chooser could not be automated in this check.
The app's file-selection logic was unchanged.

## Future entries

For each experiment, append the date, hypothesis, exact source/model revisions,
data split and camera setup, settings, results/artifact links, limitations, and
the decision. Keep old entries as the history rather than rewriting the outcome.
