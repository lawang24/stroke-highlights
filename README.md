# Stroke Highlights

Review tennis shots and create separate forehand/backhand MP4 compilations.
**Tennis RNN is the only classifier**, in both the browser and Python CLI.
Video processing stays on your device; videos, frames, poses, and shot lists are
not uploaded.

In `npm run dev`, **Test clips** beside Video opens a picker for the local
recordings (`small-clip.mov`, `inputs/long-demo-last-5-minutes.mp4`, and
`outputs/adaspot-comparison-input.mp4`) and the clips in
`outputs/regression-review-v1/clips/`. Loading one replaces the current video and
shot list. Missing files are omitted. The picker and clip endpoint are dev-only;
recordings are never copied into the production build.

The model choice, retired approaches, exact experiment settings, and measured
results are recorded in [experiments/DECISIONS.md](experiments/DECISIONS.md).

## Setup

Requirements: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). The browser
build also requires Node.js 22+ and npm. CLI video export requires FFmpeg
(`brew install ffmpeg` on macOS).

```bash
uv sync
python3 scripts/setup_tennis_rnn.py
npm ci
npm run dev
```

Open the Vite URL, normally **http://127.0.0.1:5173**. Setup downloads and verifies
MoveNet Lightning float16 v4 and the pretrained GRU checkpoint (about 4.9 MB
combined). Browser setup exports the GRU's numeric weights, bundles the worker,
and copies local WASM assets. Generated models/runtime files are ignored by Git.
Python is needed during setup/build; the running website is static and needs no
Python server, GPU, credentials, or API key.

## Browser workflow

1. Choose a recording. The full video becomes available immediately.
2. Set the playing hand, clip padding, and mirrored-video option. Tennis RNN
   currently follows only the **near-court player**.
3. Click **Detect shots**. The worker samples at 30 Hz, runs MoveNet and the
   30-frame GRU, and automatically compiles included forehands and backhands.
4. Review each excerpt. Adjust its start/end or category, exclude a false
   detection, or add a missed shot at the playhead. Editing invalidates old reels.
5. Compile again and download each MP4. Overlapping excerpts are merged, source
   audio is retained when present, and browser export uses 30 FPS.
6. **Save shot list** downloads JSON. To resume, reopen the original recording and
   use **Load shot list**. Reloading the main app otherwise clears its session.

Manual shot review and compilation also work without detection. Cancel stops the
active worker and retains the video and existing shot list.

Fast sequential decoding is the default, with no detection-mode selector. Videos
outside the verified fast path automatically restart with the compatibility reader
and fresh tracking state. Cancellation stops analysis without restarting it.
The model, 30 Hz sample grid, thresholds and shot scoring are unchanged. See the
[default-reader validation](experiments/2026-11-16-fast-default-report.json).

The browser uses the same pinned MoveNet model, GRU weights, feature order,
handedness transform, thresholds, and postprocessing as Python. The GRU arithmetic
is tested against Keras reference probabilities. Browser/OpenCV video decoding
and WASM/native pose inference can differ, so end-to-end results need not be
bit-for-bit identical.

Use a fixed camera behind the baseline with the near player fully visible.
The model distinguishes backhand, forehand, neutral, and serve. Neutral produces
no event. Serves are excluded `unknown` events with `serve` retained in the note;
only forehands/backhands enter compilations. This remains experimental recognition,
not a substitute for reviewing the video.

### Browser limits and hosting

- Static app with local `blob:` video URLs, no upload endpoint, analytics, database,
  or remote inference. All model/WASM requests go to the same website.
- Requires a current desktop browser with WebAssembly, Web Workers, OffscreenCanvas,
  and support for the input codec. H.264 MP4 is recommended. MOV does not guarantee
  HEVC support. Inputs are limited to 500 MB; shorter clips work best.
- MoveNet and FFmpeg run in workers. FFmpeg uses WORKERFS to avoid duplicating the
  source file in its WASM filesystem. Long recordings can still exceed memory.
- The app is not an offline PWA: loading/reloading needs its static assets.

```bash
npm run build
npm run preview
```

Build output is `dist/`; never host the repository root or personal recordings.
GitHub Pages is deployed by [the Pages workflow](.github/workflows/pages.yml).
Every push to `main` installs locked dependencies, prepares the model/runtime,
runs the JavaScript tests, builds the app, and deploys only `dist/` after success.
Pull requests targeting `main` run the same tests and build without deploying.
You can also rerun a deployment using **Actions → Test, build, and deploy Pages →
Run workflow** on `main`. Pages settings must use **GitHub Actions** as the source;
the old `gh-pages` branch is retained as a historical build and is no longer used.

Relative asset paths support subdirectory hosting when the URL ends with `/`.
Serve `.wasm` as `application/wasm`. No cross-origin isolation is required.
The document has a restrictive CSP. TensorFlow Lite's Emscripten worker uses
runtime-generated JS bindings; if your host applies CSP to worker responses,
account for that requirement separately from the document policy.

### Model attribution

Shot recognition uses the pretrained Tennis RNN (GRU) model from
[antoinekeller/tennis_shot_recognition — Tennis shot recognition with Dense NN and RNN](https://github.com/antoinekeller/tennis_shot_recognition).
The pinned checkpoint is
[`tennis_rnn.h5` at revision `6fdfd938f8d6893e2c926eb12fcc7f9bfc3e792f`](https://github.com/antoinekeller/tennis_shot_recognition/blob/6fdfd938f8d6893e2c926eb12fcc7f9bfc3e792f/tennis_rnn.h5).
This project exports its numeric weights for browser inference and uses MoveNet
Lightning for pose estimation. Credit for the pretrained shot classifier belongs
to the upstream project; this app does not train a new classifier.

The upstream Tennis RNN repository does not declare a license. Its weights are
fetched locally rather than committed here; a browser build contains derived
weights. Resolve the upstream redistribution terms before sharing such a build.

## Python CLI

No classifier selection or optional model dependencies are needed:

```bash
uv run python stroke_highlights.py tennis.mp4 --output-dir outputs/session
```

For a quick analysis without video export:

```bash
uv run python stroke_highlights.py tennis.mp4 \
  --output-dir outputs/trial --analyze-only --limit 30
```

Outputs:

- `strokes.csv`: proposed shot times, label, score, inclusion flag, and reason.
- `stroke-review.json`: loadable in the browser with the same source video.
- `tennis-rnn-scores.npz`: cached window probabilities for threshold experiments.
- `forehands.mp4` / `backhands.mp4`: included-category compilations, unless using
  `--analyze-only`. Empty categories do not create an MP4.

Useful settings:

| Option | Default | Purpose |
| --- | --- | --- |
| `--handedness` | `right` | `right` or `left` |
| `--mirrored` | off | Unflip mirrored input before pose inference |
| `--before`, `--after` | 1.2, 0.9 s | Excerpt padding |
| `--tennis-rnn-threshold` | 0.98 | Minimum class probability, not calibrated accuracy |
| `--tennis-rnn-min-gap` | 2.0 s | Cross-class duplicate suppression |
| `--min-player-height` | 0.2 | Minimum player height as a fraction of frame height |
| `--stroke` | `both` | Export forehand, backhand, or both |
| `--overwrite` | off | Replace existing results; reuse a compatible score cache |

The retained `--classifier tennis-rnn` and `--player near` spellings are optional;
no other values are supported. Inference runs on CPU. A complete valid 30-frame
pose window is required; short clips and tracking gaps may produce no proposals.
Window timestamps use their centers. Each confident class run contributes its
highest-scoring window, followed by duplicate suppression.

Changing threshold, minimum gap, or padding can reuse cached scores with
`--overwrite`. Changing source video, handedness, mirroring, player-height filter,
or analysis duration requires a new output directory. Original videos are never
overwritten.

Edit the CSV's categories, inclusion flags, or excerpt bounds, then export it
without running the model again:

```bash
uv run python stroke_highlights.py tennis.mp4 --output-dir outputs/reviewed \
  --events-csv outputs/session/strokes.csv
```

Use constant-frame-rate input for repeatable timing. For variable-frame-rate or
inconsistently decoded MOV files, normalize to a new source first:

```bash
ffmpeg -i input.mov -map 0:v:0 -map '0:a:0?' -vf fps=30 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac source-30fps.mp4
```

## Experiments and verification

### Human-reviewed regression set

```bash
uv run python regression_review.py propose
uv run python regression_review.py serve
```

Open **http://127.0.0.1:8780/**. The simple review queue contains 20 shuffled short
clips: six prior shot moments, two prior no-shot/partial-shot moments, eight random
sections, and four low-motion candidates. The original continuous context segment
is retained on disk but excluded from the single-label queue.

The existing Tennis RNN pipeline (MoveNet Lightning plus the pretrained GRU) runs
on each clip using its current Python CPU implementation and default thresholds.
`proposals.json` records its suggestions, model hashes, backend revision, settings,
and clip hashes. Cached inference scores are kept locally. Suggestions are never
human ground truth until explicitly approved or corrected.

Watch the near-court player and click **Approve & next**, or **Deny / choose another
label** and select **Forehand**, **Backhand**, **No shot**, or **Unsure**. Each decision
saves immediately to `annotations.json` and advances to an unfinished clip. You
can revisit clips to change a decision. No timestamps or contact markers are
required. Choose Unsure for mixed forehands/backhands, serves, or unclear footage;
these labels are retained in the review. The current regression policy combines
Unsure and No shot as negative. Insufficient tracked poses also
produce an Unsure suggestion, never an automatic No shot label.

Approval stays beside the video at desktop and embedded-browser widths. Keyboard
shortcuts: **A** approve, **D** show corrections, **1–4** choose forehand/backhand/
no shot/unsure while corrections are open, **P** play/pause, and **R** replay.
Shortcuts ignore held-key repeats, modifier keys, and editable controls.

Reloading resumes at the first unfinished clip. Existing human labels from the
older detailed reviewer are preserved when unambiguous; the original annotations
are backed up to `annotations-before-simple-review.json` before migration.
**Download review** provides a portable backup.

**Freeze reviewed dataset** requires a human decision for every review clip,
verifies video/source SHA-256 hashes, and saves a separate `frozen-r…json` snapshot.
The snapshot evaluates clip labels (agreement, precision/recall, confusion matrix).
**Detection times and contact times are not regression criteria.** Source intervals
are retained solely to reproduce the excerpts. Freezing does not merge an
optimization, and the draft remains editable.

The current policy and acceptance gates are defined in [docs/regression-evaluation-policy.json](docs/regression-evaluation-policy.json). Future optimization agents must follow [docs/optimization-regression.md](docs/optimization-regression.md): retain all positives and report zero false negatives for each stroke class. The dataset-local `evaluation-policy.json` pins this policy to the immutable snapshot.

Run a fresh browser baseline against a frozen snapshot (choose a new output file
for each run), open the printed loopback URL, and click **Run baseline**:

```bash
node scripts/evaluate-browser.mjs outputs/regression-review-v1/frozen-r26-a895bb7c8595.json experiments/browser-regression-run.json
node --test tests/regression-metrics.test.js
```

The runner verifies clip hashes and builds the current browser worker, then runs
each clip with a fresh tracker. Its JSON records code/model hashes, settings,
browser information, raw predictions, confusion matrix, precision/recall, and
individual mismatches. Forehand and Backhand are the only positive classes.
Human Unsure and No shot labels are both negative; no positive detection is also
negative, including insufficient tracked poses. Tracking failures remain diagnostic.
Mixed forehand/backhand detections count toward both positive classes and cannot
pass as negative. The scoring policy is versioned separately from frozen labels. Runtime is
descriptive, not a controlled speed benchmark. Review suggestions and labels are
never overwritten.

This is a starter corpus of existing local footage, not evidence of broad model
accuracy. Keep related recordings and overlapping clips in the same data split.
The original source videos and review excerpts remain local under ignored
`outputs/`. To reproduce sampling, use a new output directory:

```bash
uv run python regression_review.py build --directory outputs/regression-review-v2 --seed 20261115
uv run python regression_review.py propose --directory outputs/regression-review-v2
uv run python regression_review.py serve --directory outputs/regression-review-v2
```

[The decision log](experiments/DECISIONS.md) preserves the motion heuristic,
MediaPipe browser rules, and AdaSpot trials, their limitations, and why Tennis RNN
was selected. Those classifiers are no longer executable options. Small historical
prediction/settings snapshots are under `experiments/2026-11-15-classifier-selection/`.

`review_comparison.py` is an **archive viewer**, not an inference entry point.
It can still replay existing comparison outputs with the matching local source:

```bash
python3 review_comparison.py \
  experiments/2026-11-15-classifier-selection/adaspot-comparison-sensitive \
  --video outputs/adaspot-comparison-input.mp4
```

It serves only the chosen video and review assets on localhost. Historical review
calls/notes stay in browser storage and can be downloaded. The old local
`outputs/` experiments remain available as well.

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
npm run setup
npm test
npm run build
```

Tests cover default-only CLI routing, rejected retired classifiers, output
protection, cache identity, 30 Hz sampling, feature order/reflection, neutral and
serve handling, temporal suppression, Keras/browser GRU parity, review validation,
and actual FFmpeg audio/silent-input exports. These verify implementation, not
recognition accuracy. See the decision log for the remaining evaluation work.
