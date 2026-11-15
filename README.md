# Stroke Highlights

Create local forehand and backhand video compilations from tennis footage.
This standalone tool proposes strokes from player poses and supports manual CSV review.

## Setup

Requirements: Python 3.10–3.12, uv, and FFmpeg (including ffprobe for export tests).
On macOS, install FFmpeg with `brew install ffmpeg`.

From this repository, install the project environment with uv:

```bash
uv sync
```

## Forehand and Backhand Compilations

`stroke_highlights.py` finds swing candidates for one player and exports separate
`forehands.mp4` and `backhands.mp4` compilations, with the source audio. It works
with short clips or longer recordings, processing sampled frames without loading
the whole video into memory. Install FFmpeg for video export (`brew install ffmpeg`
on macOS).

```bash
uv run python stroke_highlights.py "/path/to/tennis.mov" \
  --output-dir outputs/tennis_strokes \
  --player near --handedness right --stroke both --device mps
```

For **only forehands**, use `--stroke forehand`. For a short trial, add `--limit 30`;
use a different output directory when subsequently analyzing the full video.
Use `--device cpu` on machines without Apple GPU support.

This is an **experimental motion heuristic**, not a trained tennis action model.
It uses the [Ultralytics COCO pose keypoints](https://docs.ultralytics.com/tasks/pose/)
to detect fast wrist movement relative to the hips, then checks preparation side
and swing direction. Scores measure rule strength, **not prediction probability**.
No video is uploaded. The pose weights download on first use.

### Camera and player settings

- Use a fixed camera behind a baseline, with the player's body and racket arm
  visible throughout the stroke. Side-on or moving cameras are not supported.
  Timestamps use frame index / FPS; convert variable-frame-rate recordings to
  constant frame rate first for accurate excerpt timing.
- `--player near` selects the largest person with feet in the lower 45% of the
  image, then follows their box by proximity.
  `--player far` initially limits selection to the upper 65% of the image. These
  are geometric selection rules, not player identity recognition.
- On crowded courts, restrict the player's **foot position** with a normalized
  `--roi LEFT TOP RIGHT BOTTOM`, e.g. `--roi 0 0.4 0.7 1`. Preview your framing
  before picking these bounds. Players can be reacquired after a tracking gap;
  a region helps avoid switching to someone on an adjacent court.
- Detections smaller than 20% of the frame height are excluded for the near
  player (5% for the far player). Adjust `--min-player-height` for your framing.
  The default 1.8-second `--min-gap` reduces repeat detections from follow-through
  and recovery; reduce it for drills with more frequent strokes.
- Set `--handedness left` for a left-handed player; add `--mirrored` for a flipped
  recording. If the player changes ends, process each camera-side section with
  the appropriate settings.
- Defaults include 1.2 seconds before and 0.9 seconds after each swing. Change
  `--before` and `--after` to include more preparation or shorten the compilation.
  Overlapping excerpts in the same category are merged to avoid repeated footage.

### Review and re-export

The output directory contains:

- `strokes.csv`: proposed timestamps, clip boundaries, labels, rule scores, and
  an `include` column. Ambiguous/overhead movements use `unknown` and are excluded.
- `poses.json`: cached sampled player poses. Matching inputs and inference settings
  can reuse this cache when changing detection thresholds or excerpt lengths.
- `forehands.mp4` / `backhands.mp4`: created only for categories with included strokes.
  Re-exporting an empty category with `--overwrite` removes its previous compilation.

Edit the CSV to correct `stroke` (`forehand`, `backhand`, `unknown`, or `ignore`),
set `include` to `true`/`false`, change `start`/`end` in seconds, or add missed strokes.
Then export without running inference again:

```bash
uv run python stroke_highlights.py "/path/to/tennis.mov" \
  --events-csv outputs/tennis_strokes/strokes.csv \
  --output-dir outputs/tennis_strokes --stroke forehand --overwrite
```

Use `--analyze-only` to generate detections before exporting. `--overwrite` allows
replacing previous CSV/video outputs; running detection again replaces CSV edits.
Incompatible pose caches cause an error: choose a new output directory or remove
the cache explicitly. The source video is never overwritten.

Brief wrist occlusions are interpolated; long occlusions, off-screen strokes,
and uncertain body landmarks can cause missed shots. Practice swings, volleys,
backswing movements, and other arm motions can cause false detections. Serves
and smashes are not reliably classified. Inspect the proposals before treating
a compilation as complete. Reliable shot recall across camera angles will require
labeled tennis footage and a temporal action classifier, potentially using ball
contact as an additional signal. This tool does not use ball tracking.

### Verification

```bash
uv run python -m unittest discover -s tests -v
```

Tests cover handedness/view direction, occlusion boundaries, overhead exclusion,
translation rejection, player regions, CSV validation, excerpt merging, and real
FFmpeg exports with and without audio. These checks validate pipeline behavior;
they do not measure stroke recognition accuracy on labeled tennis footage.

## Repository layout

- `stroke_highlights.py`: command-line detection, CSV review, and video export.
- `pose_utils.py`: model loading and pose result conversion.
- `tests/test_stroke_highlights.py`: detection and real FFmpeg export checks.
- `pyproject.toml`, `uv.lock`: standalone environment and locked dependencies.

Videos, generated outputs, pose caches inside `outputs/`, and downloaded model
weights are ignored by Git. Supply your own video path; no source footage or
model weights are bundled. Run `uv run python stroke_highlights.py --help` for
all options.

## Origin

Extracted from the local `tennis-highlights` project, including its uncommitted
stroke-highlights implementation and tests. Only the pose helpers required by
this tool were carried over. Set analysis/review, ball tracking, court detection,
and 3D reconstruction remain in the original project. This repository starts
with a fresh history; the original project and its working files are preserved.
