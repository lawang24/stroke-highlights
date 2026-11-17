# Regression contract for optimization agents

The current user priority is **retain every positive stroke, with zero false negatives**. Evaluate clips with `forehand-backhand-positives-v1`, defined in [regression-evaluation-policy.json](regression-evaluation-policy.json) and implemented by `scripts/regression-metrics.mjs`. Do not infer the active policy solely from older frozen snapshots: their embedded `unsure` exclusion rule is superseded by this versioned policy.

## Dataset and labels

Use `outputs/regression-review-v1/frozen-r26-a895bb7c8595.json` (annotation revision 26). Verify its full SHA-256 against `outputs/regression-review-v1/evaluation-policy.json`, then verify the manifest, all scored review clips and original source videos against the recorded hashes. Preserve the frozen bytes, annotations, raw labels and source intervals. Update scoring policy separately from human ground truth.

The corpus has 20 human-reviewed short clips: 5 forehands, 2 backhands and 13 no-shot clips, from two source files. The continuous context clip is not a single-label example. This is a starter regression corpus, not a general accuracy benchmark. Keep overlapping clips and related source recordings in the same split; the independence of these recordings is unverified.

A reviewer process started before this policy update must be restarted before creating a new snapshot so it loads the updated freeze metadata. Existing snapshots use the dataset-local policy sidecar immediately and remain immutable.

## Positive-only scoring

- Forehand and backhand are positives. Human `unsure` and `no_shot` both map to the negative class for this policy. Include human unsure rows even when an older snapshot marked them excluded under the old policy.
- Use emitted stroke events to determine predictions. No forehand/backhand event means negative, including a completed inference run with zero classified windows. Preserve the zero-window/tracking-failure diagnostic.
- A positive clip with zero windows or no positive event is a false negative. Never exclude it from the denominator, relabel it or count a missing/failed inference run as a negative result.
- Keep mixed forehand/backhand detections as both positives. They cannot be collapsed to unsure/negative to hide false positives. A wrong stroke class is a false negative for the expected class and a false positive for the detected class.
- Primary acceptance: **forehand false negatives = 0 and backhand false negatives = 0**. Report positive support, recall and missed clip IDs. Also report any-stroke recall, precision, false positives, the confusion matrix and tracking failures; a high overall accuracy alone cannot pass this gate.
- Score clip labels only. Do **not** score detection timestamps or contact timestamps against the human annotations. Source intervals reproduce clips, not contact-time ground truth.

With this policy, the frozen browser baseline and corrected decoder experiment currently detect all 7/7 positive clips and classify all 13/13 negatives correctly: 20/20 clip agreement, zero false negatives and zero false positives. Six negatives have zero usable tracking windows; that diagnostic remains recorded. This statement applies to this small corpus only.

## Baseline and candidate procedure

1. Work in an isolated checkout/worktree; preserve existing user changes. Freeze the actual baseline source and record its hashes, including uncommitted files, dependency lock, preprocessing, worker and model weights. Git HEAD alone may not represent the active baseline.
2. Run fresh browser inference on every scored clip with identical settings and a fresh tracker per clip. The canonical baseline runner is:

   ```sh
   node scripts/evaluate-browser.mjs outputs/regression-review-v1/frozen-r26-a895bb7c8595.json experiments/browser-regression-NEW-RUN.json 8775
   node --test tests/regression-metrics.test.js
   ```

   Open the printed local URL in the embedded browser and run the baseline. Use a new output filename; never overwrite prior evidence. Model proposals, cached feature CSVs and labels are not fresh inference results.
3. Compare the candidate against that baseline. For an equivalent decode/preprocessing optimization, require exact full-input tensor SHA-256, sampled times, poses, used/output crops, tracking validity, complete score vectors and event arrays. Equal final labels or positive recall do not establish input/score parity. Do not weaken these checks or change thresholds, preprocessing, model weights, the 30 Hz sampling grid or tracking behavior merely to make a speed experiment pass.
4. Check the **complete original sources** as well as the re-encoded review MP4s. Preserve continuous crop/sequence state across clip boundaries. Record whether the fast decoder or baseline fallback handled each file; all-fallback results cannot prove the optimized path. Keep fallback restart state fresh and propagate cancellation.
5. Run the positive-only metrics through the shared scoring module. Require exactly one successful run per scored clip. Preserve raw events, scores, hashes, settings, browser identity and per-clip diagnostics, then verify zero false negatives for both positive classes. Report false positives too.
6. Measure speed separately after correctness passes: same machine/browser, warmups, alternating baseline/candidate order, same settings and inputs, and no competing inference runs. Correctness runs alongside an interactive preview are not controlled timing benchmarks. Report scope and limitations.

The isolated sequential-decode experiment is at `/Users/lawrence/.codex/worktrees/tennis-sequential-decode/stroke-highlights/experiments/sequential-decode/`. Its `frozen-r26/check.mjs` recomputes exact comparisons from raw traces and rejects stale code/model/media identities; `frozen-r26/REPORT.md` and `report.json` contain results. Its `trial/README.md` documents the separate local preview. Inspect the latest status rather than assuming an in-progress full-source run has passed.

Do not merge, deploy or promote an optimization without explicit user authorization. Passing this regression suite is evidence, not authorization. Broader browser, codec, VFR, rotation, cancellation/native failure and resource coverage must be reported honestly rather than inferred from the starter corpus.

## Default reader promotion (2026-11-16)

At the user's request, the compatible sequential reader is now the application default. The preview's original-mode selector is removed; the original seeking reader remains only as an automatic compatibility fallback. [Promotion validation](../experiments/2026-11-16-fast-default-report.json) records fresh paired inference on all 20 scored clips and both continuous original sources: 12,844 exact input tensors, 6,801 score windows and 35 events per version. All 7 positive clips are retained, with zero false negatives for either positive class and zero false positives. The six zero-window negatives remain tracking diagnostics.

Raw baseline/candidate traces, frozen baseline source/model/lock bytes, source identities, the checker and lifecycle evidence remain under `experiments/sequential-decode/promotion-20261116/` in the isolated decoder worktree referenced above. This is correctness evidence on the starter corpus, not a new speed benchmark or broader browser/codec claim.
