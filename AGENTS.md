# Optimization regression instructions

Before changing video decoding, frame selection, preprocessing, tracking, models or performance, read [docs/optimization-regression.md](docs/optimization-regression.md) and follow its baseline/candidate validation procedure.

The active evaluation policy is `forehand-backhand-positives-v1`: `unsure` and `no_shot` are negatives; preserve every forehand/backhand positive and require zero false negatives for each positive class. Keep tracking failures diagnostic. Do not use old frozen metadata to exclude unsure clips, and do not collapse mixed positive detections to negative. Use the shared scorer and report false positives as well.

Preserve immutable frozen datasets and raw human labels. Equivalent optimizations must also preserve full inputs and scores; clip accuracy alone is insufficient. Keep optimization work isolated. Do not merge or deploy without explicit user authorization.
