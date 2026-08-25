# Results

Measured outputs of the runs reported in the paper. The COLMAP databases and
sparse models they were derived from are not distributed (70.4 GB); see
`../run_benchmark.sh` to regenerate them.

| File | Contents |
|---|---|
| `reconstruction_<dataset>.json` | Per method: registered images, triangulated points, mean reprojection error, mean track length |
| `timings/<dataset>/<method>.json` | Per-stage wall-clock time in **seconds** (`feature_extraction`, `matching`, `sparse_reconstruction`, `total`) |
| `keypoint_stats/<dataset>/<method>.json` | Keypoints retained per image after aggregation |
| `inliers_vs_angle.json` | Verified inlier counts and pair-verification rates binned by pairwise viewing-angle baseline |
| `pose_plausibility.json` | Cross-method pose-consistency audit (see `../audit_pose_plausibility.py`) |
| `merge_radius_ablation.json` | Dense-to-sparse clustering radius sweep |

Table I of the paper reports `total` from `timings/` converted to minutes.

`pose_plausibility.json` contains a `RoMa*fixed` entry from runs made after the
tag `paper-runs`. It is **not** part of the paper's results; see
`../PROVENANCE.md`.
