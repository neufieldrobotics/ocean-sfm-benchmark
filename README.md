# ocean-sfm-benchmark

Code, configurations and measured results for **"Feature Matching for Ocean
Robotics: A SfM Benchmark Across Marine, Polar, and Aerial Datasets"**
(IEEE OCEANS 2026), Hamza Naeem, Dennis Giaya and Hanumant Singh,
Northeastern University Field Robotics.

Nine feature matching pipelines --- SIFT, ORB, AKAZE, SuperPoint+SuperGlue,
ALIKED, DISK+LightGlue, LoFTR, DKM and RoMa (`tiny_roma_v1_outdoor`) --- are
evaluated inside a single controlled COLMAP SfM framework on three sequences: a
UAV pass over a marine-terminating glacier in Svalbard, a deep-sea photographic
survey of the Bio9 hydrothermal vent on the East Pacific Rise, and a UAV orbit
around a heritage building as an in-distribution control. Every pipeline runs
under identical resolution, keypoint and match budgets, and geometric
verification settings, so differences reflect the matching front end rather
than per-method tuning.

> **Reproducing the paper: check out the tag `paper-runs`.** The tip of `main`
> contains a later change to RoMa's correspondence selection that postdates
> every reported run. See [PROVENANCE.md](PROVENANCE.md).

## What is and is not in this repository

Measured results are in [`results/`](results/): registered images, triangulated
point counts, reprojection error, per-stage timings, keypoint statistics,
inliers as a function of viewing-angle baseline, the merge-radius ablation, and
the cross-method pose-consistency audit.

The COLMAP databases are **not** distributed. The three sequences produce
70.4 GB of `database.db` files, and the hydrothermal DKM database alone is
24 GB — well past what GitHub can host. Regenerate them from the source
imagery:

```bash
./run_benchmark.sh /path/to/MVS-HyrdoThermal /path/to/output
```

Source imagery is not redistributed here either. The City Hall sequence is from
the Heritage3DMTL dataset (Shende et al., 2024).

### Expected runtimes

One NVIDIA GPU, per sequence. The dense matchers dominate; budget accordingly
before launching a full sweep.

| | Glacier (66 img) | Hydrothermal (108 img) | City Hall (65 img) |
|---|---|---|---|
| SIFT | 3.3 min | 3.2 min | 1.8 min |
| ALIKED | 10.3 min | 27.4 min | 8.6 min |
| DISK+LightGlue | 5.7 min | 7.6 min | 3.4 min |
| LoFTR | 61.7 min | 92.6 min | 27.2 min |
| DKM | 79.3 min | 174.4 min | 50.3 min |
| RoMa | 152.2 min | **548.8 min (9.1 h)** | 196.9 min |

## Setup

```bash
# Create conda environment from exported file
conda env create -f environment.yml
conda activate benchmark

# COLMAP 3.8+ must be available in PATH as `colmap`
# (a Docker-backed wrapper script on PATH works too; set COLMAP_BIN in
#  config.py if the binary is named differently)

# SuperGlue is auto-cloned on first use from:
# https://github.com/magicleap/SuperGluePretrainedNetwork
```

### Dependencies (already in environment.yml)

- PyTorch + torchvision (CUDA)
- LightGlue (`pip install lightglue`)
- Kornia (`pip install kornia`)
- RoMatch (`pip install romatch`)
- OpenCV, NumPy, Matplotlib, Pandas, Seaborn, SciPy, tqdm

## Configuration

All parameters are centralized in `config.py` — single source of truth for both COLMAP pipeline and benchmark matchers:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_IMAGE_DIM` | 1600 | Uniform image resize for all detectors |
| `MAX_MATCHES_PER_PAIR` | 8000 | Uniform match cap per pair (top by confidence) |
| `SP_KEYPOINT_THRESHOLD` | 0.005 | SuperPoint detection threshold |
| `SP_NMS_RADIUS` | 2 | SuperPoint NMS radius |
| `ALIKED_DETECTION_THRESHOLD` | 0.01 | ALIKED detection threshold |
| `LOFTR_CONFIDENCE_THRESHOLD` | 0.1 | LoFTR match confidence threshold |
| `ROMA_CONFIDENCE_THRESHOLD` | 0.1 | RoMa match confidence threshold |

All matchers (`matchers/`) and extractors (`extractors/`) pull from these values.

## Supported Methods

### COLMAP Reconstruction Pipeline (`run_colmap.py`)

| Method | Detector | Matcher | Implementation |
|--------|----------|---------|----------------|
| `sift` | Covariant SIFT | SIFT BF + guided matching | Native COLMAP |
| `aliked` | ALIKED-N16ROT | ALIKED LightGlue | Native COLMAP |
| `superpoint+superglue` | SuperPoint | SuperGlue | Custom DB injection |
| `superpoint+lightglue` | SuperPoint | LightGlue | Custom DB injection |
| `aliked+lightglue` | ALIKED | LightGlue | Custom DB injection |
| `disk+lightglue` | DISK | LightGlue | Custom DB injection |
| `loftr` | LoFTR (dense) | LoFTR (dense) | Custom + keypoint aggregation |
| `roma` | RoMa Tiny (dense) | RoMa Tiny (dense) | Custom + keypoint aggregation |
| `roma-full` | RoMa Full (dense) | RoMa Full (dense) | Custom + keypoint aggregation |
| `dkm` | DKM (dense) | DKM (dense) | Custom + keypoint aggregation |

### Feature Matching Benchmark (`run_benchmark.py`)

| Matcher | Type | Description |
|---------|------|-------------|
| `sift` | Sparse | OpenCV SIFT + BF ratio test |
| `superglue` | Sparse | SuperPoint + SuperGlue |
| `loftr` | Dense | LoFTR outdoor |
| `aliked` | Sparse | ALIKED-N16ROT + BF ratio test |
| `aliked+lg` | Sparse | ALIKED + LightGlue |
| `sp+lg` | Sparse | SuperPoint + LightGlue |
| `disk` | Sparse | DISK + BF ratio test |
| `disk+lg` | Sparse | DISK + LightGlue |
| `roma` | Dense | RoMa Tiny |
| `roma-full` | Dense | RoMa Full |
| `dkm` | Dense | DKM |

## Architecture

```
Harsh-Feature-Bench/
    config.py                    # Centralized parameters (thresholds, resolution, caps)
    colmap_db.py                 # COLMAP SQLite database interface
    colmap_pipeline.py           # Shared COLMAP pipeline stages with timing
    keypoint_aggregator.py       # Dense matcher keypoint clustering (LoFTR, RoMa, DKM)

    extractors/                  # Feature extractors for COLMAP reconstruction
        __init__.py              # Extractor registry
        base.py                  # BaseExtractor ABC
        sift_native.py           # Native COLMAP SIFT & ALIKED
        superpoint_superglue.py  # SuperPoint + SuperGlue
        lightglue_extractor.py   # Unified SP/ALIKED/DISK + LightGlue
        dense_extractor.py       # LoFTR, RoMa, RoMa-full, DKM (dense matchers)

    matchers/                    # Feature matchers for benchmark evaluation
        __init__.py              # Matcher registry
        base.py                  # BaseMatcher + MatchResult + BenchmarkSummary
        sift.py                  # SIFT + BF
        superglue.py             # SuperPoint + SuperGlue
        loftr.py                 # LoFTR
        aliked.py                # ALIKED-N16ROT + BF
        roma.py                  # RoMa Tiny + Full
        dkm.py                   # DKM
        lightglue_matcher.py     # Unified SP/ALIKED/DISK + LightGlue
        disk_bf.py               # DISK + BF

    run_colmap.py                # COLMAP reconstruction entry point
    run_benchmark.py             # Feature matching benchmark entry point
    compare_recons.py            # Reconstruction comparison + timing + keypoint plots
    plot_db_stats.py             # Plot raw match/keypoint stats from COLMAP databases
    environment.yml              # Conda environment (name: benchmark)

    # Paper figure generation (OCEANS 2026)
    render_pointclouds.py        # Side-by-side sparse reconstructions per method
    plot_inliers_vs_angle.py     # Verified inliers vs pairwise viewing-angle baseline
    visualize_matches.py         # Verified correspondences on a representative pair
    ablate_merge_radius.py       # Merge-radius ablation for dense-matcher aggregation

    paper/                       # OCEANS 2026 full paper (IEEEtran, modular sections)
        main.tex
        sections/*.tex
        figures/*.png
```

## Usage

### Run COLMAP Reconstruction

```bash
# Single method (sparse only by default)
python run_colmap.py --method sift --images ./images --output ./MVS/sift

# Multiple methods
python run_colmap.py --method sift,aliked,superpoint+superglue --images ./images --output ./MVS

# All methods
python run_colmap.py --method all --images ./images --output ./MVS

# Include dense reconstruction (MVS)
python run_colmap.py --method sift --images ./images --output ./MVS/sift --dense
```

Each run produces:
- `database.db` — COLMAP database with keypoints, matches, two-view geometries
- `sparse/<N>/` — COLMAP sparse reconstruction
- `dense/fused.ply` — Dense point cloud (only with `--dense`)
- `timings.json` — Per-stage timing breakdown
- `keypoint_stats.json` — Per-image keypoint counts

### Compare Reconstructions

```bash
# Auto-discover from base directory (picks latest sparse model per method)
python compare_recons.py --base_dir ./MVS

# Manual paths
python compare_recons.py --recon_dirs ./MVS/sift/sparse/0 ./MVS/aliked/sparse/0 \
                         --labels "SIFT" "ALIKED"
```

Outputs:
- `reconstruction_comparison.png` — 6-panel quality comparison
- `reconstruction_comparison.json` — Machine-readable metrics
- `viewing_angle_analysis.png` — Pairwise viewing angle and baseline distributions
- `timing_comparison.png` — Pipeline timing breakdown
- `keypoint_stats_comparison.png` — Keypoint detection comparison per method

### Generate Paper Figures

```bash
# Side-by-side sparse reconstructions, one panel per method
python render_pointclouds.py --base_dir /media/goku/data/hamza/MVS-HyrdoThermal

# Verified inliers vs pairwise viewing-angle baseline (all three datasets)
python plot_inliers_vs_angle.py

# Verified correspondences on an auto-selected representative image pair
python visualize_matches.py --dataset MVS-HyrdoThermal --target-angle 20
```

Figures are written to `paper/figures/` alongside a JSON of the underlying
numbers. `render_pointclouds.py` and `plot_inliers_vs_angle.py` read COLMAP
`.bin`/`.db` files directly and need no GPU; `visualize_matches.py` reads the
verified correspondences straight from each method's `database.db` so the figure
always matches the reported table.

### Merge-Radius Ablation (dense-matcher aggregation)

Dense matchers need their pixel correspondences aggregated into canonical
keypoints before COLMAP can form tracks. To check that the 3 px clustering
radius does not bias results, cache one matching pass and re-aggregate it at
several radii:

```bash
# 1. run the dense matcher once, keeping the raw correspondences
python run_colmap.py --method loftr --images ./1_uav_images \
    --output /tmp/abl_base --keep-h5 /tmp/abl_pairs.h5

# 2. sweep the merge radius against that cache (matching is NOT re-run)
python ablate_merge_radius.py --h5 /tmp/abl_pairs.h5 --images ./1_uav_images \
    --workdir /tmp/abl --radii 0 1 2 3 4 6 8

# quantisation displacement only (fast, no mapper)
python ablate_merge_radius.py --h5 /tmp/abl_pairs.h5 --images ./1_uav_images \
    --quant-only
```

Radius 0 is the no-clustering control. `run_colmap.py` also accepts
`--merge-radius R` to run the whole pipeline at a non-default radius.

### Plot Database Statistics

```bash
# Raw match/keypoint counts from COLMAP databases (before reconstruction)
python plot_db_stats.py ./MVS
python plot_db_stats.py ./MVS --output ./MVS/db_stats.png
```

Outputs:
- Avg keypoints per image, avg raw matches per pair, avg inliers per pair
- Total keypoints, total matches
- Keypoint distribution box plot
- Summary table + JSON

### Run Feature Matching Benchmark

The benchmark processes **one method at a time** to avoid GPU OOM, saving per-method results incrementally.

```bash
# Sequential pairs (fast — N-1 pairs)
python run_benchmark.py -i ./images -m sift,superglue,aliked+lg,disk+lg,roma --sequential --no_vis

# Log output to file
python run_benchmark.py -i ./images -m all --sequential --no_vis 2>&1 | tee benchmark_log.txt

# Run methods one at a time (results accumulate, won't overwrite)
python run_benchmark.py -i ./images -m sift --sequential --no_vis
python run_benchmark.py -i ./images -m aliked+lg --sequential --no_vis

# Regenerate combined plots from saved per-method CSVs
python run_benchmark.py -i ./images --combine

# Viewpoint resilience analysis (multiple gap sizes)
python run_benchmark.py -i ./images -m all --multi-gap --no_vis

# Raw feature detection count (uncapped, per detector)
python run_benchmark.py -i ./images --detect

# All pairs (slow — only practical for <20 images)
python run_benchmark.py -i ./images -m sift,superglue --all_pairs --no_vis
```

Outputs:
- `results_<method>.csv` — Per-method results (incremental, not overwritten)
- `detailed_results.csv` — Combined results
- `summary.csv` — Aggregated statistics
- `benchmark_plots.png` — 4-panel comparison
- `inliers_heatmap.png` — Pairs x methods heatmap
- `viewpoint_resilience.png` — Inliers/ratio vs gap size (with `--multi-gap`)
- `feature_detection_analysis.png` — Raw feature counts per detector (with `--detect`)
- `report.txt` — Text summary with rankings

## Fair Comparison Design

All detectors operate under identical conditions:

1. **Same resolution**: All images resized to `MAX_IMAGE_DIM=1600` px (longest edge)
2. **Same parameters**: Detection thresholds, NMS radii from `config.py` (shared by extractors and matchers)
3. **Uncapped detection**: Each detector extracts as many features as it naturally can
4. **Uniform match cap**: Top `MAX_MATCHES_PER_PAIR=8000` matches per pair by confidence score
5. **Same COLMAP mapper**: Zero flag overrides — COLMAP's exact defaults for all methods
6. **Same geometric verification**: F-matrix RANSAC with `max_error=4.0`, `confidence=0.999`, `min_inliers=15`

## Notes

- **SIFT** uses COLMAP's native Covariant SIFT with guided matching
- **ALIKED** uses ALIKED-N16ROT (rotation-equivariant) consistently across native COLMAP, extractors, and matchers. COLMAP's bundled `aliked-n16rot.onnx` has a **hardcoded TopK k=4096** in the ONNX graph that silently caps features regardless of `max_num_features`. We patch it with `patch_aliked_onnx.py` to raise the limit to 16384 (`aliked-n16rot-16k.onnx`)
- **RoMa Full** uses `upsample_res=560` to fit in 32GB VRAM; model reloads every 10 pairs to manage VRAM leaks
- **Dense matchers** (LoFTR, RoMa, DKM) use `KeypointAggregator` for COLMAP integration — per-image keypoint counts can exceed `MAX_MATCHES_PER_PAIR` due to aggregation across all pairs
- **DISK** requires a positive `max_num_keypoints` (kornia limitation); set to `MAX_MATCHES_PER_PAIR`
- COLMAP runs in Docker (via wrapper in `/usr/local/bin/colmap`); the wrapper auto-mounts paths from arguments
- The Docker wrapper allocates a TTY (`docker run -it`), which **fails under `nohup`, cron or any
  non-interactive shell** with `the input device is not a TTY`. `colmap/docker/export_colmap_docker.sh`
  has been updated to add `-it` only when a TTY is attached; re-run it with sudo to install the fix.
  Until then, wrap batch invocations in a pty: `script -qec "python run_colmap.py ..." /dev/null`
- Some conda envs load the system `libstdc++` via torch before `sqlite3`, causing
  `CXXABI_1.3.15 not found`. Work around it with
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically to reduce CUDA memory fragmentation

## Citing

```bibtex
@inproceedings{naeem2026oceansfm,
  title     = {Feature Matching for Ocean Robotics: A {SfM} Benchmark Across
               Marine, Polar, and Aerial Datasets},
  author    = {Naeem, Hamza and Giaya, Dennis and Singh, Hanumant},
  booktitle = {OCEANS},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

MIT for the code in this repository. The bundled ALIKED ONNX weights
(`aliked-n16rot*.onnx`) and any third-party matchers installed via
`environment.yml` remain under their own licences.
