# Harsh-Feature-Bench

A comprehensive benchmarking framework for evaluating feature detectors and descriptors on challenging aerial/glacier imagery. Compares traditional (SIFT) and learned (SuperPoint, ALIKED, DISK, LoFTR, RoMa) methods across feature matching quality, COLMAP 3D reconstruction, and viewpoint resilience.

## Objective

Glacier and harsh-environment aerial imagery presents unique challenges for Structure-from-Motion (SfM): repetitive textures, large viewpoint changes between flight passes, and high-resolution drone images. This project benchmarks multiple feature detection and matching pipelines to determine:

1. **Which methods register the most images** in COLMAP sparse reconstruction
2. **Which methods produce the best 3D reconstructions** (reprojection error, track length, point cloud density)
3. **How matching performance degrades with viewpoint change** (viewpoint resilience)
4. **Speed vs quality trade-offs** across the full pipeline (extraction, matching, reconstruction, dense MVS)

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

### Feature Matching Benchmark (`run_benchmark.py`)

| Matcher | Type | Description |
|---------|------|-------------|
| `sift` | Sparse | OpenCV SIFT + BF ratio test |
| `superglue` | Sparse | SuperPoint + SuperGlue |
| `loftr` | Dense | LoFTR outdoor |
| `aliked` | Sparse | ALIKED + BF ratio test |
| `aliked+lg` | Sparse | ALIKED + LightGlue |
| `sp+lg` | Sparse | SuperPoint + LightGlue |
| `disk` | Sparse | DISK + BF ratio test |
| `disk+lg` | Sparse | DISK + LightGlue |
| `roma` | Dense | RoMa Tiny |
| `dkm` | Dense | DKM |

## Architecture

```
Harsh-Feature-Bench/
    config.py                    # Shared constants (thresholds, COLMAP defaults)
    colmap_db.py                 # COLMAP SQLite database interface
    colmap_pipeline.py           # Shared COLMAP pipeline stages with timing
    keypoint_aggregator.py       # Dense matcher keypoint clustering (LoFTR, RoMa)

    extractors/                  # Feature extractors for COLMAP reconstruction
        __init__.py              # Extractor registry
        base.py                  # BaseExtractor ABC
        sift_native.py           # Native COLMAP SIFT & ALIKED
        superpoint_superglue.py  # SuperPoint + SuperGlue
        lightglue_extractor.py   # Unified SP/ALIKED/DISK + LightGlue
        dense_extractor.py       # LoFTR, RoMa (dense matchers)

    matchers/                    # Feature matchers for benchmark evaluation
        __init__.py              # Matcher registry
        base.py                  # BaseMatcher + MatchResult + BenchmarkSummary
        sift.py                  # SIFT + BF
        superglue.py             # SuperPoint + SuperGlue
        loftr.py                 # LoFTR
        aliked.py                # ALIKED + BF
        roma.py                  # RoMa Tiny
        dkm.py                   # DKM
        lightglue_matcher.py     # Unified SP/ALIKED/DISK + LightGlue
        disk_bf.py               # DISK + BF

    run_colmap.py                # COLMAP reconstruction entry point
    run_benchmark.py             # Feature matching benchmark entry point
    compare_recons.py            # Reconstruction comparison + timing plots
```

## Approach

### COLMAP Integration

For **native methods** (SIFT, ALIKED), we use COLMAP's own `feature_extractor` and `exhaustive_matcher` commands with the same parameters as `colmap automatic_reconstructor`:

- **SIFT**: Covariant SIFT (`estimate_affine_shape=1`, `domain_size_pooling=1`) with guided matching
- **ALIKED**: ALIKED-N16ROT with 8192 max features and LightGlue matching

For **custom methods** (SuperPoint+SuperGlue, LightGlue variants), we:
1. Extract features per image using the custom detector
2. Match all pairs exhaustively using the custom matcher
3. Verify matches with F-matrix RANSAC (same thresholds as COLMAP: `max_error=4.0`, `confidence=0.999`, `min_inliers=15`, `min_inlier_ratio=0.25`)
4. Store **raw matches** in the `matches` table and **verified inliers** in `two_view_geometries` (matching COLMAP's convention)
5. Run `colmap mapper` with **zero flag overrides** (uses COLMAP's exact defaults for fair comparison)

For **dense matchers** (LoFTR, RoMa), which produce per-pair correspondences without consistent keypoint IDs:
1. Match all pairs, collecting pixel-level correspondences
2. Cluster nearby detections per image into canonical keypoints using greedy radius clustering (`KeypointAggregator`)
3. Re-index matches to canonical keypoint IDs
4. Store in COLMAP database with dummy descriptors

### COLMAP Parameter Parity

All methods use identical COLMAP mapper parameters (the exact defaults):

| Parameter | Value | Source |
|-----------|-------|--------|
| `init_min_num_inliers` | 100 | COLMAP default |
| `abs_pose_min_num_inliers` | 30 | COLMAP default |
| `min_num_matches` | 15 | COLMAP default |
| `multiple_models` | 1 | COLMAP default |
| `ba_refine_focal_length` | 1 | COLMAP default |
| `filter_max_reproj_error` | 4 | COLMAP default |

This ensures fair comparison: same mapper, same thresholds, only different features.

### Benchmark Metrics

**Feature Matching** (`run_benchmark.py`):
- Number of matches and inliers per pair
- Inlier ratio (geometric consistency)
- Processing time per pair
- Success rate (pairs with >= 10 inliers)
- Viewpoint resilience (inliers vs image separation gap)

**Reconstruction Quality** (`compare_recons.py`):
- Number of registered images
- Number of 3D points
- Mean/median reprojection error
- Mean track length
- Mean observations per image
- Pairwise viewing angle distribution
- Pairwise baseline distribution

**Pipeline Timing** (`compare_recons.py`):
- Per-stage breakdown: feature extraction, matching, sparse reconstruction, undistortion, PatchMatch, fusion
- Total pipeline time comparison
- Stacked and grouped bar charts

## Usage

### Prerequisites

```bash
# Conda environment (tested with Python 3.10)
conda activate afr

# Core dependencies
pip install torch torchvision opencv-python numpy matplotlib pandas seaborn tqdm scipy

# Feature matchers
pip install lightglue kornia romatch

# SuperGlue (auto-cloned on first use)
# https://github.com/magicleap/SuperGluePretrainedNetwork

# COLMAP (must be in PATH)
# Tested with COLMAP 3.13
```

### Run COLMAP Reconstruction

```bash
# Single method
python run_colmap.py --method sift --images ./Section-25-PNGs-Win --output ./MVS/SIFT
python run_colmap.py --method aliked --images ./Section-25-PNGs-Win --output ./MVS/ALIKED
python run_colmap.py --method superpoint+superglue --images ./Section-25-PNGs-Win --output ./MVS/SP-SG

# Sparse only (skip dense MVS)
python run_colmap.py --method disk+lightglue --images ./images --skip-dense --output ./MVS/DISK-LG

# Multiple methods
python run_colmap.py --method sift,aliked,loftr --images ./images --output ./MVS

# All methods
python run_colmap.py --method all --images ./images --output ./MVS
```

Each run produces:
- `sparse/0/` — COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
- `dense/fused.ply` — Dense point cloud (if not `--skip-dense`)
- `timings.json` — Per-stage timing breakdown

### Compare Reconstructions

```bash
# Auto-discover from base directory
python compare_recons.py --base_dir ./MVS

# Manual paths
python compare_recons.py --recon_dirs ./MVS/SIFT/sparse/0 ./MVS/ALIKED/sparse/0 \
                         --labels "SIFT" "ALIKED"
```

Outputs:
- `reconstruction_comparison.png` — 6-panel quality comparison (3D points, registered images, reprojection error, track length, observations)
- `reconstruction_comparison.json` — Machine-readable metrics
- `viewing_angle_analysis.png` — Pairwise viewing angle and baseline distributions
- `timing_comparison.png` — Pipeline timing breakdown (total, stacked, per-stage)

### Run Feature Matching Benchmark

```bash
# Sequential pairs, all matchers
python run_benchmark.py --image_dir ./Section-25-PNGs-Win --methods all --sequential

# Specific matchers
python run_benchmark.py -i ./images -m sift,aliked+lg,disk+lg,loftr,roma

# Viewpoint resilience analysis (multiple gap sizes)
python run_benchmark.py -i ./images --methods all --multi-gap

# All pairs (combinatorial)
python run_benchmark.py -i ./images --methods sift,superglue --all_pairs --no_vis
```

Outputs:
- `detailed_results.csv` — Per-pair results for all methods
- `summary.csv` — Aggregated statistics
- `benchmark_plots.png` — 4-panel comparison (inliers, ratio, trend, time)
- `inliers_heatmap.png` — Pairs x methods heatmap
- `viewpoint_resilience.png` — Inliers/ratio vs gap size (with `--multi-gap`)
- `report.txt` — Text summary with rankings
- `visualizations/` — Match visualization images per pair

## Dataset

The benchmark is designed for DJI drone imagery of glaciers. The test dataset (`Section-25-PNGs-Win/`) contains 39 high-resolution PNG images (5280x3956, ~100MB each) from 5 separate flight passes:

| Group | Images | DJI IDs |
|-------|--------|---------|
| 1 | 9 | 1479-1487 |
| 2 | 11 | 1963-1973 |
| 3 | 7 | 2840-2846 |
| 4 | 4 | 3005-3008 |
| 5 | 8 | 3509-3516 |

The 5 disconnected groups make cross-group matching challenging — a key test for feature detector resilience.

## Notes

- **SIFT** uses COLMAP's native Covariant SIFT with guided matching (best registration: 39/39 images)
- **ALIKED** uses COLMAP's native ALIKED-N16ROT with LightGlue (8192 max features)
- **SuperPoint+SuperGlue** uses custom DB injection; SP_MAX_DIM configurable in `config.py`
- **RoMa** uses the `tiny_roma_v1_outdoor` variant (full RoMa requires 24GB+ VRAM)
- **Dense matchers** (LoFTR, RoMa) use `KeypointAggregator` for COLMAP integration: greedy radius clustering to build consistent per-image keypoints from per-pair correspondences
- All custom methods store `config=3` (UNCALIBRATED / F-matrix only) in COLMAP's `two_view_geometries` table
- COLMAP mapper runs with zero flag overrides for fair comparison across all methods

## License

MIT
