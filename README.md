# Harsh-Feature-Bench

A comprehensive benchmarking framework for evaluating feature detectors and descriptors on challenging aerial/glacier imagery. Compares traditional (SIFT) and learned (SuperPoint, ALIKED, DISK, LoFTR, RoMa, DKM) methods across feature matching quality, COLMAP 3D reconstruction, and viewpoint resilience.

## Setup

```bash
# Create conda environment from exported file
conda env create -f environment.yml
conda activate benchmark

# COLMAP 4.0+ must be available in PATH
# If using Docker-based COLMAP, run the export script:
bash ~/hamza-workdir/colmap/docker/export_colmap_docker.sh

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
```

## Usage

### Run COLMAP Reconstruction

```bash
# Single method
python run_colmap.py --method sift --images ./images --output ./MVS/sift

# Sparse only (skip dense MVS)
python run_colmap.py --method disk+lightglue --images ./images --skip-dense --output ./MVS/disk

# Multiple methods
python run_colmap.py --method sift,aliked,superpoint+superglue --images ./images --output ./MVS

# All methods
python run_colmap.py --method all --images ./images --output ./MVS
```

Each run produces:
- `database.db` — COLMAP database with keypoints, matches, two-view geometries
- `sparse/<N>/` — COLMAP sparse reconstruction
- `dense/fused.ply` — Dense point cloud (unless `--skip-dense`)
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
- **ALIKED** uses ALIKED-N16ROT (rotation-equivariant) consistently across native COLMAP, extractors, and matchers
- **RoMa Full** uses `upsample_res=560` to fit in 32GB VRAM; model reloads every 10 pairs to manage VRAM leaks
- **Dense matchers** (LoFTR, RoMa, DKM) use `KeypointAggregator` for COLMAP integration — per-image keypoint counts can exceed `MAX_MATCHES_PER_PAIR` due to aggregation across all pairs
- **DISK** requires a positive `max_num_keypoints` (kornia limitation); set to `MAX_MATCHES_PER_PAIR`
- COLMAP runs in Docker (via wrapper in `/usr/local/bin/colmap`); the wrapper auto-mounts paths from arguments
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically to reduce CUDA memory fragmentation

## License

MIT
