#!/usr/bin/env bash
# Rebuild every COLMAP database and sparse reconstruction reported in
#   "Feature Matching for Ocean Robotics: A SfM Benchmark Across Marine,
#    Polar, and Aerial Datasets" (IEEE OCEANS 2026)
#
# The databases themselves are not distributed: the three sequences produce
# 70.4 GB of database.db files, with a single file (hydrothermal DKM) reaching
# 24 GB. This script regenerates them from the source imagery.
#
# Usage:
#   ./run_benchmark.sh <image_dir> <output_dir> [method ...]
#
# Example:
#   ./run_benchmark.sh /data/MVS-HyrdoThermal /data/out
#   ./run_benchmark.sh /data/MVS /data/out sift aliked dkm
#
# Every method runs under the identical resolution, keypoint, match and
# geometric-verification budget defined in config.py. Do not override them
# per-method: the comparison depends on that parity.

set -euo pipefail

IMAGES=${1:?usage: run_benchmark.sh <image_dir> <output_dir> [method ...]}
OUTPUT=${2:?usage: run_benchmark.sh <image_dir> <output_dir> [method ...]}
shift 2

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(sift orb akaze superpoint+superglue aliked disk+lightglue loftr dkm roma)
fi

for m in "${METHODS[@]}"; do
  echo "=== $m ==="
  python run_colmap.py --method "$m" --images "$IMAGES" --output "$OUTPUT/$m"
done

echo
echo "Reconstructions written under $OUTPUT/<method>/sparse/"
echo "Per-stage timings written to  $OUTPUT/<method>/timings.json"
