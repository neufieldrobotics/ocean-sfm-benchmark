#!/usr/bin/env python3
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""Single entry point for COLMAP reconstruction with any feature method.

Usage:
    python run_colmap.py --method sift --images ./images --output ./MVS/sift
    python run_colmap.py --method aliked --images ./images --output ./MVS/aliked
    python run_colmap.py --method superpoint+superglue --images ./images --output ./MVS/sp-sg
    python run_colmap.py --method loftr,roma --images ./images --output ./MVS
    python run_colmap.py --method all --images ./images --output ./MVS
    python run_colmap.py --method sift --images ./images --output ./MVS/sift --dense  # include MVS
"""

import argparse
import os
import shutil
import torch
from pathlib import Path

from config import DEVICE
from extractors import get_extractor, AVAILABLE_METHODS
from colmap_db import ColmapDatabase
from colmap_pipeline import (
    setup_output_dirs, discover_images, save_timings, save_keypoint_stats,
    run_sparse_pipeline, run_dense_pipeline,
    run_colmap_mapper, run_colmap_mvs,
)


def run_single_method(method, image_dir, output_dir, run_dense=False,
                      quality="high", merge_radius=None, keep_h5=None):
    """Run full COLMAP pipeline for one method."""
    print(f"\n{'='*70}")
    print(f"  Running: {method}")
    print(f"{'='*70}")

    image_dir = str(Path(image_dir).resolve())
    device = DEVICE
    extractor = get_extractor(method, device)

    # All methods: separate commands with per-stage timing
    output_dir, db_path, sparse_path, dense_path = setup_output_dirs(output_dir)
    image_paths = discover_images(image_dir)
    timings = {}

    keypoint_counts = {}

    if getattr(extractor, "is_native", False):
        # Native-style: extractor handles extraction+matching, stores timings
        extractor.run(image_dir, db_path)
        timings.update(extractor.timings)
        # Read keypoint counts from database
        db = ColmapDatabase(db_path)
        for row in db.connection.execute(
                "SELECT i.name, k.rows FROM images i "
                "JOIN keypoints k ON i.image_id = k.image_id"):
            keypoint_counts[row[0]] = row[1]
        db.close()
    elif extractor.is_dense:
        db = ColmapDatabase(db_path)
        stage_timings, keypoint_counts = run_dense_pipeline(
            db, extractor, image_paths, device,
            merge_radius=merge_radius, h5_path=keep_h5)
        timings.update(stage_timings)
        db.close()
    else:
        db = ColmapDatabase(db_path)
        stage_timings, keypoint_counts = run_sparse_pipeline(db, extractor, image_paths, device)
        timings.update(stage_timings)
        db.close()

    # Sparse reconstruction
    sparse_model, mapper_time = run_colmap_mapper(db_path, image_dir, sparse_path)
    timings["sparse_reconstruction"] = mapper_time
    if sparse_model is None:
        print(f"Mapper failed for {method}. Skipping dense reconstruction.")
        save_timings(output_dir, timings)
        return

    # Dense reconstruction
    if run_dense:
        mvs_timings = run_colmap_mvs(image_dir, sparse_model, dense_path)
        timings.update(mvs_timings)

    timings["total"] = sum(timings.values())
    save_timings(output_dir, timings)
    save_keypoint_stats(output_dir, keypoint_counts)

    print(f"\nDone: {method}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP reconstruction with custom feature methods")
    parser.add_argument("--method", "-m", required=True,
                        help=f"Method name(s), comma-separated, or 'all'. "
                             f"Available: {', '.join(AVAILABLE_METHODS)}")
    parser.add_argument("--images", "-i", required=True,
                        help="Path to image directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: ./results/<method>_reconstruction)")
    parser.add_argument("--dense", action="store_true",
                        help="Run dense reconstruction (MVS) stages (off by default)")
    parser.add_argument("--merge-radius", type=float, default=None,
                        help="Keypoint clustering radius in px for dense "
                             "matchers (default: config KEYPOINT_MERGE_RADIUS)")
    parser.add_argument("--keep-h5", default=None,
                        help="Persist raw dense correspondences to this HDF5 "
                             "path so they can be re-aggregated at other radii")
    parser.add_argument("--quality", default="high",
                        choices=["low", "medium", "high", "extreme"],
                        help="Quality level for native COLMAP methods (default: high)")
    args = parser.parse_args()

    # Parse methods
    if args.method.lower() == "all":
        methods = AVAILABLE_METHODS
    else:
        methods = [m.strip() for m in args.method.split(",")]
        for m in methods:
            if m not in AVAILABLE_METHODS:
                parser.error(f"Unknown method: {m}. Available: {AVAILABLE_METHODS}")

    for method in methods:
        if args.output and len(methods) == 1:
            output_dir = args.output
        else:
            base = args.output or "./results"
            output_dir = f"{base}/{method}_reconstruction"

        try:
            run_single_method(method, args.images, output_dir,
                              args.dense, args.quality,
                              merge_radius=args.merge_radius,
                              keep_h5=args.keep_h5)
        except Exception as e:
            print(f"\nERROR running {method}: {e}")
            import traceback
            traceback.print_exc()
            if len(methods) > 1:
                print("Continuing with next method...")
                continue
            raise


if __name__ == "__main__":
    main()
