#!/usr/bin/env python3
"""Single entry point for COLMAP reconstruction with any feature method.

Usage:
    python run_colmap.py --method sift --images ./Section-25-PNGs-Win
    python run_colmap.py --method aliked --images ./Section-25-PNGs-Win
    python run_colmap.py --method superpoint+superglue --images ./images --output ./recon
    python run_colmap.py --method disk+lightglue --images ./images --skip-dense
    python run_colmap.py --method loftr,roma --images ./images
    python run_colmap.py --method all --images ./images --output ./results
"""

import argparse
import os
import shutil
import torch
from pathlib import Path

from config import DEVICE
from extractors import get_extractor, AVAILABLE_METHODS
from extractors.sift_native import NativeColmapExtractor
from colmap_db import ColmapDatabase
from colmap_pipeline import (
    setup_output_dirs, discover_images,
    run_sparse_pipeline, run_dense_pipeline,
    run_colmap_mapper, run_colmap_mvs,
)


def run_single_method(method, image_dir, output_dir, skip_dense=False,
                      quality="high"):
    """Run full COLMAP pipeline for one method."""
    print(f"\n{'='*70}")
    print(f"  Running: {method}")
    print(f"{'='*70}")

    image_dir = str(Path(image_dir).resolve())
    device = DEVICE
    extractor = get_extractor(method, device)

    if isinstance(extractor, NativeColmapExtractor) and extractor.runs_full_pipeline:
        # SIFT: automatic_reconstructor handles everything
        output_dir = str(Path(output_dir).resolve())
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        extractor.run(image_dir, output_dir, quality=quality,
                      skip_dense=skip_dense)
        print(f"\nDone: {method}")
        print(f"Output: {output_dir}")
        return

    # All other methods: we handle DB + mapper + MVS
    output_dir, db_path, sparse_path, dense_path = setup_output_dirs(output_dir)
    image_paths = discover_images(image_dir)

    if isinstance(extractor, NativeColmapExtractor):
        # ALIKED: native extraction + matching, but we run mapper/MVS
        extractor.run(image_dir, db_path)
    elif extractor.is_dense:
        db = ColmapDatabase(db_path)
        run_dense_pipeline(db, extractor, image_paths, device)
        db.close()
    else:
        db = ColmapDatabase(db_path)
        run_sparse_pipeline(db, extractor, image_paths, device)
        db.close()

    # Sparse reconstruction
    sparse_model = run_colmap_mapper(db_path, image_dir, sparse_path)
    if sparse_model is None:
        print(f"Mapper failed for {method}. Skipping dense reconstruction.")
        return

    # Dense reconstruction
    if not skip_dense:
        run_colmap_mvs(image_dir, sparse_model, dense_path)

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
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense reconstruction (MVS) stages")
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
                              args.skip_dense, args.quality)
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
