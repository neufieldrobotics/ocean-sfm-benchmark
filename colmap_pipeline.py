"""Shared COLMAP pipeline stages for all feature extractors.

Handles sparse extraction+matching, dense aggregation, and COLMAP execution
(mapper, undistorter, PatchMatch, fusion).
"""

import os
import subprocess
import shutil
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

from config import (
    COLMAP_BIN, MIN_INLIERS, EMPTY_CACHE_EVERY,
    MAPPER_FLAGS, IMAGE_EXTENSIONS,
)
from colmap_db import ColmapDatabase, verify_matches_cv2
from keypoint_aggregator import KeypointAggregator


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def setup_output_dirs(output_dir):
    """Create output directory structure. Returns (output_dir, db_path, sparse_path, dense_path)."""
    output_dir = str(Path(output_dir).resolve())
    if os.path.exists(output_dir):
        print(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    sparse_path = os.path.join(output_dir, "sparse")
    dense_path = os.path.join(output_dir, "dense")
    ensure_dir(sparse_path)
    ensure_dir(dense_path)
    db_path = os.path.join(output_dir, "database.db")
    return output_dir, db_path, sparse_path, dense_path


def discover_images(image_dir):
    """Find all supported image files, sorted."""
    image_dir = Path(image_dir)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Found {len(image_paths)} images in {image_dir}")
    return image_paths


def run_sparse_pipeline(db, extractor, image_paths, device):
    """Feature extraction + exhaustive matching for sparse extractors.

    Creates per-image cameras, extracts features, runs exhaustive matching
    with geometric verification, and stores everything in the COLMAP DB.
    """
    image_id_map = {}
    feat_cache = {}

    print(f"\n=== 1) Feature Extraction ({extractor.name}) ===")
    for img_path in tqdm(image_paths, desc="Extracting"):
        feat = extractor.extract_features_image(img_path)
        if feat is None:
            print(f"  Could not read {img_path.name}, skipping.")
            continue

        # Per-image camera
        h, w = feat["orig_hw"]
        cam_id = db.add_camera(w, h)
        image_id = db.add_image(img_path.name, cam_id)

        db.add_keypoints(image_id, feat["kps_orig"])
        db.add_descriptors(image_id, feat["desc"])

        image_id_map[img_path] = image_id
        feat_cache[img_path] = feat

    db.commit()

    image_list = list(image_id_map.keys())
    n_images = len(image_list)
    n_pairs = n_images * (n_images - 1) // 2
    print(f"Extracted features for {n_images} images. Matching {n_pairs} pairs.")

    print(f"\n=== 2) Exhaustive Matching ({extractor.name}) ===")
    match_counter = 0
    pair_counter = 0

    for i in tqdm(range(n_images), desc="Matching"):
        img0 = image_list[i]
        f0 = feat_cache[img0]
        kp0_orig = f0["kps_orig"]

        for j in range(i + 1, n_images):
            img1 = image_list[j]
            f1 = feat_cache[img1]
            kp1_orig = f1["kps_orig"]

            matches_arr, _conf = extractor.match_pair(f0, f1)
            pair_counter += 1

            if matches_arr is None or len(matches_arr) < MIN_INLIERS:
                continue

            inlier_matches, F = verify_matches_cv2(kp0_orig, kp1_orig, matches_arr)
            if inlier_matches is None:
                continue

            id1, id2 = image_id_map[img0], image_id_map[img1]
            # matches table: ALL raw putative matches (before RANSAC)
            # two_view_geometries: only RANSAC-verified inliers
            # COLMAP mapper uses raw match counts for pair importance & track building
            db.add_matches(id1, id2, matches_arr)
            db.add_two_view_geometry(id1, id2, inlier_matches, F)
            match_counter += 1

            if match_counter % 100 == 0:
                db.commit()

            if device == "cuda" and pair_counter % EMPTY_CACHE_EVERY == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    db.commit()
    print(f"Verified pairs: {match_counter} / {n_pairs}")

    # Free GPU memory
    del feat_cache
    if device == "cuda":
        torch.cuda.empty_cache()


def run_dense_pipeline(db, extractor, image_paths, device):
    """Dense matching pipeline: match all pairs, aggregate keypoints, write to DB.

    Uses KeypointAggregator to cluster per-pair correspondences into
    canonical per-image keypoints.
    """
    aggregator = KeypointAggregator()

    n_images = len(image_paths)
    total_pairs = n_images * (n_images - 1) // 2

    # Phase 1: Extract image dims
    feat_cache = {}
    for img_path in image_paths:
        feat = extractor.extract_features_image(img_path)
        if feat is not None:
            feat_cache[img_path] = feat

    # Phase 2: Match all pairs
    print(f"\n=== 1) Dense Matching ({extractor.name}, {total_pairs} pairs) ===")
    pair_count = 0
    for i in tqdm(range(n_images), desc="Matching"):
        if image_paths[i] not in feat_cache:
            continue
        for j in range(i + 1, n_images):
            if image_paths[j] not in feat_cache:
                continue

            result = extractor.match_pair(feat_cache[image_paths[i]],
                                          feat_cache[image_paths[j]])
            pair_count += 1

            if result[0] is not None and len(result[0]) >= MIN_INLIERS:
                pts0, pts1, confs = result
                aggregator.add_pair(
                    image_paths[i].name, image_paths[j].name,
                    pts0, pts1, confs,
                )

            if device == "cuda" and pair_count % EMPTY_CACHE_EVERY == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Phase 3: Aggregate keypoints
    print("\n=== 2) Keypoint Aggregation ===")
    keypoints_per_image, kd_trees = aggregator.aggregate()

    for img_name, kps in keypoints_per_image.items():
        print(f"  {img_name}: {len(kps)} canonical keypoints")

    # Phase 4: Re-index matches
    print("\n=== 3) Re-indexing Matches ===")
    reindexed_pairs = aggregator.reindex_matches(kd_trees)

    # Phase 5: Write to DB
    print("\n=== 4) Writing COLMAP Database ===")
    image_id_map = {}
    for img_path in image_paths:
        if img_path not in feat_cache:
            continue
        h, w = feat_cache[img_path]["orig_hw"]
        cam_id = db.add_camera(w, h)
        image_id = db.add_image(img_path.name, cam_id)
        image_id_map[img_path.name] = image_id

        kps = keypoints_per_image.get(img_path.name,
                                      np.zeros((0, 2), dtype=np.float32))
        db.add_keypoints(image_id, kps)

        # Dummy descriptors (COLMAP requires them but mapper uses matches directly)
        n_kps = len(kps)
        dummy_desc = np.zeros((n_kps, 128), dtype=np.float32) if n_kps > 0 \
            else np.zeros((0, 128), dtype=np.float32)
        db.add_descriptors(image_id, dummy_desc)

    db.commit()

    # Phase 6: Geometric verification + write matches
    print("\n=== 5) Geometric Verification ===")
    verified_count = 0
    for img0_name, img1_name, matches, _ in tqdm(reindexed_pairs, desc="Verifying"):
        kps0 = keypoints_per_image.get(img0_name)
        kps1 = keypoints_per_image.get(img1_name)
        if kps0 is None or kps1 is None:
            continue

        inliers, F = verify_matches_cv2(kps0, kps1, matches)
        if inliers is None:
            continue

        id0 = image_id_map[img0_name]
        id1 = image_id_map[img1_name]
        # Raw matches in matches table, verified inliers in two_view_geometries
        db.add_matches(id0, id1, matches)
        db.add_two_view_geometry(id0, id1, inliers, F)
        verified_count += 1

    db.commit()
    print(f"Verified pairs: {verified_count} / {len(reindexed_pairs)}")

    # Free GPU memory
    del feat_cache
    if device == "cuda":
        torch.cuda.empty_cache()


def run_colmap_mapper(db_path, image_dir, sparse_path, mapper_flags=None):
    """Run COLMAP mapper for sparse reconstruction."""
    flags = dict(MAPPER_FLAGS)
    if mapper_flags:
        flags.update(mapper_flags)

    cmd = [
        COLMAP_BIN, "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_path,
    ]
    for key, val in flags.items():
        cmd.extend([f"--Mapper.{key}", str(val)])

    print("\n=== Sparse Reconstruction (mapper) ===")
    subprocess.run(cmd, check=True)

    # Find the sparse model directory
    sparse_model_dirs = sorted([
        d for d in Path(sparse_path).iterdir() if d.is_dir()
    ])
    if not sparse_model_dirs:
        print("ERROR: Mapper produced no models.")
        return None
    sparse_model = str(sparse_model_dirs[0])
    print(f"Using sparse model: {sparse_model}")

    # Print stats
    try:
        result = subprocess.run(
            [COLMAP_BIN, "model_analyzer", "--path", sparse_model],
            capture_output=True, text=True,
        )
        print(result.stdout)
    except Exception:
        pass

    return sparse_model


def run_colmap_mvs(image_dir, sparse_model, dense_path):
    """Run dense reconstruction: undistortion + PatchMatch + fusion."""
    print("\n=== Image Undistortion ===")
    subprocess.run([
        COLMAP_BIN, "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model,
        "--output_path", dense_path,
        "--output_type", "COLMAP",
    ], check=True)

    print("\n=== Dense Reconstruction (PatchMatch) ===")
    subprocess.run([
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], check=True)

    subprocess.run([
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(dense_path, "fused.ply"),
    ], check=True)

    print(f"Dense reconstruction complete: {os.path.join(dense_path, 'fused.ply')}")
