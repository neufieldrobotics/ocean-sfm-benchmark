#!/usr/bin/env python3
"""
colmap_sparse_stats.py

Usage:
    python colmap_sparse_stats.py /path/to/sparse/0

This prints COLMAP sparse model statistics similar to:

Rigs: 39
Cameras: 39
Frames: 39
Registered frames: 39
Images: 39
Registered images: 39
Points: 36302
Observations: 142736
Mean track length: 3.931905
Mean observations per image: 3659.897436
Mean reprojection error: 1.056808px
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def try_pycolmap_stats(model_path: Path) -> bool:
    """
    Try to read the sparse model with pycolmap and print stats.
    Returns True on success, False if pycolmap is unavailable.
    """
    try:
        import pycolmap  # type: ignore
    except ImportError:
        return False

    try:
        reconstruction = pycolmap.Reconstruction(str(model_path))
    except Exception as e:
        print(f"Error loading model with pycolmap: {e}", file=sys.stderr)
        sys.exit(1)

    # Basic containers
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D

    num_cameras = len(cameras)
    num_images = len(images)
    num_registered_images = sum(1 for img in images.values() if img.has_pose)
    num_points = len(points3D)

    # COLMAP logs now distinguish rigs / frames.
    # Depending on pycolmap version, these may or may not exist.
    rigs = getattr(reconstruction, "rigs", {})
    frames = getattr(reconstruction, "frames", {})

    num_rigs = len(rigs) if rigs is not None else 0
    num_frames = len(frames) if frames is not None else 0

    # Registered frames:
    # if frame support exists, count frames that have a valid pose;
    # otherwise fall back to registered images.
    num_registered_frames = 0
    if frames:
        for frame in frames.values():
            # Different pycolmap versions may expose this differently
            if hasattr(frame, "has_pose") and frame.has_pose:
                num_registered_frames += 1
            elif hasattr(frame, "cam_from_world") and frame.cam_from_world is not None:
                num_registered_frames += 1
    else:
        num_registered_frames = num_registered_images

    # Observations / track length / reprojection error
    total_observations = 0
    reproj_error_sum = 0.0
    reproj_error_count = 0

    for p in points3D.values():
        track_len = len(p.track.elements)
        total_observations += track_len

        # error is usually stored per 3D point in COLMAP/pycolmap
        if hasattr(p, "error"):
            reproj_error_sum += float(p.error) * track_len
            reproj_error_count += track_len

    mean_track_length = (
        total_observations / num_points if num_points > 0 else 0.0
    )

    mean_observations_per_image = (
        total_observations / num_registered_images if num_registered_images > 0 else 0.0
    )

    mean_reprojection_error = (
        reproj_error_sum / reproj_error_count if reproj_error_count > 0 else 0.0
    )

    print(f"Rigs: {num_rigs}")
    print(f"Cameras: {num_cameras}")
    print(f"Frames: {num_frames if num_frames > 0 else num_images}")
    print(
        f"Registered frames: "
        f"{num_registered_frames if num_frames > 0 else num_registered_images}"
    )
    print(f"Images: {num_images}")
    print(f"Registered images: {num_registered_images}")
    print(f"Points: {num_points}")
    print(f"Observations: {total_observations}")
    print(f"Mean track length: {mean_track_length:.6f}")
    print(f"Mean observations per image: {mean_observations_per_image:.6f}")
    print(f"Mean reprojection error: {mean_reprojection_error:.6f}px")

    return True


def run_colmap_model_analyzer(model_path: Path) -> None:
    """
    Fallback: call the official COLMAP binary and let it print the stats.
    """
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        print(
            "Error: neither pycolmap nor COLMAP binary was found.\n"
            "Install one of these:\n"
            "  pip install pycolmap\n"
            "or make sure `colmap` is available in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [colmap_bin, "model_analyzer", "--path", str(model_path)]

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("Error running COLMAP model_analyzer.", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(e.returncode)

    # model_analyzer usually prints to stderr in COLMAP logs, but sometimes stdout too
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += result.stderr

    print(output.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print statistics for a COLMAP sparse model folder."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to COLMAP sparse model directory (e.g. sparse/0)",
    )
    args = parser.parse_args()

    model_path = args.model_path.resolve()

    if not model_path.exists():
        print(f"Error: path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not model_path.is_dir():
        print(f"Error: path is not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Typical sparse model files
    expected_files = [
        model_path / "cameras.bin",
        model_path / "images.bin",
        model_path / "points3D.bin",
        model_path / "cameras.txt",
        model_path / "images.txt",
        model_path / "points3D.txt",
    ]
    if not any(p.exists() for p in expected_files):
        print(
            f"Warning: {model_path} does not look like a standard COLMAP sparse model folder.",
            file=sys.stderr,
        )

    # Prefer pycolmap because it gives plain clean output
    if not try_pycolmap_stats(model_path):
        run_colmap_model_analyzer(model_path)


if __name__ == "__main__":
    main()