#!/usr/bin/env python3
"""Compare COLMAP sparse reconstructions across multiple feature methods.

Usage:
    python compare_recons.py --recon_dirs ./recon_sg/sparse/0 ./recon_aliked/sparse/0
    python compare_recons.py --base_dir ./results
    python compare_recons.py --recon_dirs dir1 dir2 --labels "SP+SG" "ALIKED+LG"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import argparse

# Try pycolmap first, fall back to manual parsing
try:
    import pycolmap
    USE_PYCOLMAP = True
except ImportError:
    USE_PYCOLMAP = False
    import struct


# ============================================================================
# Manual binary parsers (fallback if pycolmap not installed)
# ============================================================================

def read_points3D_bin(path):
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            points[pid] = {"xyz": np.array(xyz), "error": error, "track_len": track_len}
    return points


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            points2D_ids = []
            for _ in range(num_points2D):
                xy = struct.unpack("<dd", f.read(16))
                p3d_id = struct.unpack("<q", f.read(8))[0]
                if p3d_id != -1:
                    points2D_ids.append(p3d_id)
            images[image_id] = {
                "name": name.decode("utf-8"),
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
                "num_points2D": num_points2D,
                "num_points3D": len(points2D_ids),
            }
    return images


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        model_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 8, 7: 12, 8: 4, 9: 5}
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_p = model_params.get(model, 4)
            params = struct.unpack(f"<{num_p}d", f.read(num_p * 8))
            cameras[cam_id] = {"model": model, "width": width,
                               "height": height, "params": params}
    return cameras


# ============================================================================
# Quaternion utilities
# ============================================================================

def qvec_to_rotmat(qvec):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def camera_center(qvec, tvec):
    """Compute camera center in world coordinates: C = -R^T * t."""
    R = qvec_to_rotmat(qvec)
    return -R.T @ tvec


def viewing_direction(qvec):
    """Get camera viewing direction (z-axis of camera in world frame)."""
    R = qvec_to_rotmat(qvec)
    return R.T @ np.array([0, 0, 1])


def angle_between_cameras(qvec1, tvec1, qvec2, tvec2):
    """Compute angle between two camera viewing directions (degrees)."""
    d1 = viewing_direction(qvec1)
    d2 = viewing_direction(qvec2)
    cos_angle = np.clip(np.dot(d1, d2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


# ============================================================================
# Metric extraction
# ============================================================================

def extract_metrics(recon_path):
    """Extract comparison metrics from a COLMAP sparse reconstruction."""
    recon_path = Path(recon_path)
    metrics = {}

    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(str(recon_path))
        points = recon.points3D
        images = recon.images

        metrics["num_points3D"] = len(points)
        metrics["num_registered_images"] = len(images)

        errors = [p.error for p in points.values()]
        track_lengths = [p.track.length() for p in points.values()]

        obs_per_image = []
        for img in images.values():
            n_obs = sum(1 for p in img.points2D if p.has_point3D())
            obs_per_image.append(n_obs)

        # Extract poses for angle analysis
        image_poses = {}
        for img_id, img in images.items():
            image_poses[img.name] = {
                "qvec": np.array([img.cam_from_world.rotation.quat[3],  # w
                                  *img.cam_from_world.rotation.quat[:3]]),  # x,y,z
                "tvec": np.array(img.cam_from_world.translation),
            }
    else:
        pts = read_points3D_bin(recon_path / "points3D.bin")
        imgs = read_images_bin(recon_path / "images.bin")

        metrics["num_points3D"] = len(pts)
        metrics["num_registered_images"] = len(imgs)

        errors = [p["error"] for p in pts.values()]
        track_lengths = [p["track_len"] for p in pts.values()]
        obs_per_image = [img["num_points3D"] for img in imgs.values()]

        image_poses = {}
        for img_id, img in imgs.items():
            image_poses[img["name"]] = {
                "qvec": img["qvec"],
                "tvec": img["tvec"],
            }

    errors = np.array(errors)
    track_lengths = np.array(track_lengths)
    obs_per_image = np.array(obs_per_image)

    metrics["mean_reproj_error"] = np.mean(errors) if len(errors) > 0 else 0
    metrics["median_reproj_error"] = np.median(errors) if len(errors) > 0 else 0
    metrics["mean_track_length"] = np.mean(track_lengths) if len(track_lengths) > 0 else 0
    metrics["mean_obs_per_image"] = np.mean(obs_per_image) if len(obs_per_image) > 0 else 0
    metrics["errors"] = errors
    metrics["track_lengths"] = track_lengths
    metrics["obs_per_image"] = obs_per_image
    metrics["image_poses"] = image_poses

    return metrics


# ============================================================================
# Plotting
# ============================================================================

def compare_reconstructions(recon_dirs, labels=None, output="reconstruction_comparison.png"):
    if labels is None:
        labels = [Path(d).name for d in recon_dirs]

    all_metrics = {}
    for label, rdir in zip(labels, recon_dirs):
        print(f"Reading: {label} ...")
        all_metrics[label] = extract_metrics(rdir)

    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sparse Reconstruction Comparison", fontsize=16, fontweight="bold")

    # 1. Number of 3D points
    ax = axes[0, 0]
    vals = [all_metrics[l]["num_points3D"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Number of 3D Points")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    # 2. Registered images
    ax = axes[0, 1]
    vals = [all_metrics[l]["num_registered_images"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Registered Images")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v}", ha="center", va="bottom", fontsize=9)

    # 3. Reprojection error distribution
    ax = axes[0, 2]
    data = [all_metrics[l]["errors"] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors[:n]):
        patch.set_facecolor(c)
    ax.set_title("Reprojection Error Distribution")
    ax.set_ylabel("Reprojection Error (px)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Mean reprojection error
    ax = axes[1, 0]
    vals = [all_metrics[l]["mean_reproj_error"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Mean Reprojection Error")
    ax.set_ylabel("Error (px)")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # 5. Track length distribution
    ax = axes[1, 1]
    data = [all_metrics[l]["track_lengths"] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors[:n]):
        patch.set_facecolor(c)
    ax.set_title("Track Length Distribution")
    ax.set_ylabel("Track Length")
    ax.tick_params(axis="x", rotation=30)

    # 6. Mean observations per image
    ax = axes[1, 2]
    vals = [all_metrics[l]["mean_obs_per_image"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Mean Observations per Image")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path = Path(output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'Points3D':>10} {'Images':>8} {'Mean Err':>10} "
          f"{'Med Err':>10} {'Track Len':>10} {'Obs/Img':>10}")
    print("=" * 90)
    for l in labels:
        m = all_metrics[l]
        print(f"{l:<25} {m['num_points3D']:>10,} {m['num_registered_images']:>8} "
              f"{m['mean_reproj_error']:>10.4f} {m['median_reproj_error']:>10.4f} "
              f"{m['mean_track_length']:>10.2f} {m['mean_obs_per_image']:>10.1f}")
    print("=" * 90)

    # Export JSON
    json_metrics = {}
    for l in labels:
        m = all_metrics[l]
        json_metrics[l] = {
            "num_points3D": m["num_points3D"],
            "num_registered_images": m["num_registered_images"],
            "mean_reproj_error": float(m["mean_reproj_error"]),
            "median_reproj_error": float(m["median_reproj_error"]),
            "mean_track_length": float(m["mean_track_length"]),
            "mean_obs_per_image": float(m["mean_obs_per_image"]),
        }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Saved: {json_path}")

    # Pose-based angle analysis
    _analyze_viewing_angles(all_metrics, labels, output_path.parent)


def _analyze_viewing_angles(all_metrics, labels, output_dir):
    """Analyze pairwise viewing angles from reconstructed camera poses."""
    output_dir = Path(output_dir)
    has_poses = any(len(all_metrics[l]["image_poses"]) >= 2 for l in labels)
    if not has_poses:
        print("\nSkipping angle analysis (not enough registered poses)")
        return

    print("\n--- Pose-based Viewing Angle Analysis ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Viewing Angle Analysis from Camera Poses", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))

    for idx, label in enumerate(labels):
        poses = all_metrics[label]["image_poses"]
        if len(poses) < 2:
            continue

        # Compute pairwise angles
        names = sorted(poses.keys())
        angles = []
        baselines = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1, p2 = poses[names[i]], poses[names[j]]
                ang = angle_between_cameras(
                    p1["qvec"], p1["tvec"], p2["qvec"], p2["tvec"])
                c1 = camera_center(p1["qvec"], p1["tvec"])
                c2 = camera_center(p2["qvec"], p2["tvec"])
                bl = np.linalg.norm(c1 - c2)
                angles.append(ang)
                baselines.append(bl)

        angles = np.array(angles)
        baselines = np.array(baselines)

        axes[0].hist(angles, bins=20, alpha=0.5, label=label, color=colors[idx])
        axes[1].hist(baselines, bins=20, alpha=0.5, label=label, color=colors[idx])

        print(f"  {label}: {len(poses)} poses, angle range: "
              f"{angles.min():.1f}-{angles.max():.1f} deg, "
              f"baseline range: {baselines.min():.2f}-{baselines.max():.2f}")

    axes[0].set_xlabel("Viewing Angle Difference (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pairwise Viewing Angle Distribution")
    axes[0].legend()

    axes[1].set_xlabel("Baseline Distance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pairwise Baseline Distribution")
    axes[1].legend()

    plt.tight_layout()
    angle_path = output_dir / "viewing_angle_analysis.png"
    plt.savefig(angle_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {angle_path}")


# ============================================================================
# Auto-discovery
# ============================================================================

def discover_reconstructions(base_dir):
    """Auto-discover reconstruction directories from run_colmap.py output."""
    base_dir = Path(base_dir)
    recon_dirs = []
    labels = []

    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        # Look for sparse/0/ pattern
        sparse_0 = method_dir / "sparse" / "0"
        if sparse_0.exists() and (sparse_0 / "points3D.bin").exists():
            recon_dirs.append(str(sparse_0))
            labels.append(method_dir.name.replace("_reconstruction", ""))
            continue
        # Also check direct method_dir/0/ pattern
        direct_0 = method_dir / "0"
        if direct_0.exists() and (direct_0 / "points3D.bin").exists():
            recon_dirs.append(str(direct_0))
            labels.append(method_dir.name)

    return recon_dirs, labels


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare COLMAP sparse reconstructions")
    parser.add_argument("--recon_dirs", nargs="+", default=None,
                        help="Paths to sparse reconstruction directories")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each reconstruction")
    parser.add_argument("--base_dir", default=None,
                        help="Base output directory from run_colmap.py "
                             "(auto-discovers reconstructions)")
    parser.add_argument("--output", default="reconstruction_comparison.png",
                        help="Output image filename")
    args = parser.parse_args()

    if args.base_dir:
        recon_dirs, labels = discover_reconstructions(args.base_dir)
        if args.recon_dirs:
            recon_dirs.extend(args.recon_dirs)
            if args.labels:
                labels.extend(args.labels)
            else:
                labels.extend([Path(d).name for d in args.recon_dirs])
    elif args.recon_dirs:
        recon_dirs = args.recon_dirs
        labels = args.labels
    else:
        parser.error("Provide --recon_dirs or --base_dir")

    # Filter to existing
    existing_dirs = []
    existing_labels = []
    for d, l in zip(recon_dirs, labels or [Path(d).name for d in recon_dirs]):
        p = Path(d)
        if p.exists():
            print(f"EXISTS: {d} ({l})")
            existing_dirs.append(d)
            existing_labels.append(l)
        else:
            print(f"MISSING: {d}")

    if not existing_dirs:
        print("No reconstruction directories found!")
    else:
        compare_reconstructions(existing_dirs, existing_labels, args.output)
