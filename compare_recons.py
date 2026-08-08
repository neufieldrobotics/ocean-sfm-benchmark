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

# Fixed color map — consistent across runs regardless of which methods appear
METHOD_COLORS = {
    "sift":                 "#1f77b4",  # blue
    "aliked":               "#ff7f0e",  # orange
    "superpoint+superglue": "#2ca02c",  # green
    "superpoint+lightglue": "#d62728",  # red
    "aliked+lightglue":     "#9467bd",  # purple
    "disk+lightglue":       "#8c564b",  # brown
    "loftr":                "#e377c2",  # pink
    "roma-tiny":            "#7f7f7f",  # grey
    "roma-full":            "#bcbd22",  # olive
    "dkm":                  "#17becf",  # cyan
    "orb":                  "#aec7e8",  # light blue
    "akaze":                "#ffbb78",  # light orange
}
_FALLBACK_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))


def _normalize_label(label):
    """Normalize method labels for consistent display."""
    # "roma" without qualifier is roma-tiny
    if label.lower() == "roma":
        return "roma-tiny"
    return label


def _get_color(label):
    """Get deterministic color for a method label."""
    return METHOD_COLORS.get(label, METHOD_COLORS.get(
        label.lower(), _FALLBACK_COLORS[hash(label) % len(_FALLBACK_COLORS)]))

# Try pycolmap first, fall back to manual parsing
try:
    import pycolmap
    USE_PYCOLMAP = True
except ImportError:
    USE_PYCOLMAP = False
    import struct

import sqlite3


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
                "obs_set": frozenset(points2D_ids),
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
# Database helpers
# ============================================================================

_COLMAP_MAX_IMAGE_ID = 2147483647  # matches ColmapDatabase._pair_id()


def read_db_pair_stats(db_path):
    """Read per-pair raw match and inlier counts from a COLMAP database.

    Returns dict keyed by (name_a, name_b) where name_a < name_b:
        {"matches": int, "inliers": int}
    """
    conn = sqlite3.connect(str(db_path))
    id_to_name = {row[0]: row[1]
                  for row in conn.execute("SELECT image_id, name FROM images")}

    pair_stats = {}

    for pair_id, rows in conn.execute("SELECT pair_id, rows FROM matches"):
        id1 = pair_id // _COLMAP_MAX_IMAGE_ID
        id2 = pair_id % _COLMAP_MAX_IMAGE_ID
        n1, n2 = id_to_name.get(id1), id_to_name.get(id2)
        if n1 and n2:
            key = (min(n1, n2), max(n1, n2))
            pair_stats.setdefault(key, {"matches": 0, "inliers": 0})
            pair_stats[key]["matches"] = rows

    for pair_id, rows in conn.execute(
            "SELECT pair_id, rows FROM two_view_geometries"):
        id1 = pair_id // _COLMAP_MAX_IMAGE_ID
        id2 = pair_id % _COLMAP_MAX_IMAGE_ID
        n1, n2 = id_to_name.get(id1), id_to_name.get(id2)
        if n1 and n2:
            key = (min(n1, n2), max(n1, n2))
            pair_stats.setdefault(key, {"matches": 0, "inliers": 0})
            pair_stats[key]["inliers"] = rows

    conn.close()
    return pair_stats


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

        # Extract poses and observation sets for angle analysis
        # pycolmap 3.13: cam_from_world() is a method, rotation.quat is xyzw order
        image_poses = {}
        image_obs_sets = {}
        for img_id, img in images.items():
            cfw = img.cam_from_world()
            quat_xyzw = cfw.rotation.quat
            image_poses[img.name] = {
                "qvec": np.array([quat_xyzw[3], quat_xyzw[0],
                                  quat_xyzw[1], quat_xyzw[2]]),  # convert to wxyz
                "tvec": np.array(cfw.translation),
            }
            image_obs_sets[img.name] = frozenset(
                p.point3D_id for p in img.points2D if p.has_point3D()
            )
    else:
        pts = read_points3D_bin(recon_path / "points3D.bin")
        imgs = read_images_bin(recon_path / "images.bin")

        metrics["num_points3D"] = len(pts)
        metrics["num_registered_images"] = len(imgs)

        errors = [p["error"] for p in pts.values()]
        track_lengths = [p["track_len"] for p in pts.values()]
        obs_per_image = [img["num_points3D"] for img in imgs.values()]

        image_poses = {}
        image_obs_sets = {}
        for img_id, img in imgs.items():
            image_poses[img["name"]] = {
                "qvec": img["qvec"],
                "tvec": img["tvec"],
            }
            image_obs_sets[img["name"]] = img["obs_set"]

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
    metrics["image_obs_sets"] = image_obs_sets

    # Auto-discover database.db (typically 2 or 1 levels above recon_path)
    pair_db_stats = {}
    for candidate in [
        recon_path.parent.parent / "database.db",  # sparse/N/ -> method_dir/
        recon_path.parent / "database.db",          # N/ -> sparse/
    ]:
        if candidate.exists():
            pair_db_stats = read_db_pair_stats(candidate)
            print(f"  DB stats: {candidate.name} ({len(pair_db_stats)} pairs)")
            break
    metrics["pair_db_stats"] = pair_db_stats

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
    label_colors = [_get_color(l) for l in labels]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sparse Reconstruction Comparison", fontsize=16, fontweight="bold")

    # 1. Number of 3D points
    ax = axes[0, 0]
    vals = [all_metrics[l]["num_points3D"] for l in labels]
    bars = ax.bar(labels, vals, color=label_colors)
    ax.set_title("Number of 3D Points")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    # 2. Registered images
    ax = axes[0, 1]
    vals = [all_metrics[l]["num_registered_images"] for l in labels]
    bars = ax.bar(labels, vals, color=label_colors)
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
    for patch, c in zip(bp["boxes"], label_colors):
        patch.set_facecolor(c)
    ax.set_title("Reprojection Error Distribution")
    ax.set_ylabel("Reprojection Error (px)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Mean reprojection error
    ax = axes[1, 0]
    vals = [all_metrics[l]["mean_reproj_error"] for l in labels]
    bars = ax.bar(labels, vals, color=label_colors)
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
    for patch, c in zip(bp["boxes"], label_colors):
        patch.set_facecolor(c)
    ax.set_title("Track Length Distribution")
    ax.set_ylabel("Track Length")
    ax.tick_params(axis="x", rotation=30)

    # 6. Mean observations per image
    ax = axes[1, 2]
    vals = [all_metrics[l]["mean_obs_per_image"] for l in labels]
    bars = ax.bar(labels, vals, color=label_colors)
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

    # Pose-based angle analysis — derive suffix from output filename
    stem = output_path.stem  # e.g. "reconstruction_comparison_MVS_KITTI"
    base_stem = "reconstruction_comparison"
    suffix = stem[len(base_stem):] if stem.startswith(base_stem) else ""

    _analyze_viewing_angles(all_metrics, labels, output_path.parent, suffix)
    _plot_viewing_angle_vs_inliers(all_metrics, labels, output_path.parent, suffix)


def _analyze_viewing_angles(all_metrics, labels, output_dir, suffix=""):
    """Analyze pairwise viewing angles from reconstructed camera poses."""
    output_dir = Path(output_dir)
    has_poses = any(len(all_metrics[l]["image_poses"]) >= 2 for l in labels)
    if not has_poses:
        print("\nSkipping angle analysis (not enough registered poses)")
        return

    print("\n--- Pose-based Viewing Angle Analysis ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Viewing Angle Analysis from Camera Poses", fontsize=14, fontweight="bold")

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

        # Use KDE-style line plot: sorted CDF / density curve
        angle_bins = np.linspace(0, max(angles.max(), 1), 30)
        baseline_bins = np.linspace(0, max(baselines.max(), 1e-6), 30)

        angle_counts, angle_edges = np.histogram(angles, bins=angle_bins)
        baseline_counts, baseline_edges = np.histogram(baselines, bins=baseline_bins)

        angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
        baseline_centers = (baseline_edges[:-1] + baseline_edges[1:]) / 2

        c = _get_color(label)
        axes[0].plot(angle_centers, angle_counts, marker='o', markersize=3,
                     linewidth=2, label=label, color=c)
        axes[1].plot(baseline_centers, baseline_counts, marker='o', markersize=3,
                     linewidth=2, label=label, color=c)

        print(f"  {label}: {len(poses)} poses, angle range: "
              f"{angles.min():.1f}-{angles.max():.1f} deg, "
              f"baseline range: {baselines.min():.2f}-{baselines.max():.2f}")

    axes[0].set_xlabel("Viewing Angle Difference (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pairwise Viewing Angle Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Baseline Distance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pairwise Baseline Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    angle_path = output_dir / f"viewing_angle_analysis{suffix}.png"
    plt.savefig(angle_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {angle_path}")


def _plot_viewing_angle_vs_inliers(all_metrics, labels, output_dir, suffix=""):
    """Plot true inliers (from DB) and co-visibility (from reconstruction) vs viewing angle.

    4-panel figure:
      Top-left:  True inlier count (two_view_geometries.rows) vs angle
      Top-right: True inlier ratio (inliers / raw_matches) vs angle
      Bot-left:  Co-visible 3D points (|obs_i ∩ obs_j|) vs angle
      Bot-right: Co-visibility ratio (shared / min(|obs_i|, |obs_j|)) vs angle
    """
    output_dir = Path(output_dir)

    has_poses = any(len(all_metrics[l]["image_poses"]) >= 2 for l in labels)
    if not has_poses:
        print("\nSkipping viewing angle vs inliers analysis (no pose data)")
        return

    print("\n--- Viewing Angle vs Inliers Analysis ---")

    angle_bin_edges = [0, 5, 10, 15, 20, 30, 45, 90]
    n_bins = len(angle_bin_edges) - 1
    bin_centers = [
        (angle_bin_edges[i] + angle_bin_edges[i + 1]) / 2
        for i in range(n_bins)
    ]
    bin_tick_labels = [
        f"{angle_bin_edges[i]}-{angle_bin_edges[i+1]}°"
        for i in range(n_bins)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Viewing Angle vs Inliers / Co-visibility (per-method)",
        fontsize=14, fontweight="bold"
    )

    for idx, label in enumerate(labels):
        poses = all_metrics[label]["image_poses"]
        obs_sets = all_metrics[label].get("image_obs_sets", {})
        db_stats = all_metrics[label].get("pair_db_stats", {})

        if len(poses) < 2:
            continue

        names = sorted(poses.keys())

        # Per-bin accumulators
        bin_true_inliers  = [[] for _ in range(n_bins)]
        bin_inlier_ratio  = [[] for _ in range(n_bins)]
        bin_covis_pts     = [[] for _ in range(n_bins)]
        bin_covis_ratio   = [[] for _ in range(n_bins)]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                p1, p2 = poses[n1], poses[n2]

                ang = angle_between_cameras(
                    p1["qvec"], p1["tvec"], p2["qvec"], p2["tvec"])

                # Find angle bin
                b = None
                for k in range(n_bins):
                    if angle_bin_edges[k] <= ang < angle_bin_edges[k + 1]:
                        b = k
                        break
                if b is None:
                    continue

                # True inliers from DB
                db_key = (min(n1, n2), max(n1, n2))
                if db_key in db_stats:
                    s = db_stats[db_key]
                    inliers = s["inliers"]
                    matches = s["matches"]
                    bin_true_inliers[b].append(inliers)
                    if matches > 0:
                        bin_inlier_ratio[b].append(inliers / matches)

                # Co-visibility from 3D reconstruction
                o1 = obs_sets.get(n1, frozenset())
                o2 = obs_sets.get(n2, frozenset())
                if o1 and o2:
                    shared = len(o1 & o2)
                    bin_covis_pts[b].append(shared)
                    bin_covis_ratio[b].append(shared / min(len(o1), len(o2)))

        def _means(bins):
            return [np.mean(b) if b else np.nan for b in bins]

        def _plot(ax, bins, ylabel, title):
            means = _means(bins)
            valid = [(bc, m) for bc, m in zip(bin_centers, means)
                     if not np.isnan(m)]
            if not valid:
                return
            vbc, vm = zip(*valid)
            ax.plot(vbc, vm, marker="o", linewidth=2,
                    label=label, color=_get_color(label))
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(bin_tick_labels, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel("Viewing Angle Bin")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        _plot(axes[0, 0], bin_true_inliers,
              "Mean Inlier Count", "True Inliers vs Viewing Angle\n(two_view_geometries)")
        _plot(axes[0, 1], bin_inlier_ratio,
              "Mean Inlier Ratio (inliers / raw matches)", "True Inlier Ratio vs Viewing Angle")
        _plot(axes[1, 0], bin_covis_pts,
              "Mean Co-visible 3D Points", "Co-visible 3D Points vs Viewing Angle\n(reconstruction)")
        _plot(axes[1, 1], bin_covis_ratio,
              "Mean Co-visibility Ratio", "Co-visibility Ratio vs Viewing Angle\n(shared / min(obs_i, obs_j))")

        n_db_pairs = sum(len(b) for b in bin_true_inliers)
        n_covis_pairs = sum(len(b) for b in bin_covis_pts)
        print(f"  {label}: {n_db_pairs} pairs with DB inliers, "
              f"{n_covis_pairs} pairs with co-visibility data")

    plt.tight_layout()
    out_path = output_dir / f"viewing_angle_vs_inliers{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# Timing comparison
# ============================================================================

def compare_timings(base_dir, output="timing_comparison.png"):
    """Load timings.json from each method directory and generate comparison plots."""
    base_dir = Path(base_dir)
    all_timings = {}

    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        timings_file = method_dir / "timings.json"
        if timings_file.exists():
            label = _normalize_label(method_dir.name.replace("_reconstruction", ""))
            with open(timings_file) as f:
                all_timings[label] = json.load(f)

    if not all_timings:
        print("No timings.json files found. Run reconstructions first.")
        return

    labels = list(all_timings.keys())
    n = len(labels)
    print(f"\nFound timings for {n} methods: {labels}")

    # Collect all stage names across methods
    all_stages = set()
    for t in all_timings.values():
        all_stages.update(t.keys())
    # Remove 'total' and 'total_pipeline' from stage breakdown
    breakdown_stages = sorted(all_stages - {"total", "total_pipeline"})

    colors_map = {
        "feature_extraction": "#4C72B0",
        "matching": "#DD8452",
        "aggregation": "#55A868",
        "geometric_verification": "#C44E52",
        "sparse_reconstruction": "#8172B3",
        "undistortion": "#937860",
        "patch_match": "#DA8BC3",
        "fusion": "#8C8C8C",
        "feature_extraction_and_matching": "#CCB974",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Pipeline Timing Comparison", fontsize=16, fontweight="bold")

    # --- 1. Total time bar chart ---
    ax = axes[0]
    totals = []
    for label in labels:
        t = all_timings[label]
        total = t.get("total", t.get("total_pipeline", sum(t.values())))
        totals.append(total)

    label_colors = [_get_color(l) for l in labels]
    bars = ax.barh(labels, totals, color=label_colors)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Total Pipeline Time")
    for bar, val in zip(bars, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}s", va="center", fontsize=9)

    # --- 2. Stacked bar chart (stage breakdown) ---
    ax = axes[1]
    bottom = np.zeros(n)
    for stage in breakdown_stages:
        vals = [all_timings[l].get(stage, 0) for l in labels]
        color = colors_map.get(stage, None)
        ax.barh(labels, vals, left=bottom, label=stage.replace("_", " ").title(),
                color=color)
        bottom += np.array(vals)

    ax.set_xlabel("Time (seconds)")
    ax.set_title("Time Breakdown by Stage")
    ax.legend(loc="lower right", fontsize=7)

    # --- 3. Per-stage grouped bars ---
    ax = axes[2]
    # Show key stages only
    key_stages = [s for s in ["feature_extraction", "matching",
                               "sparse_reconstruction", "patch_match"]
                  if s in all_stages]
    if not key_stages:
        key_stages = breakdown_stages[:4]

    x = np.arange(len(key_stages))
    width = 0.8 / n
    for i, label in enumerate(labels):
        vals = [all_timings[label].get(s, 0) for s in key_stages]
        ax.bar(x + i * width, vals, width, label=label, color=_get_color(label))

    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in key_stages],
                       rotation=30, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Key Stages Comparison")
    ax.legend(fontsize=7)

    plt.tight_layout()
    output_path = Path(output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print table
    print("\n" + "=" * 80)
    header = f"{'Method':<25}"
    for s in breakdown_stages:
        header += f" {s[:12]:>12}"
    header += f" {'TOTAL':>10}"
    print(header)
    print("=" * 80)
    for label in labels:
        row = f"{label:<25}"
        for s in breakdown_stages:
            val = all_timings[label].get(s, 0)
            row += f" {val:>11.1f}s"
        total = all_timings[label].get("total",
                all_timings[label].get("total_pipeline",
                sum(all_timings[label].values())))
        row += f" {total:>9.1f}s"
        print(row)
    print("=" * 80)


# ============================================================================
# Keypoint stats comparison
# ============================================================================

def compare_keypoint_stats(base_dir, output="keypoint_stats_comparison.png"):
    """Load keypoint_stats.json from each method directory and plot."""
    base_dir = Path(base_dir)
    all_stats = {}

    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        stats_file = method_dir / "keypoint_stats.json"
        if stats_file.exists():
            label = _normalize_label(method_dir.name.replace("_reconstruction", ""))
            with open(stats_file) as f:
                all_stats[label] = json.load(f)

    if not all_stats:
        print("No keypoint_stats.json files found. Run reconstructions first.")
        return

    labels = list(all_stats.keys())
    n = len(labels)
    label_colors = [_get_color(l) for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Keypoint Detection Comparison", fontsize=16, fontweight="bold")

    # 1. Mean keypoints per image (bar chart)
    ax = axes[0]
    means = [all_stats[l]["mean"] for l in labels]
    bars = ax.bar(labels, means, color=label_colors)
    ax.set_title("Mean Keypoints per Image")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    # 2. Per-image keypoint distribution (box plot)
    ax = axes[1]
    data = []
    for l in labels:
        counts = list(all_stats[l]["per_image"].values())
        data.append(counts)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], label_colors):
        patch.set_facecolor(c)
    ax.set_title("Keypoints per Image Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_path = Path(output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'Mean':>10} {'Median':>10} {'Total':>12} {'Images':>8}")
    print("=" * 70)
    for l in labels:
        s = all_stats[l]
        print(f"{l:<25} {s['mean']:>10,.0f} {s['median']:>10,.0f} "
              f"{s['total']:>12,} {s['num_images']:>8}")
    print("=" * 70)


# ============================================================================
# Auto-discovery
# ============================================================================

def _count_images_in_model(model_dir):
    """Read the registered-image count from a COLMAP sparse model's images.bin.

    The file header is a uint64 image count — cheap to read regardless of
    model size.
    """
    import struct as _struct
    images_bin = Path(model_dir) / "images.bin"
    if not images_bin.exists():
        # Fall back to file size as a rough proxy for .txt-format models
        images_txt = Path(model_dir) / "images.txt"
        return images_txt.stat().st_size if images_txt.exists() else 0
    try:
        with open(images_bin, "rb") as f:
            return _struct.unpack("<Q", f.read(8))[0]
    except Exception:
        return 0


def _pick_largest_model(candidates):
    """Pick the sub-model with the most registered images."""
    best = None
    best_count = -1
    for d in candidates:
        n = _count_images_in_model(d)
        if n > best_count:
            best = d
            best_count = n
    return best, best_count


def discover_reconstructions(base_dir):
    """Auto-discover reconstruction directories from run_colmap.py output.

    When COLMAP's mapper produces multiple sub-models (sparse/0, sparse/1, ...)
    we pick the one with the most registered images, not the latest by index.
    """
    base_dir = Path(base_dir)
    recon_dirs = []
    labels = []

    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        label = _normalize_label(
            method_dir.name.replace("_reconstruction", ""))

        # Look for sparse/<N>/ pattern — pick largest sub-model
        sparse_dir = method_dir / "sparse"
        if sparse_dir.exists():
            candidates = [d for d in sparse_dir.iterdir()
                          if d.is_dir() and (d / "points3D.bin").exists()]
            best, n = _pick_largest_model(candidates)
            if best is not None:
                if len(candidates) > 1:
                    print(f"  {label}: {len(candidates)} sub-models, "
                          f"selected {best.name} ({n} images)")
                recon_dirs.append(str(best))
                labels.append(label)
                continue

        # Also check direct method_dir/<N>/ pattern
        candidates = [d for d in method_dir.iterdir()
                      if d.is_dir() and (d / "points3D.bin").exists()]
        best, n = _pick_largest_model(candidates)
        if best is not None:
            if len(candidates) > 1:
                print(f"  {method_dir.name}: {len(candidates)} sub-models, "
                      f"selected {best.name} ({n} images)")
            recon_dirs.append(str(best))
            labels.append(_normalize_label(method_dir.name))

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

    # Derive suffix from base_dir name for unique plot filenames
    if args.base_dir:
        dir_suffix = "_" + Path(args.base_dir).resolve().name
    else:
        dir_suffix = ""

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
        output = str(Path(args.output).with_stem(
            Path(args.output).stem + dir_suffix))
        compare_reconstructions(existing_dirs, existing_labels, output)

    # Timing comparison (uses timings.json from method dirs)
    if args.base_dir:
        compare_timings(args.base_dir,
                        output=str(Path(args.output).with_name(
                            f"timing_comparison{dir_suffix}.png")))
        compare_keypoint_stats(args.base_dir,
                               output=str(Path(args.output).with_name(
                                   f"keypoint_stats_comparison{dir_suffix}.png")))
