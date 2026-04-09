#!/usr/bin/env python3
"""Plot keypoint and match statistics from COLMAP databases.

Reads database.db from each method directory and plots:
  - Average keypoints per image
  - Average raw matches per pair (before inlier rejection)
  - Average inlier matches per pair (two_view_geometries)
  - Total raw matches and inliers

Usage:
    python plot_db_stats.py /media/goku/data/hamza/MVS
    python plot_db_stats.py /media/goku/data/hamza/MVS --output db_stats.png
"""

import argparse
import sqlite3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def read_db_stats(db_path):
    """Read keypoint, match, and inlier counts from a COLMAP database."""
    conn = sqlite3.connect(str(db_path))

    # Keypoints per image
    kp_rows = conn.execute("SELECT rows FROM keypoints").fetchall()
    kp_counts = [r[0] for r in kp_rows]

    # Raw matches per pair (before geometric verification)
    match_rows = conn.execute("SELECT rows FROM matches").fetchall()
    match_counts = [r[0] for r in match_rows]

    # Inlier matches per pair (after geometric verification)
    inlier_rows = conn.execute("SELECT rows FROM two_view_geometries").fetchall()
    inlier_counts = [r[0] for r in inlier_rows]

    # Image count
    num_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    conn.close()

    return {
        "num_images": num_images,
        "num_pairs": len(match_counts),
        "num_verified_pairs": len(inlier_counts),
        "keypoints": kp_counts,
        "matches": match_counts,
        "inliers": inlier_counts,
        "avg_keypoints": float(np.mean(kp_counts)) if kp_counts else 0,
        "avg_matches": float(np.mean(match_counts)) if match_counts else 0,
        "avg_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0,
        "total_keypoints": int(np.sum(kp_counts)),
        "total_matches": int(np.sum(match_counts)),
        "total_inliers": int(np.sum(inlier_counts)),
    }


def discover_databases(base_dir):
    """Find all database.db files in method subdirectories."""
    base_dir = Path(base_dir)
    results = {}
    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        db_path = method_dir / "database.db"
        if db_path.exists():
            label = method_dir.name.replace("_reconstruction", "")
            results[label] = db_path
    return results


def plot_db_stats(base_dir, output="db_stats.png"):
    base_dir = Path(base_dir)
    databases = discover_databases(base_dir)

    if not databases:
        print("No database.db files found!")
        return

    print(f"Found {len(databases)} databases:")
    all_stats = {}
    for label, db_path in databases.items():
        stats = read_db_stats(db_path)
        all_stats[label] = stats
        print(f"  {label}: {stats['num_images']} imgs, "
              f"{stats['avg_keypoints']:.0f} avg kpts, "
              f"{stats['avg_matches']:.0f} avg matches, "
              f"{stats['avg_inliers']:.0f} avg inliers")

    labels = list(all_stats.keys())
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("COLMAP Database Statistics (before reconstruction)",
                 fontsize=16, fontweight="bold")

    # 1. Average keypoints per image
    ax = axes[0, 0]
    vals = [all_stats[l]["avg_keypoints"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Avg Keypoints per Image")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # 2. Average raw matches per pair
    ax = axes[0, 1]
    vals = [all_stats[l]["avg_matches"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Avg Raw Matches per Pair")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # 3. Average inliers per pair
    ax = axes[0, 2]
    vals = [all_stats[l]["avg_inliers"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Avg Inlier Matches per Pair")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # 4. Total keypoints
    ax = axes[1, 0]
    vals = [all_stats[l]["total_keypoints"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Total Keypoints (all images)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=8)

    # 5. Total raw matches
    ax = axes[1, 1]
    vals = [all_stats[l]["total_matches"] for l in labels]
    bars = ax.bar(labels, vals, color=colors[:n])
    ax.set_title("Total Raw Matches (all pairs)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=8)

    # 6. Keypoint distribution (box plot)
    ax = axes[1, 2]
    data = [all_stats[l]["keypoints"] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors[:n]):
        patch.set_facecolor(c)
    ax.set_title("Keypoints per Image Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_path = Path(output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Method':<20} {'Images':>7} {'Pairs':>7} {'Avg KPs':>10} "
          f"{'Avg Match':>10} {'Avg Inlier':>11} {'Tot Match':>12} {'Tot Inlier':>12}")
    print("=" * 100)
    for l in labels:
        s = all_stats[l]
        print(f"{l:<20} {s['num_images']:>7} {s['num_pairs']:>7} "
              f"{s['avg_keypoints']:>10,.0f} {s['avg_matches']:>10,.0f} "
              f"{s['avg_inliers']:>11,.0f} {s['total_matches']:>12,} "
              f"{s['total_inliers']:>12,}")
    print("=" * 100)

    # Save JSON
    json_stats = {l: {k: v for k, v in s.items()
                       if k not in ("keypoints", "matches", "inliers")}
                  for l, s in all_stats.items()}
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot keypoint/match stats from COLMAP databases")
    parser.add_argument("base_dir",
                        help="Base directory containing method subdirectories")
    parser.add_argument("--output", "-o", default="db_stats.png",
                        help="Output plot filename (default: db_stats.png)")
    args = parser.parse_args()

    plot_db_stats(args.base_dir, args.output)
