#!/usr/bin/env python3
"""Feature matching benchmark across multiple methods.

Supports 10 matchers: SIFT, SuperGlue, LoFTR, ALIKED, ALIKED+LG, SP+LG,
DISK, DISK+LG, RoMa, DKM.

Usage:
    python run_benchmark.py --image_dir ./Section-25-PNGs-Win --methods all --sequential
    python run_benchmark.py --image_dir ./images --methods sift,superglue,aliked+lg,disk+lg
    python run_benchmark.py --image_dir ./images --methods all --multi-gap
    python run_benchmark.py --image_dir ./images --methods all --all_pairs --no_vis
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import combinations 
import json
import warnings
warnings.filterwarnings("ignore")

from config import MAX_IMAGE_DIM, DEVICE
from matchers import init_matchers, MATCHER_NAMES
from matchers.base import MatchResult, BenchmarkSummary, compute_summary


# ============================================================================
# Utility Functions
# ============================================================================

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray, scale


def get_image_files(image_dir):
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = [f for f in sorted(image_dir.iterdir()) if f.suffix.lower() in extensions]
    print(f"Found {len(files)} images in {image_dir}")
    return files


def generate_pairs(image_files, mode="sequential", gap=1):
    """Generate image pairs.

    Args:
        mode: "sequential", "all_pairs", "skip_one", or "gap"
        gap: For "gap" mode, the step size between paired images.
    """
    if mode == "sequential":
        pairs = [(image_files[i], image_files[i + 1])
                 for i in range(len(image_files) - 1)]
    elif mode == "all_pairs":
        pairs = list(combinations(image_files, 2))
    elif mode == "skip_one":
        pairs = [(image_files[i], image_files[i + 2])
                 for i in range(len(image_files) - 2)]
    elif mode == "gap":
        pairs = [(image_files[i], image_files[i + gap])
                 for i in range(len(image_files) - gap)]
    else:
        pairs = [(image_files[i], image_files[i + 1])
                 for i in range(len(image_files) - 1)]

    print(f"Generated {len(pairs)} image pairs using '{mode}' mode" +
          (f" (gap={gap})" if mode == "gap" else ""))
    return pairs


def visualize_matches(img0, img1, mkpts0, mkpts1, inliers=None,
                      title="", max_matches=500):
    """Create side-by-side match visualization."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    h = max(h0, h1)
    w = w0 + w1
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0 if len(img0.shape) == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    canvas[:h1, w0:w0 + w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    n = len(mkpts0)
    if n > max_matches:
        idx = np.random.choice(n, max_matches, replace=False)
        mkpts0, mkpts1 = mkpts0[idx], mkpts1[idx]
        if inliers is not None:
            inliers = inliers[idx]

    for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
        pt0 = tuple(map(int, pt0))
        pt1 = (int(pt1[0] + w0), int(pt1[1]))
        color = (0, 255, 0) if (inliers is None or inliers[i]) else (0, 0, 255)
        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return canvas


# ============================================================================
# Core Benchmark
# ============================================================================

def _unload_matcher(matcher):
    """Free GPU memory held by a matcher."""
    import gc
    for attr in ("matching", "loftr", "aliked", "roma", "dkm",
                 "extractor", "matcher", "model"):
        obj = getattr(matcher, attr, None)
        if obj is not None:
            if hasattr(obj, "cpu"):
                try:
                    obj.cpu()
                except Exception:
                    pass
            setattr(matcher, attr, None)
    del matcher
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _save_method_csv(results, output_dir, method_name):
    """Save per-method results CSV (incremental, won't overwrite other methods)."""
    safe_name = method_name.replace("+", "_").replace(" ", "_")
    df = pd.DataFrame([
        {"method": r.method, "img0": r.img0_name, "img1": r.img1_name,
         "matches": r.num_matches, "inliers": r.num_inliers,
         "inlier_ratio": r.inlier_ratio, "time": r.time_taken}
        for r in results
    ])
    path = output_dir / f"results_{safe_name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def _load_all_method_csvs(output_dir):
    """Load all per-method result CSVs and combine."""
    output_dir = Path(output_dir)
    dfs = []
    for csv_path in sorted(output_dir.glob("results_*.csv")):
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            dfs.append(df)
            print(f"  Loaded: {csv_path.name} ({len(df)} rows, method={df['method'].iloc[0]})")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def combine_results(output_dir, image_dir=None):
    """Combine all per-method CSVs and generate plots + summary."""
    output_dir = Path(output_dir)
    print("\nCombining results from per-method CSVs...")
    df = _load_all_method_csvs(output_dir)
    if df.empty:
        print("No results found!")
        return

    # Reconstruct MatchResult objects for compatibility
    all_results = []
    for _, row in df.iterrows():
        all_results.append(MatchResult(
            method=row["method"], img0_name=row["img0"], img1_name=row["img1"],
            num_matches=int(row["matches"]), num_inliers=int(row["inliers"]),
            inlier_ratio=float(row["inlier_ratio"]), time_taken=float(row["time"]),
            mkpts0=np.array([]), mkpts1=np.array([]),
        ))

    methods = df["method"].unique().tolist()
    summaries = [compute_summary(all_results, m) for m in methods]

    # Print summary
    print(f"\n{'Method':<15} {'Pairs':>6} {'Avg Match':>10} {'Avg Inlier':>11} "
          f"{'Inlier %':>9} {'Avg Time':>9} {'Success':>8}")
    print("-" * 80)
    for s in summaries:
        print(f"{s.method:<15} {s.total_pairs:>6} {s.avg_matches:>10.1f} "
              f"{s.avg_inliers:>11.1f} {s.avg_inlier_ratio:>8.1%} "
              f"{s.avg_time:>8.2f}s {s.success_rate:>7.1f}%")

    # Save combined CSV
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    df_summary = pd.DataFrame([
        {"method": s.method, "total_pairs": s.total_pairs,
         "avg_matches": s.avg_matches, "std_matches": s.std_matches,
         "avg_inliers": s.avg_inliers, "std_inliers": s.std_inliers,
         "avg_inlier_ratio": s.avg_inlier_ratio,
         "avg_time": s.avg_time, "total_time": s.total_time,
         "success_rate": s.success_rate}
        for s in summaries
    ])
    df_summary.to_csv(output_dir / "summary.csv", index=False)

    # Use a dummy matchers list for plot generation
    class _DummyMatcher:
        def __init__(self, name): self.name = name
    dummy_matchers = [_DummyMatcher(m) for m in methods]

    # Reconstruct pairs list from data
    pairs_set = list(dict.fromkeys(zip(df["img0"], df["img1"])))

    _generate_plots(all_results, summaries, dummy_matchers, pairs_set, output_dir)

    print(f"\nCombined results saved to: {output_dir}")


def run_benchmark(image_dir, output_dir="./benchmark_results",
                  pair_mode="sequential", save_visualizations=True,
                  method_names=None):
    """Run benchmark one method at a time to avoid OOM. Results saved per-method."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = output_dir / "visualizations"
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)

    image_files = get_image_files(image_dir)
    if len(image_files) < 2:
        print("Need at least 2 images!")
        return None, None

    pairs = generate_pairs(image_files, pair_mode)

    from matchers import AVAILABLE_MATCHERS, MATCHER_NAMES as ALL_KEYS

    # Resolve registry keys to run
    if method_names is None:
        registry_keys = list(ALL_KEYS)
    else:
        registry_keys = [k for k in method_names if k in AVAILABLE_MATCHERS]
        for u in [k for k in method_names if k not in AVAILABLE_MATCHERS]:
            print(f"  Unknown matcher: {u}, skipping.")

    print(f"\nMethods to benchmark: {registry_keys}")

    all_results: List[MatchResult] = []

    for method_key in registry_keys:
        print(f"\n{'='*70}")
        print(f"  BENCHMARKING: {method_key}")
        print(f"{'='*70}")

        # Init single matcher by registry key
        matcher_list = init_matchers([method_key])
        if not matcher_list:
            print(f"  Failed to init {method_key}, skipping.")
            continue
        matcher = matcher_list[0]
        method_name = matcher.name  # display name for CSVs/plots

        method_results = []

        for pair_idx, (path0, path1) in enumerate(pairs):
            print(f"  Pair {pair_idx + 1}/{len(pairs)}: {path0.name} <-> {path1.name}",
                  end=" ", flush=True)
            result = matcher(str(path0), str(path1))
            method_results.append(result)
            print(f"M:{result.num_matches:4d} I:{result.num_inliers:4d} "
                  f"R:{result.inlier_ratio:.0%} T:{result.time_taken:.2f}s")

            if save_visualizations and result.num_matches > 0:
                img0, _, _ = load_image(str(path0))
                img1, _, _ = load_image(str(path1))
                mkpts0_vis = result.mkpts0.copy()
                mkpts1_vis = result.mkpts1.copy()

                if matcher.name in ("RoMa", "RoMa-full", "DKM"):
                    orig_img0 = cv2.imread(str(path0))
                    orig_img1 = cv2.imread(str(path1))
                    h0_vis, w0_vis = img0.shape[:2]
                    h0_orig, w0_orig = orig_img0.shape[:2]
                    h1_vis, w1_vis = img1.shape[:2]
                    h1_orig, w1_orig = orig_img1.shape[:2]
                    mkpts0_vis[:, 0] *= w0_vis / w0_orig
                    mkpts0_vis[:, 1] *= h0_vis / h0_orig
                    mkpts1_vis[:, 0] *= w1_vis / w1_orig
                    mkpts1_vis[:, 1] *= h1_vis / h1_orig

                vis = visualize_matches(
                    img0, img1, mkpts0_vis, mkpts1_vis, result.inliers,
                    f"{result.method} | M:{result.num_matches} "
                    f"I:{result.num_inliers} ({result.inlier_ratio:.0%})")
                cv2.imwrite(str(vis_dir / f"pair_{pair_idx:03d}_{path0.stem}_{path1.stem}_{method_name}.png"), vis)

        # Save this method's results
        _save_method_csv(method_results, output_dir, method_name)
        all_results.extend(method_results)

        # Print method summary
        s = compute_summary(method_results, method_name)
        print(f"\n  {method_name}: Avg Inliers={s.avg_inliers:.0f}, "
              f"Inlier%={s.avg_inlier_ratio:.1%}, Time={s.avg_time:.2f}s")

        # Unload matcher and free GPU
        _unload_matcher(matcher)
        del matcher_list, method_results

    # Generate combined summary + plots
    combine_results(output_dir)

    # Generate report
    method_display_names = list(dict.fromkeys(r.method for r in all_results))
    _generate_report(all_results,
                     [compute_summary(all_results, m) for m in method_display_names],
                     image_dir, image_files, pairs, pair_mode, output_dir)

    print(f"\nResults saved to: {output_dir}")
    return all_results, None


# ============================================================================
# Viewpoint Resilience Analysis (multi-gap)
# ============================================================================

def run_viewpoint_analysis(image_dir, output_dir="./benchmark_results",
                           method_names=None, gaps=None,
                           save_visualizations=False):
    """Run benchmark at multiple gap sizes to analyze viewpoint resilience.

    Produces viewpoint_resilience.csv and viewpoint_resilience.png showing
    how each method's performance degrades with increasing viewpoint change.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_files = get_image_files(image_dir)
    if gaps is None:
        max_gap = min(10, len(image_files) - 1)
        gaps = [g for g in [1, 2, 3, 5, 7, 10] if g < len(image_files)]

    print("\nInitializing matchers...")
    matchers = init_matchers(method_names)
    print(f"\nActive matchers: {[m.name for m in matchers]}")

    rows = []

    for gap in gaps:
        pairs = generate_pairs(image_files, mode="gap", gap=gap)
        if not pairs:
            continue

        print(f"\n{'='*50}")
        print(f"  Gap = {gap} ({len(pairs)} pairs)")
        print(f"{'='*50}")

        for pair_idx, (path0, path1) in enumerate(pairs):
            for matcher in matchers:
                result = matcher(str(path0), str(path1))
                rows.append({
                    "gap": gap,
                    "method": result.method,
                    "img0": result.img0_name,
                    "img1": result.img1_name,
                    "matches": result.num_matches,
                    "inliers": result.num_inliers,
                    "inlier_ratio": result.inlier_ratio,
                    "time": result.time_taken,
                })

            print(f"  Pair {pair_idx + 1}/{len(pairs)}: {path0.name} <-> {path1.name} "
                  + " | ".join(f"{m.name}:{rows[-len(matchers) + i]['inliers']}"
                               for i, m in enumerate(matchers)))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "viewpoint_resilience.csv", index=False)

    # Aggregate per gap per method
    agg = df.groupby(["gap", "method"]).agg(
        avg_inliers=("inliers", "mean"),
        std_inliers=("inliers", "std"),
        avg_inlier_ratio=("inlier_ratio", "mean"),
        avg_matches=("matches", "mean"),
        success_rate=("inliers", lambda x: (x >= 10).mean() * 100),
    ).reset_index()

    agg.to_csv(output_dir / "viewpoint_resilience_summary.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Viewpoint Resilience Analysis", fontsize=16, fontweight="bold")

    methods = df["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 3)))

    for ax, metric, ylabel, title in [
        (axes[0, 0], "avg_inliers", "Average Inliers", "Inliers vs Viewpoint Gap"),
        (axes[0, 1], "avg_inlier_ratio", "Average Inlier Ratio", "Inlier Ratio vs Viewpoint Gap"),
        (axes[1, 0], "avg_matches", "Average Matches", "Matches vs Viewpoint Gap"),
        (axes[1, 1], "success_rate", "Success Rate (%)", "Success Rate vs Viewpoint Gap"),
    ]:
        for i, method in enumerate(methods):
            mdata = agg[agg["method"] == method]
            ax.plot(mdata["gap"], mdata[metric], marker="o",
                    label=method, color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel("Gap Size (image separation)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(gaps)

    plt.tight_layout()
    plt.savefig(output_dir / "viewpoint_resilience.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nViewpoint analysis saved to: {output_dir}")
    return df


# ============================================================================
# Internal helpers
# ============================================================================

def _save_csvs(all_results, summaries, output_dir):
    df_results = pd.DataFrame([
        {"method": r.method, "img0": r.img0_name, "img1": r.img1_name,
         "matches": r.num_matches, "inliers": r.num_inliers,
         "inlier_ratio": r.inlier_ratio, "time": r.time_taken}
        for r in all_results
    ])
    df_results.to_csv(output_dir / "detailed_results.csv", index=False)

    df_summary = pd.DataFrame([
        {"method": s.method, "total_pairs": s.total_pairs,
         "avg_matches": s.avg_matches, "std_matches": s.std_matches,
         "avg_inliers": s.avg_inliers, "std_inliers": s.std_inliers,
         "avg_inlier_ratio": s.avg_inlier_ratio,
         "avg_time": s.avg_time, "total_time": s.total_time,
         "success_rate": s.success_rate}
        for s in summaries
    ])
    df_summary.to_csv(output_dir / "summary.csv", index=False)


def _generate_plots(all_results, summaries, matchers, pairs, output_dir):
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [s.method for s in summaries]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

    # Average inliers
    ax = axes[0, 0]
    avg_inliers = [s.avg_inliers for s in summaries]
    std_inliers = [s.std_inliers for s in summaries]
    bars = ax.bar(methods, avg_inliers, yerr=std_inliers, capsize=5, color=colors)
    ax.set_ylabel("Average Inliers")
    ax.set_title("Average Inliers per Method")
    ax.set_xticklabels(methods, rotation=45, ha="right")
    for bar, val in zip(bars, avg_inliers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # Inlier ratio
    ax = axes[0, 1]
    avg_ratios = [s.avg_inlier_ratio * 100 for s in summaries]
    bars = ax.bar(methods, avg_ratios, color=colors)
    ax.set_ylabel("Average Inlier Ratio (%)")
    ax.set_title("Average Inlier Ratio per Method")
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, avg_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Inliers per pair trend
    ax = axes[1, 0]
    for matcher in matchers:
        method_results = [r for r in all_results if r.method == matcher.name]
        inliers = [r.num_inliers for r in method_results]
        ax.plot(range(len(inliers)), inliers, marker="o",
                label=matcher.name, markersize=4)
    ax.set_xlabel("Pair Index")
    ax.set_ylabel("Number of Inliers")
    ax.set_title("Inliers per Image Pair")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Processing time
    ax = axes[1, 1]
    avg_times = [s.avg_time for s in summaries]
    bars = ax.bar(methods, avg_times, color=colors)
    ax.set_ylabel("Average Time (s)")
    ax.set_title("Average Processing Time per Method")
    ax.set_xticklabels(methods, rotation=45, ha="right")
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap
    if len(pairs) > 1:
        df_results = pd.DataFrame([
            {"method": r.method, "img0": r.img0_name, "img1": r.img1_name,
             "inliers": r.num_inliers}
            for r in all_results
        ])
        pivot_data = df_results.pivot(index=["img0", "img1"],
                                      columns="method", values="inliers")
        plt.figure(figsize=(12, max(6, len(pairs) * 0.4)))
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlGnBu",
                    cbar_kws={"label": "Inliers"})
        plt.title("Inliers Heatmap: Pairs vs Methods")
        plt.tight_layout()
        plt.savefig(output_dir / "inliers_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()


def _generate_report(all_results, summaries, image_dir, image_files,
                     pairs, pair_mode, output_dir):
    best_method = max(summaries, key=lambda s: s.avg_inliers)
    fastest_method = min(summaries, key=lambda s: s.avg_time)
    best_ratio_method = max(summaries, key=lambda s: s.avg_inlier_ratio)

    report = f"""
================================================================================
GLACIER IMAGE FEATURE MATCHING BENCHMARK REPORT
================================================================================

Dataset: {image_dir}
Number of Images: {len(image_files)}
Number of Pairs: {len(pairs)}
Pair Mode: {pair_mode}

RESULTS SUMMARY
---------------
Best Average Inliers:     {best_method.method} ({best_method.avg_inliers:.1f} +/- {best_method.std_inliers:.1f})
Best Inlier Ratio:        {best_ratio_method.method} ({best_ratio_method.avg_inlier_ratio:.1%})
Fastest Method:           {fastest_method.method} ({fastest_method.avg_time:.2f}s avg)

DETAILED METRICS
----------------
"""
    for s in summaries:
        report += f"""
{s.method}:
  - Avg Matches:     {s.avg_matches:.1f} +/- {s.std_matches:.1f}
  - Avg Inliers:     {s.avg_inliers:.1f} +/- {s.std_inliers:.1f}
  - Avg Inlier %:    {s.avg_inlier_ratio:.1%}
  - Avg Time:        {s.avg_time:.2f}s
  - Success Rate:    {s.success_rate:.1f}% (pairs with >=10 inliers)
"""

    report += f"""
================================================================================
Output Files:
  - detailed_results.csv    : Per-pair results for all methods
  - summary.csv             : Aggregated statistics
  - benchmark_plots.png     : Comparison charts
  - inliers_heatmap.png     : Heatmap visualization
  - visualizations/         : Match visualizations per pair
================================================================================
"""

    print(report)
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)


# ============================================================================
# Feature Detection Analysis
# ============================================================================

def run_feature_detection_analysis(image_dir, output_dir="./benchmark_results"):
    """Count raw features per image per detector and plot results.

    Uses the extractors module (same config as COLMAP pipeline) to extract
    features from each image and count them — uncapped, at MAX_IMAGE_DIM.
    """
    import torch
    from extractors.lightglue_extractor import LightGlueExtractor
    from extractors.superpoint_superglue import SuperPointSuperGlueExtractor

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_files = get_image_files(image_dir)
    device = DEVICE

    # Detectors to evaluate (sparse only — dense matchers don't have per-image features)
    detectors = {}

    # SIFT (OpenCV)
    print("Initializing detectors...")
    sift = cv2.SIFT_create(nfeatures=0)  # uncapped
    detectors["SIFT"] = ("opencv", sift)

    try:
        ext = SuperPointSuperGlueExtractor(device)
        detectors["SuperPoint"] = ("extractor", ext)
    except Exception as e:
        print(f"  SuperPoint init failed: {e}")

    for feat_type in ("aliked", "disk"):
        try:
            ext = LightGlueExtractor(feat_type, device)
            detectors[feat_type.upper()] = ("extractor", ext)
        except Exception as e:
            print(f"  {feat_type.upper()} init failed: {e}")

    print(f"Active detectors: {list(detectors.keys())}")

    # Count features per image per detector
    counts = {name: [] for name in detectors}
    image_names = []

    print("\nCounting features per image...")
    for img_path in image_files:
        image_names.append(img_path.name)
        img_bgr, gray, scale = load_image(str(img_path))

        for name, (dtype, det) in detectors.items():
            try:
                if dtype == "opencv":
                    kps = det.detect(gray, None)
                    n = len(kps)
                else:
                    feat = det.extract_features_image(img_path)
                    n = len(feat["kps_orig"]) if feat is not None else 0
                counts[name].append(n)
            except Exception as e:
                print(f"  {name} failed on {img_path.name}: {e}")
                counts[name].append(0)

        print(f"  {img_path.name}: " +
              " | ".join(f"{k}:{counts[k][-1]}" for k in detectors))

        if device == "cuda":
            torch.cuda.empty_cache()

    # Save to JSON
    stats = {}
    for name in detectors:
        c = counts[name]
        stats[name] = {
            "per_image": dict(zip(image_names, c)),
            "mean": float(np.mean(c)),
            "median": float(np.median(c)),
            "min": int(np.min(c)),
            "max": int(np.max(c)),
            "total": int(np.sum(c)),
        }

    with open(output_dir / "feature_detection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot
    detector_names = list(detectors.keys())
    n = len(detector_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Raw Feature Detection (uncapped, {MAX_IMAGE_DIM}px)",
                 fontsize=16, fontweight="bold")

    # 1. Mean feature count bar chart
    ax = axes[0]
    means = [stats[d]["mean"] for d in detector_names]
    bars = ax.bar(detector_names, means, color=colors[:n])
    ax.set_title("Mean Features per Image")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    # 2. Box plot of per-image distributions
    ax = axes[1]
    data = [counts[d] for d in detector_names]
    bp = ax.boxplot(data, labels=detector_names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors[:n]):
        patch.set_facecolor(c)
    ax.set_title("Feature Count Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    # 3. Per-image line plot (feature count across the image sequence)
    ax = axes[2]
    x = range(len(image_names))
    for i, name in enumerate(detector_names):
        ax.plot(x, counts[name], marker=".", markersize=3,
                label=name, color=colors[i], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Feature Count")
    ax.set_title("Features per Image (sequence)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_detection_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Detector':<15} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Total':>12}")
    print("=" * 70)
    for d in detector_names:
        s = stats[d]
        print(f"{d:<15} {s['mean']:>10,.0f} {s['median']:>10,.0f} "
              f"{s['min']:>8,} {s['max']:>8,} {s['total']:>12,}")
    print("=" * 70)

    # Cleanup
    for name, (dtype, det) in detectors.items():
        if dtype == "extractor" and hasattr(det, 'matching'):
            del det.matching
        if dtype == "extractor":
            del det
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nSaved: {output_dir / 'feature_detection_analysis.png'}")
    print(f"Saved: {output_dir / 'feature_detection_stats.json'}")
    return stats


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark feature matchers on glacier images")
    parser.add_argument("--image_dir", "-i", required=True,
                        help="Directory containing images")
    parser.add_argument("--output", "-o", default="./benchmark_results",
                        help="Output directory")
    parser.add_argument("--methods", "-m", default="all",
                        help=f"Comma-separated matcher names or 'all'. "
                             f"Available: {', '.join(MATCHER_NAMES)}")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential pairs (0-1, 1-2, ...)")
    parser.add_argument("--all_pairs", action=        "store_true",
                        help="Use all possible pairs")
    parser.add_argument("--skip_one", action="store_true",
                        help="Skip one image (0-2, 1-3, ...)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip saving visualizations")
    parser.add_argument("--multi-gap", action="store_true",
                        help="Run viewpoint resilience analysis at multiple gap sizes")
    parser.add_argument("--detect", action="store_true",
                        help="Run raw feature detection analysis (count features per image)")
    parser.add_argument("--combine", action="store_true",
                        help="Combine existing per-method CSVs and regenerate plots")
    args = parser.parse_args()

    # Parse methods
    if args.methods.lower() == "all":
        method_names = None  # all available
    else:
        method_names = [m.strip() for m in args.methods.split(",")]

    if args.combine:
        combine_results(output_dir=args.output)
    elif args.detect:
        run_feature_detection_analysis(
            image_dir=args.image_dir,
            output_dir=args.output,
        )
    elif args.multi_gap:
        run_viewpoint_analysis(
            image_dir=args.image_dir,
            output_dir=args.output,
            method_names=method_names,
            save_visualizations=not args.no_vis,
        )
    else:
        if args.all_pairs:
            pair_mode = "all_pairs"
        elif args.skip_one:
            pair_mode = "skip_one"
        else:
            pair_mode = "sequential"

        run_benchmark(
            image_dir=args.image_dir,
            output_dir=args.output,
            pair_mode=pair_mode,
            save_visualizations=not args.no_vis,
            method_names=method_names,
        )
