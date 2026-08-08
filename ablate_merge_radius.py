#!/usr/bin/env python3
"""Merge-radius ablation for the dense-matcher -> COLMAP aggregation step.

Answers the OCEANS reviewer's concern that aggregating dense pixel-level
correspondences into canonical keypoints by 3 px radius clustering "could
introduce systematic bias when comparing reprojection errors against sparse
methods".

The expensive dense matching pass is run ONCE (with run_colmap.py --keep-h5) and
cached to HDF5. This script then re-runs only the cheap part -- aggregation,
re-indexing, geometric verification, database write and the COLMAP mapper -- at a
range of merge radii, and reports how the reconstruction responds.

A radius of 0 is the no-clustering control: cKDTree.query_ball_point with r=0
returns only points at distance 0, so every distinct observation keeps its own
canonical keypoint (exact duplicates still collapse, which is intended).

Two things are measured:

  1. RECONSTRUCTION SENSITIVITY -- registered images, triangulated points, mean
     reprojection error and mean track length as a function of radius. If mean
     reprojection error is roughly flat in radius, the aggregation is not the
     dominant term in the dense-vs-sparse error comparison.

  2. QUANTISATION DISPLACEMENT -- the distance each raw observation is moved when
     it is replaced by its cluster centroid. This is the mechanism by which
     aggregation could bias reprojection error, so it is measured directly rather
     than assumed. For points uniformly distributed in a disc of radius r the
     expected distance to the centre is 2r/3; the empirical value is reported
     alongside it.

Usage:
    # 1. cache one dense matching pass
    python run_colmap.py --method loftr --images ./1_uav_images \
        --output /tmp/abl_base --keep-h5 /tmp/abl_pairs.h5

    # 2. sweep radii against that cache
    python ablate_merge_radius.py --h5 /tmp/abl_pairs.h5 \
        --images ./1_uav_images --workdir /tmp/abl --radii 0 1 2 3 4 6 8
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from colmap_db import ColmapDatabase, verify_matches_cv2
from colmap_pipeline import run_colmap_mapper, discover_images
from keypoint_aggregator import KeypointAggregator
from config import COLMAP_BIN
from render_pointclouds import read_points3D_bin, count_images_in_model, pick_largest_model


# --------------------------------------------------------------------------
# Quantisation displacement
# --------------------------------------------------------------------------

def measure_quantisation(h5_path, radius, max_images=12, rng_seed=0):
    """Distance each raw observation moves when replaced by its centroid.

    Returns dict with mean/median/p95/max displacement in pixels, plus the
    theoretical 2r/3 reference for uniform points in a disc of radius r.
    """
    agg = KeypointAggregator(merge_radius=radius, h5_path=str(h5_path),
                             read_only=True)
    try:
        # Index which pairs contribute observations to which image
        from collections import defaultdict
        idx = defaultdict(list)
        for pk in agg.h5["pairs"]:
            g = agg.h5["pairs"][pk]
            idx[g.attrs["img0"]].append((pk, True))
            idx[g.attrs["img1"]].append((pk, False))

        names = sorted(idx)
        rng = np.random.default_rng(rng_seed)
        if len(names) > max_images:
            names = [names[i] for i in
                     sorted(rng.choice(len(names), max_images, replace=False))]

        disp = []
        for nm in names:
            chunks = []
            for pk, is0 in idx[nm]:
                chunks.append(agg.h5["pairs"][pk]["pts0" if is0 else "pts1"][:]
                              .astype(np.float32))
            if not chunks:
                continue
            pts = np.concatenate(chunks, axis=0)
            canon = agg._cluster_points(pts)
            if len(canon) == 0:
                continue
            # Each raw observation is re-indexed to its nearest canonical
            # keypoint, exactly as reindex_matches() does.
            d, _ = cKDTree(canon).query(pts)
            disp.append(d)

        if not disp:
            return None
        d = np.concatenate(disp)
        return {
            "radius": float(radius),
            "n_observations": int(d.size),
            "mean_px": float(d.mean()),
            "median_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(d.max()),
            "theoretical_uniform_disc_2r_over_3": float(2.0 * radius / 3.0),
        }
    finally:
        agg.close()


# --------------------------------------------------------------------------
# One radius -> one reconstruction
# --------------------------------------------------------------------------

def run_one_radius(h5_path, image_dir, workdir, radius, min_inliers=15):
    """Re-aggregate the cached correspondences at `radius`, map, and measure."""
    workdir = Path(workdir) / f"r{radius:g}"
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "sparse").mkdir(parents=True)
    db_path = str(workdir / "database.db")

    t0 = time.time()
    agg = KeypointAggregator(merge_radius=radius, h5_path=str(h5_path),
                             read_only=True)
    keypoints, kd_trees = agg.aggregate()
    reindexed = agg.reindex_matches(kd_trees)
    agg.close()
    t_agg = time.time() - t0

    kp_counts = {n: len(k) for n, k in keypoints.items()}

    # Image dimensions come from the source images (the HDF5 stores only
    # correspondences, and COLMAP needs a camera per image).
    import cv2
    db = ColmapDatabase(db_path)
    image_id_map = {}
    for p in discover_images(image_dir):
        if p.name not in keypoints:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        cam_id = db.add_camera(w, h)
        iid = db.add_image(p.name, cam_id)
        image_id_map[p.name] = iid
        kps = keypoints[p.name]
        db.add_keypoints(iid, kps)
        db.add_descriptors(iid, np.zeros((len(kps), 128), dtype=np.float32))
    db.commit()

    t0 = time.time()
    verified = 0
    for n0, n1, matches, _ in reindexed:
        if n0 not in image_id_map or n1 not in image_id_map:
            continue
        inl, F = verify_matches_cv2(keypoints[n0], keypoints[n1], matches)
        if inl is None:
            continue
        db.add_matches(image_id_map[n0], image_id_map[n1], matches)
        db.add_two_view_geometry(image_id_map[n0], image_id_map[n1], inl, F)
        verified += 1
    db.commit()
    db.close()
    t_gv = time.time() - t0

    model, t_map = run_colmap_mapper(db_path, str(Path(image_dir).resolve()),
                                     str(workdir / "sparse"))

    res = {
        "radius": float(radius),
        "mean_keypoints_per_image": float(np.mean(list(kp_counts.values()))),
        "total_keypoints": int(sum(kp_counts.values())),
        "verified_pairs": int(verified),
        "total_pairs": int(len(reindexed)),
        "t_aggregate_s": round(t_agg, 1),
        "t_geometric_verification_s": round(t_gv, 1),
        "t_mapper_s": round(t_map, 1),
    }

    if model is None:
        res.update(registered_images=0, num_points3D=0,
                   mean_reproj_error=None, mean_track_length=None)
        return res

    best = pick_largest_model(workdir)
    best = best if best is not None else Path(model)
    xyz, rgb, err, tracks = read_points3D_bin(best / "points3D.bin")
    res.update(
        registered_images=int(count_images_in_model(best)),
        num_points3D=int(len(xyz)),
        mean_reproj_error=float(err.mean()) if len(err) else None,
        median_reproj_error=float(np.median(err)) if len(err) else None,
        mean_track_length=float(tracks.mean()) if len(tracks) else None,
        model=str(best),
    )
    return res


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot(results, quant, output):
    r = [x["radius"] for x in results]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    ax = axes[0]
    e = [x.get("mean_reproj_error") for x in results]
    # Reference band: the spread of mean reprojection error ACROSS the nine
    # methods in the main results table (SIFT 0.98 px to RoMa 1.76 px). Plotting
    # the ablation on this scale is the point of the panel -- it shows that the
    # merge radius moves the error far less than the choice of matcher does.
    ax.axhspan(0.98, 1.76, color="#cfd8e3", alpha=0.55, zorder=0,
               label="spread across methods")
    ax.plot(r, e, "o-", color="#17becf", lw=1.8, label="mean", zorder=3)
    if all(x.get("median_reproj_error") is not None for x in results):
        ax.plot(r, [x["median_reproj_error"] for x in results], "s--",
                color="#8c564b", lw=1.3, ms=3.5, label="median", zorder=3)
    ax.set_ylim(0.90, 1.90)
    ax.set_xlabel("Merge radius (px)")
    ax.set_ylabel("Reprojection error (px)")
    ax.set_title("Reconstruction accuracy", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(r, [x["registered_images"] for x in results], "o-",
            color="#2ca02c", lw=1.8)
    ax.set_xlabel("Merge radius (px)"); ax.set_ylabel("Registered images",
                                                      color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax2 = ax.twinx()
    ax2.plot(r, [x["num_points3D"] for x in results], "s--",
             color="#d62728", lw=1.5, ms=3.5)
    ax2.set_ylabel("3D points", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("Reconstruction size", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if quant:
        qr = [q["radius"] for q in quant]
        ax.plot(qr, [q["mean_px"] for q in quant], "o-", color="#1f77b4",
                lw=1.8, label="measured mean")
        ax.plot(qr, [q["p95_px"] for q in quant], "^-", color="#9467bd",
                lw=1.2, ms=3.5, label="measured p95")
        ax.plot(qr, [q["theoretical_uniform_disc_2r_over_3"] for q in quant],
                "k:", lw=1.2, label=r"uniform disc $2r/3$")
        ax.axhline(4.0, color="#d62728", ls="--", lw=1.0,
                   label="RANSAC threshold (4 px)")
    ax.set_xlabel("Merge radius (px)")
    ax.set_ylabel("Observation displacement (px)")
    ax.set_title("Quantisation displacement", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.tight_layout()
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Cached dense correspondences")
    ap.add_argument("--images", required=True, help="Source image directory")
    ap.add_argument("--workdir", default="/tmp/merge_radius_ablation")
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8])
    ap.add_argument("--output", default="paper/figures/merge_radius_ablation.png")
    ap.add_argument("--quant-only", action="store_true",
                    help="Only measure quantisation displacement (no mapper)")
    args = ap.parse_args()

    print("=== quantisation displacement ===")
    quant = []
    for rad in args.radii:
        q = measure_quantisation(args.h5, rad)
        if q:
            quant.append(q)
            print(f"  r={rad:>4.1f}  mean={q['mean_px']:.3f}px  "
                  f"median={q['median_px']:.3f}  p95={q['p95_px']:.3f}  "
                  f"max={q['max_px']:.3f}  (2r/3={q['theoretical_uniform_disc_2r_over_3']:.3f})"
                  f"  n={q['n_observations']:,}")

    results = []
    if not args.quant_only:
        print("\n=== reconstruction sweep ===")
        for rad in args.radii:
            print(f"\n--- radius {rad} px ---")
            try:
                res = run_one_radius(args.h5, args.images, args.workdir, rad)
                results.append(res)
                print(f"  Nr={res['registered_images']} Np={res['num_points3D']:,} "
                      f"e={res.get('mean_reproj_error')} "
                      f"kp/img={res['mean_keypoints_per_image']:.0f}")
            except Exception as exc:
                print(f"  FAILED: {exc}")

    outj = Path(args.output).with_suffix(".json")
    outj.parent.mkdir(parents=True, exist_ok=True)
    with open(outj, "w") as f:
        json.dump({"sweep": results, "quantisation": quant}, f, indent=2)
    print(f"\nSaved: {outj}")

    if results or quant:
        plot(results, quant, args.output)

    if results:
        print(f"\n{'r(px)':>6}{'Nr':>5}{'Np':>10}{'e_mean':>9}{'e_med':>8}"
              f"{'track':>8}{'kp/img':>9}")
        print("-" * 55)
        for x in results:
            print(f"{x['radius']:>6.1f}{x['registered_images']:>5}"
                  f"{x['num_points3D']:>10,}"
                  f"{(x.get('mean_reproj_error') or 0):>9.3f}"
                  f"{(x.get('median_reproj_error') or 0):>8.3f}"
                  f"{(x.get('mean_track_length') or 0):>8.2f}"
                  f"{x['mean_keypoints_per_image']:>9.0f}")
