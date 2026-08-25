#!/usr/bin/env python3
"""Verified inlier counts as a function of pairwise viewing-angle baseline.

Addresses the OCEANS reviewer's first request. Two design decisions make this a
fair cross-method comparison rather than a misleading one:

1. COMMON POSE REFERENCE.
   The viewing angle between two cameras can only be computed from a
   reconstruction that registered both. If each method's angles were taken from
   its *own* reconstruction, then a method that registers only 23 of 108 images
   would be evaluated only on the pairs it already succeeded on -- a severe
   selection bias that flatters weak methods. Instead we fix ONE reference
   reconstruction per dataset (the one registering the most images), compute all
   pairwise angles once from its poses, and then evaluate every method's verified
   inlier count on that same fixed set of pairs.

2. MISSING PAIRS COUNT AS ZERO.
   A pair that never reached a method's two_view_geometries table is a pair whose
   geometric verification failed. Dropping such pairs would again bias the result
   upward for weak methods. They are counted as zero verified inliers, which is
   what they are.

Verified inliers are read from each method's own COLMAP database
(two_view_geometries.rows); raw matches from matches.rows. The inlier ratio
(verified / raw) normalises away the fact that dense matchers simply propose far
more correspondences, and is the fairer robustness measure of the two.

Usage:
    python plot_inliers_vs_angle.py
    python plot_inliers_vs_angle.py --output paper/figures/inliers_vs_angle.png
"""

import argparse
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/media/goku/data/hamza")

DATASETS = [
    ("MVS", "Glacier (Svalbard)"),
    ("MVS-HyrdoThermal", "Hydrothermal vent (Bio9)"),
    ("MVS-cityhall", "City Hall (Westmount)"),
]

METHODS = [
    ("sift", "SIFT", "#1f77b4"),
    ("orb", "ORB", "#aec7e8"),
    ("akaze", "AKAZE", "#ffbb78"),
    ("superpoint+superglue", "SP + SG", "#2ca02c"),
    ("aliked", "ALIKED", "#ff7f0e"),
    ("disk+lightglue", "DISK + LG", "#8c564b"),
    ("loftr", "LoFTR", "#e377c2"),
    ("roma", "RoMa", "#7f7f7f"),
    ("dkm", "DKM", "#17becf"),
]

# Bins chosen from the measured angle distributions. Medians are 33-58 deg and
# maxima reach 98-129 deg, so bins must span the full range; earlier exploratory
# bins that stopped at 90 deg lumped the majority of pairs into one bucket.
ANGLE_BINS = [0, 10, 20, 30, 45, 60, 80, 100, 130]

_COLMAP_MAX_IMAGE_ID = 2147483647


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def read_images_bin(path):
    """Return {name: (qvec, tvec)} for every registered image."""
    poses = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            f.read(4)
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n2d * 24)
            poses[name.decode("utf-8")] = (qvec, tvec)
    return poses


def count_images(model_dir):
    p = Path(model_dir) / "images.bin"
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def largest_model(method_dir):
    sparse = Path(method_dir) / "sparse"
    if not sparse.exists():
        return None
    c = [d for d in sparse.iterdir()
         if d.is_dir() and (d / "images.bin").exists()]
    return max(c, key=count_images) if c else None


def viewing_direction(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])
    return R.T @ np.array([0.0, 0.0, 1.0])


def pair_angle(p1, p2):
    d1, d2 = viewing_direction(p1[0]), viewing_direction(p2[0])
    return float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))


def read_db_pairs(db_path):
    """{(nameA,nameB): {'matches':int,'inliers':int}} with nameA < nameB."""
    conn = sqlite3.connect(str(db_path))
    id2name = {r[0]: r[1] for r in conn.execute("SELECT image_id, name FROM images")}
    out = {}
    for table, field in (("matches", "matches"), ("two_view_geometries", "inliers")):
        try:
            rows = conn.execute(f"SELECT pair_id, rows FROM {table}").fetchall()
        except sqlite3.OperationalError:
            continue
        for pair_id, nrows in rows:
            a = id2name.get(pair_id // _COLMAP_MAX_IMAGE_ID)
            b = id2name.get(pair_id % _COLMAP_MAX_IMAGE_ID)
            if a and b:
                k = (min(a, b), max(a, b))
                out.setdefault(k, {"matches": 0, "inliers": 0})[field] = nrows
    conn.close()
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def analyse_dataset(ds):
    base = DATA / ds

    # Reference reconstruction = the one registering the most images.
    ref_key, ref_model, ref_n = None, None, -1
    for key, _, _ in METHODS:
        m = largest_model(base / key)
        if m is None:
            continue
        n = count_images(m)
        if n > ref_n:
            ref_key, ref_model, ref_n = key, m, n
    if ref_model is None:
        return None

    poses = read_images_bin(ref_model / "images.bin")
    names = sorted(poses)

    pair_ang = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_ang[(a, b)] = pair_angle(poses[a], poses[b])

    nb = len(ANGLE_BINS) - 1
    bin_of = {}
    for k, ang in pair_ang.items():
        for b in range(nb):
            if ANGLE_BINS[b] <= ang < ANGLE_BINS[b + 1]:
                bin_of[k] = b
                break

    counts = np.zeros(nb, dtype=int)
    for b in bin_of.values():
        counts[b] += 1

    per_method = {}
    for key, pretty, color in METHODS:
        db = base / key / "database.db"
        if not db.exists():
            continue
        stats = read_db_pairs(db)
        inl = [[] for _ in range(nb)]
        rat = [[] for _ in range(nb)]
        for k, b in bin_of.items():
            s = stats.get(k)
            # Absent pair == verification failed == zero verified inliers.
            if s is None:
                inl[b].append(0); rat[b].append(0.0)
            else:
                inl[b].append(s["inliers"])
                # NOTE: COLMAP's native SIFT pipeline uses guided matching, which
                # re-matches along epipolar lines after estimating the two-view
                # geometry. Its verified set is therefore NOT a subset of the raw
                # match table and this ratio can exceed 1. The ratio is retained
                # in the JSON for reference but is not plotted, and must not be
                # compared across guided and non-guided pipelines.
                rat[b].append(s["inliers"] / s["matches"] if s["matches"] > 0 else 0.0)

        # Success rate: fraction of pairs in the bin reaching COLMAP's
        # min_num_inliers = 15, i.e. pairs the mapper could actually use.
        # Bounded in [0,1], so it is directly comparable across methods whose
        # raw inlier counts differ by two orders of magnitude.
        success = [float(np.mean(np.asarray(v) >= 15)) if v else np.nan for v in inl]

        per_method[key] = {
            "pretty": pretty, "color": color,
            "median_inliers": [float(np.median(v)) if v else np.nan for v in inl],
            "mean_inliers": [float(np.mean(v)) if v else np.nan for v in inl],
            "q1_inliers": [float(np.percentile(v, 25)) if v else np.nan for v in inl],
            "q3_inliers": [float(np.percentile(v, 75)) if v else np.nan for v in inl],
            "median_ratio": [float(np.median(v)) if v else np.nan for v in rat],
            "mean_ratio": [float(np.mean(v)) if v else np.nan for v in rat],
            "success_rate_ge15": success,
        }

    return {
        "reference_method": ref_key,
        "reference_registered": int(ref_n),
        "n_pairs": int(len(pair_ang)),
        "bin_counts": counts.tolist(),
        "bins": ANGLE_BINS,
        "methods": per_method,
    }


def plot(results, output):
    centers = [(ANGLE_BINS[i] + ANGLE_BINS[i + 1]) / 2 for i in range(len(ANGLE_BINS) - 1)]
    labels = [f"{ANGLE_BINS[i]}–{ANGLE_BINS[i+1]}" for i in range(len(ANGLE_BINS) - 1)]

    ncols = len(results)
    fig, axes = plt.subplots(2, ncols, figsize=(4.35 * ncols, 5.15), sharex="col")
    axes = np.atleast_2d(axes)

    for c, (ds, title) in enumerate([d for d in DATASETS if d[0] in results]):
        R = results[ds]
        counts = np.array(R["bin_counts"], dtype=float)

        for row, (kmed, ylab) in enumerate(
                [("mean_inliers", "Verified inliers / pair"),
                 ("success_rate_ge15", r"Frac. pairs $\geq$15 inliers")]):
            ax = axes[row, c]

            # Pair-count histogram behind the lines: a bin holding 4 pairs must
            # not be read like a bin holding 400.
            axb = ax.twinx()
            axb.bar(centers, counts, width=[ANGLE_BINS[i + 1] - ANGLE_BINS[i]
                                            for i in range(len(centers))],
                    color="#dfe6ee", edgecolor="none", zorder=0)
            axb.set_ylim(0, counts.max() * 4.2 if counts.max() else 1)
            axb.set_yticks([])
            axb.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

            for key, _, _ in METHODS:
                m = R["methods"].get(key)
                if m is None:
                    continue
                y = np.array(m[kmed], dtype=float)
                ok = ~np.isnan(y) & (counts > 0)
                if not ok.any():
                    continue
                ax.plot(np.array(centers)[ok], y[ok], marker="o", ms=3.2, lw=1.5,
                        color=m["color"], label=m["pretty"], zorder=3)

            if row == 0:
                ax.set_yscale("symlog", linthresh=10)
                ax.set_title(title, fontsize=10, fontweight="bold")
            else:
                ax.set_ylim(-0.03, 1.05)
                ax.set_xlabel("Pairwise viewing-angle baseline (deg)", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            ax.grid(True, alpha=0.25, zorder=1)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.tick_params(axis="y", labelsize=7.5)

        axes[0, c].text(
            0.98, 0.96,
            f"ref: {R['reference_method']} ({R['reference_registered']} imgs)\n"
            f"{R['n_pairs']:,} pairs",
            transform=axes[0, c].transAxes, ha="right", va="top",
            fontsize=6.5, color="#555",
            bbox=dict(fc="white", ec="#ccc", alpha=0.85, pad=2))

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=9, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle("Verified inliers vs pairwise viewing-angle baseline "
                 "(grey bars = number of pairs per bin)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="paper/figures/inliers_vs_angle.png")
    ap.add_argument("--replot", action="store_true",
                    help="Redraw from the saved JSON instead of re-reading the "
                         "COLMAP databases. Use when only presentation changes.")
    args = ap.parse_args()

    if args.replot:
        js = Path(args.output).with_suffix(".json")
        with open(js) as f:
            results = json.load(f)
        print(f"Loaded: {js}")
        plot(results, args.output)
        raise SystemExit

    results = {}
    for ds, title in DATASETS:
        print(f"\n=== {ds} ===")
        R = analyse_dataset(ds)
        if R is None:
            print("  no data"); continue
        results[ds] = R
        print(f"  reference: {R['reference_method']} "
              f"({R['reference_registered']} imgs), {R['n_pairs']:,} pairs")
        print(f"  bins  {R['bins']}")
        print(f"  n/bin {R['bin_counts']}")
        print(f"  {'method':<22}" + "".join(f"{l:>9}" for l in
              [f"{ANGLE_BINS[i]}-{ANGLE_BINS[i+1]}" for i in range(len(ANGLE_BINS) - 1)]))
        for key, _, _ in METHODS:
            m = R["methods"].get(key)
            if m is None:
                continue
            print(f"  {m['pretty']:<22}" + "".join(
                f"{v:>9.0f}" if not np.isnan(v) else f"{'-':>9}"
                for v in m["mean_inliers"]))
        print(f"  -- success rate (fraction of pairs with >=15 verified inliers) --")
        for key, _, _ in METHODS:
            m = R["methods"].get(key)
            if m is None:
                continue
            print(f"  {m['pretty']:<22}" + "".join(
                f"{v:>9.2f}" if not np.isnan(v) else f"{'-':>9}"
                for v in m["success_rate_ge15"]))

    plot(results, args.output)
    js = Path(args.output).with_suffix(".json")
    with open(js, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {js}")
