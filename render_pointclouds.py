#!/usr/bin/env python3
"""Render COLMAP sparse reconstructions side by side for every feature method.

Addresses the OCEANS reviewer request to "show the impact on the reconstructions"
by rendering the actual sparse point cloud each matcher produced.

Why an orthographic PCA projection rather than a perspective 3D scatter:
a perspective matplotlib scatter of a sparse cloud reads as a grey smear at print
size. Projecting onto the two dominant PCA axes gives a clean orthographic
"plan view" of the scene that shows structure and, critically, shows fragmentation.

Handling SfM gauge freedom: each reconstruction is recovered up to an arbitrary
similarity transform (7-DoF: rotation, translation, scale), so raw coordinates are
not comparable across methods. Every cloud is therefore independently normalised:

  1. centred on the robust (median) centroid of its points,
  2. rotated into its own principal axes via PCA of the point cloud,
  3. scaled so the 2nd-98th percentile extent of axis 1 maps to a common range.

Panels are therefore comparable in *shape and completeness*, not in absolute scale.
This is the standard way to compare reconstructions that have no common datum, and
is stated in the figure caption.

Outlier handling: sparse reconstructions routinely contain a handful of points
triangulated at near-infinite depth. Axis limits are set from percentiles
(default 1-99) rather than min/max, otherwise those few points collapse all the
real structure into a single pixel.

Usage:
    python render_pointclouds.py --base_dir /media/goku/data/hamza/MVS
    python render_pointclouds.py --base_dir /media/goku/data/hamza/MVS \
                                 --output paper/figures/pointclouds_MVS.png
"""

import argparse
import struct
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Fixed per-method colours, shared with compare_recons.py / plot_db_stats.py
# --------------------------------------------------------------------------
METHOD_COLORS = {
    "sift": "#1f77b4",
    "aliked": "#ff7f0e",
    "superpoint+superglue": "#2ca02c",
    "superpoint+lightglue": "#d62728",
    "aliked+lightglue": "#9467bd",
    "disk+lightglue": "#8c564b",
    "loftr": "#e377c2",
    "roma-tiny": "#7f7f7f",
    "roma-full": "#bcbd22",
    "dkm": "#17becf",
    "orb": "#aec7e8",
    "akaze": "#ffbb78",
}

# Display order and pretty names for the paper
METHOD_ORDER = [
    ("sift", "SIFT"),
    ("orb", "ORB"),
    ("akaze", "AKAZE"),
    ("superpoint+superglue", "SP + SG"),
    ("aliked", "ALIKED"),
    ("disk+lightglue", "DISK + LG"),
    ("loftr", "LoFTR"),
    ("roma-tiny", "RoMa"),
    ("dkm", "DKM"),
]

DATASET_TITLES = {
    "MVS": "Glacier (Svalbard, 66 images)",
    "MVS-HyrdoThermal": "Hydrothermal vent (Bio9, 108 images)",
    "MVS-cityhall": "City Hall (Montreal, 65 images)",
}


def _normalize_label(label):
    """`roma` on disk is the tiny_roma_v1_outdoor variant."""
    return "roma-tiny" if label.lower() == "roma" else label


# --------------------------------------------------------------------------
# COLMAP binary readers (no pycolmap dependency)
# --------------------------------------------------------------------------

def read_points3D_bin(path):
    """Return (xyz [N,3], rgb [N,3] uint8, error [N], track_len [N])."""
    xyz, rgb, err, tracks = [], [], [], []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)                                   # point id
            xyz.append(struct.unpack("<ddd", f.read(24)))
            rgb.append(struct.unpack("<BBB", f.read(3)))
            err.append(struct.unpack("<d", f.read(8))[0])
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)                       # skip track
            tracks.append(track_len)
    return (np.asarray(xyz, dtype=np.float64),
            np.asarray(rgb, dtype=np.uint8),
            np.asarray(err, dtype=np.float64),
            np.asarray(tracks, dtype=np.int64))


def read_images_bin(path):
    """Return (centers [M,3], names [M]) for registered images.

    Centres are returned sorted by image NAME. COLMAP stores images by image_id,
    which is *registration* order, not capture order -- joining those with a line
    produces meaningless spaghetti. All three datasets use lexicographically
    ordered filenames that follow acquisition order, so sorting by name recovers
    the true survey trajectory.
    """
    centers, names = [], []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            f.read(4)                                   # image id
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)                                   # camera id
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)                   # skip 2D points
            w, x, y, z = qvec
            R = np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ])
            centers.append(-R.T @ tvec)
            names.append(name.decode("utf-8"))
    if not centers:
        return np.zeros((0, 3)), []
    order = np.argsort(names)
    centers = np.asarray(centers, dtype=np.float64)[order]
    names = [names[i] for i in order]
    return centers, names


def count_images_in_model(model_dir):
    """Registered-image count straight from the images.bin header (uint64)."""
    p = Path(model_dir) / "images.bin"
    if not p.exists():
        return 0
    try:
        with open(p, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except Exception:
        return 0


def pick_largest_model(method_dir):
    """Return the sparse sub-model with the most registered images."""
    sparse = Path(method_dir) / "sparse"
    if not sparse.exists():
        return None
    cands = [d for d in sparse.iterdir()
             if d.is_dir() and (d / "points3D.bin").exists()]
    if not cands:
        return None
    return max(cands, key=count_images_in_model)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def pca_normalize(xyz, cams=None, lo=1.0, hi=99.0):
    """Centre, PCA-align and robustly scale a cloud (and its cameras).

    Returns (xyz_n, cams_n, eigenvalue_ratio).
    """
    # Robust centring: use a percentile-trimmed subset so a few wild outliers
    # do not drag the centroid.
    med = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - med, axis=1)
    keep = d <= np.percentile(d, 98.0)
    core = xyz[keep] if keep.sum() >= 10 else xyz

    centre = core.mean(axis=0)
    X = xyz - centre

    # PCA on the core points only
    C = np.cov((core - centre).T)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]          # descending variance
    evals, evecs = evals[order], evecs[:, order]
    Xr = X @ evecs

    # Robust isotropic scale from the dominant axis
    span = np.percentile(Xr[:, 0], hi) - np.percentile(Xr[:, 0], lo)
    scale = 1.0 / span if span > 1e-12 else 1.0
    Xr *= scale

    cams_r = None
    if cams is not None and len(cams):
        cams_r = ((cams - centre) @ evecs) * scale

    ratio = float(evals[1] / evals[0]) if evals[0] > 0 else 0.0
    return Xr, cams_r, ratio


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_dataset(base_dir, output, color_mode="depth", max_points=120000,
                   point_size=0.35, clip=(2.5, 97.5), ncols=3, panel=3.05):
    base_dir = Path(base_dir)
    suffix = base_dir.resolve().name

    # Collect what actually exists, in the fixed paper order
    found = []
    for key, pretty in METHOD_ORDER:
        disk_name = "roma" if key == "roma-tiny" else key
        mdir = base_dir / disk_name
        if not mdir.is_dir():
            continue
        model = pick_largest_model(mdir)
        if model is None:
            continue
        found.append((key, pretty, model))

    if not found:
        print(f"No reconstructions found under {base_dir}")
        return None

    n = len(found)
    nrows = int(np.ceil(n / ncols))

    # Panels are square (equal aspect), so the figure's own aspect is set by the
    # grid shape. A wide, shallow grid costs far less vertical space on the page
    # than a square one at the same printed width.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel * ncols, (panel - 0.10) * nrows))
    axes = np.atleast_1d(axes).ravel()

    # Scale annotation text with the panel size so labels stay legible when the
    # grid is widened and each panel shrinks.
    fs_title = max(6.0, min(9.0, 9.0 * panel / 3.05))
    fs_note = max(5.0, min(7.2, 7.2 * panel / 3.05))

    stats = {}
    for ax, (key, pretty, model) in zip(axes, found):
        xyz, rgb, err, tracks = read_points3D_bin(model / "points3D.bin")
        cams, _ = read_images_bin(model / "images.bin")
        nr, npts = len(cams), len(xyz)
        stats[key] = {
            "registered_images": int(nr),
            "num_points3D": int(npts),
            "model": str(model),
            "mean_track_length": float(tracks.mean()) if len(tracks) else 0.0,
            "mean_reproj_error": float(err.mean()) if len(err) else 0.0,
        }

        if npts < 10:
            ax.text(0.5, 0.5, "reconstruction\ntoo small", ha="center",
                    va="center", fontsize=8, color="#999", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        Xn, Cn, _ = pca_normalize(xyz, cams, lo=clip[0], hi=clip[1])

        # Compactness / diffuseness of the cloud, scale-invariant by construction:
        # ratio of the 99th-percentile to the median in-plane radius about the
        # cloud centre. A tight, well-constrained reconstruction sits near 1.5-2;
        # a diffuse one with many poorly-triangulated points has a heavy tail and
        # a much larger ratio. This is reported alongside reprojection error
        # because the two capture different failure modes.
        _r = np.linalg.norm(Xn[:, :2] - np.median(Xn[:, :2], axis=0), axis=1)
        _med = float(np.percentile(_r, 50))
        stats[key]["spread_ratio_p99_over_p50"] = (
            round(float(np.percentile(_r, 99)) / _med, 2) if _med > 1e-12 else None)

        # Subsample for render speed / file size, preserving spatial spread
        if len(Xn) > max_points:
            idx = np.random.default_rng(0).choice(len(Xn), max_points, replace=False)
            Xs, rgbs = Xn[idx], rgb[idx]
        else:
            Xs, rgbs = Xn, rgb

        # Colour. Underwater and glacier imagery is very dark, so the true RGB
        # carried by points3D.bin renders as near-black and conveys nothing at
        # print size. Depth along the 3rd PCA axis is far more legible and shows
        # the surface structure, so it is the default; RGB stays available.
        if color_mode == "rgb" and rgbs.std() > 3.0:
            # Normalise brightness so dark scenes remain visible
            c = rgbs.astype(np.float32) / 255.0
            p99 = np.percentile(c, 99)
            colors = np.clip(c / max(p99, 0.15), 0, 1)
            ax.scatter(Xs[:, 0], Xs[:, 1], s=point_size, c=colors,
                       linewidths=0, rasterized=True)
        else:
            depth = Xs[:, 2]
            vmin, vmax = np.percentile(depth, [3, 97])
            ax.scatter(Xs[:, 0], Xs[:, 1], s=point_size, c=depth,
                       cmap="viridis", vmin=vmin, vmax=vmax,
                       linewidths=0, rasterized=True)

        # Camera trajectory in acquisition order. Fragmented reconstructions show
        # up as long jumps between disconnected clusters, which is exactly the
        # failure mode the reviewer asked us to make visible.
        if Cn is not None and len(Cn) > 1:
            ax.plot(Cn[:, 0], Cn[:, 1], "-", color="#d62728", lw=0.55,
                    alpha=0.55, zorder=5, solid_capstyle="round")
            ax.scatter(Cn[:, 0], Cn[:, 1], s=2.2, c="#d62728",
                       marker="o", zorder=6, linewidths=0, alpha=0.9)

        # Square, symmetric window sized from robust percentiles. Equal windows
        # keep the panels visually comparable; percentiles stop a handful of
        # badly-triangulated far points from collapsing the real structure.
        xlo, xhi = np.percentile(Xn[:, 0], clip)
        ylo, yhi = np.percentile(Xn[:, 1], clip)
        cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
        half = 0.5 * max(xhi - xlo, yhi - ylo, 1e-9) * 1.08
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # Colour-coded frame ties the panel to every other figure in the paper
        c = METHOD_COLORS.get(key, "#333333")
        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(1.8)

        ax.set_title(f"{pretty}", fontsize=fs_title, color=c, fontweight="bold", pad=2)
        ax.text(0.5, -0.055, f"{nr} imgs · {npts:,} pts",
                ha="center", va="top", fontsize=fs_note, color="#333",
                transform=ax.transAxes)

    for ax in axes[len(found):]:
        ax.axis("off")

    fig.suptitle(DATASET_TITLES.get(suffix, suffix),
                 fontsize=max(8.5, min(11, 11 * panel / 3.05)),
                 fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.005, 1, 0.975])
    # Equal-aspect panels leave slack between columns; reclaiming it lets each
    # panel be larger for the same figure height, which matters when the grid is
    # wide and the print size is small.
    fig.subplots_adjust(wspace=0.04, hspace=0.22)

    out = Path(output) if output else Path(f"paper/figures/pointclouds_{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    js = out.with_suffix(".json")
    with open(js, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {js}")

    print(f"\n{'Method':<24}{'Nr':>6}{'Np':>12}{'track':>8}{'reproj':>8}{'spread':>8}")
    print("-" * 66)
    for k, v in stats.items():
        sp = v.get("spread_ratio_p99_over_p50")
        print(f"{k:<24}{v['registered_images']:>6}{v['num_points3D']:>12,}"
              f"{v['mean_track_length']:>8.2f}{v.get('mean_reproj_error', 0):>8.2f}"
              f"{(sp if sp is not None else float('nan')):>8.2f}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Render COLMAP sparse reconstructions per feature method")
    ap.add_argument("--base_dir", required=True,
                    help="Results dir containing per-method subdirectories")
    ap.add_argument("--output", default=None, help="Output PNG path")
    ap.add_argument("--color", default="depth", choices=["rgb", "depth"],
                    help="Point colouring (default: depth along 3rd PCA axis; "
                         "rgb is near-black for underwater/glacier scenes)")
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--point_size", type=float, default=0.35)
    ap.add_argument("--ncols", type=int, default=3,
                    help="Panels per row. A wide grid (e.g. 5) costs much less "
                         "vertical space on the page than a square one.")
    ap.add_argument("--panel", type=float, default=3.05,
                    help="Panel size in inches (default 3.05)")
    args = ap.parse_args()

    render_dataset(args.base_dir, args.output, color_mode=args.color,
                   max_points=args.max_points, point_size=args.point_size,
                   ncols=args.ncols, panel=args.panel)
