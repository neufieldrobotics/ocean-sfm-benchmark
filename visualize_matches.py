#!/usr/bin/env python3
"""Qualitative match visualisations on a representative image pair per dataset.

Promised in the accepted OCEANS abstract. For each dataset we select one image
pair with a meaningful viewing-angle baseline and draw the geometrically
verified correspondences each pipeline produced, so the reader can see *why* the
quantitative differences in Table I arise.

Correspondences are read directly from each method's COLMAP database
(`keypoints` and `two_view_geometries`) rather than by re-running the matchers.
This matters: it guarantees the figure shows exactly the data that produced the
reported reconstructions. Re-running would silently substitute a different
implementation for some methods -- our COLMAP ALIKED pipeline uses COLMAP's
native ALIKED_N16ROT + ALIKED_LIGHTGLUE, which is not the same code path as the
standalone ALIKED+LightGlue benchmark matcher -- and the figure would then
appear to contradict the table.

Pair selection is automatic and reported rather than hand-picked: we take the
reference reconstruction (the one registering the most images), compute all
pairwise viewing angles, and choose the pair whose angle is closest to a target
baseline. Consecutive frames would make every method succeed; the median
baseline would make nearly all of them fail. A moderate baseline is where the
methods actually separate.

Usage:
    python visualize_matches.py --dataset MVS-HyrdoThermal --target-angle 20
"""

import argparse
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/media/goku/data/hamza")
_COLMAP_MAX_IMAGE_ID = 2147483647

DATASET_IMAGES = {
    "MVS": "Section-26-PNGs-Win",
    "MVS-HyrdoThermal": "/media/goku/data/2025-Bio9-subset",
    "MVS-cityhall": "1_uav_images",
}
DATASET_TITLE = {
    "MVS": "Glacier (Svalbard)",
    "MVS-HyrdoThermal": "Hydrothermal vent (Bio9)",
    "MVS-cityhall": "City Hall (Westmount)",
}

PANEL_METHODS = [
    ("sift", "SIFT"),
    ("orb", "ORB"),
    ("akaze", "AKAZE"),
    ("superpoint+superglue", "SP + SG"),
    ("aliked", "ALIKED"),
    ("disk+lightglue", "DISK + LG"),
    ("loftr", "LoFTR"),
    ("roma", "RoMa"),
    ("dkm", "DKM"),
]

MAX_LINES = 200


# --------------------------------------------------------------------------

def read_images_bin(path):
    poses = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = np.array(struct.unpack("<dddd", f.read(32)))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)
            nm = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                nm += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            f.read(n2 * 24)
            poses[nm.decode()] = (q, t)
    return poses


def count_images(d):
    p = Path(d) / "images.bin"
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def view_dir(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return R.T @ np.array([0.0, 0.0, 1.0])


def _verified_pairs(db_path):
    """{(nameA,nameB): n_verified} for one method's database."""
    out = {}
    try:
        conn = sqlite3.connect(str(db_path))
        id2n = {r[0]: r[1] for r in conn.execute("SELECT image_id, name FROM images")}
        for pid, nrows in conn.execute(
                "SELECT pair_id, rows FROM two_view_geometries"):
            a, b = id2n.get(pid // _COLMAP_MAX_IMAGE_ID), id2n.get(pid % _COLMAP_MAX_IMAGE_ID)
            if a and b and nrows:
                out[(min(a, b), max(a, b))] = nrows
        conn.close()
    except sqlite3.Error:
        pass
    return out


def pick_pair(dataset, target_angle):
    """Pick a pair near `target_angle` that actually DISCRIMINATES the methods.

    Viewing angle alone is not sufficient: two cameras can share an orientation
    yet be far apart in space and observe no common scene, in which case every
    method fails and the figure is empty. We therefore additionally require that
    a useful number of methods verified the pair, and prefer pairs where some
    succeed and some fail -- that contrast is the point of the figure.
    """
    base = DATA / dataset
    best_model, best_n = None, -1
    for key, _ in PANEL_METHODS:
        sp = base / key / "sparse"
        if not sp.exists():
            continue
        for c in sp.iterdir():
            if c.is_dir() and (c / "images.bin").exists():
                n = count_images(c)
                if n > best_n:
                    best_model, best_n = c, n
    if best_model is None:
        return None, None, None, None

    poses = read_images_bin(best_model / "images.bin")
    names = sorted(poses)

    verified = {k: _verified_pairs(base / k / "database.db")
                for k, _ in PANEL_METHODS
                if (base / k / "database.db").exists()}
    n_methods = len(verified)

    best, best_score = None, -1e9
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d1, d2 = view_dir(poses[a][0]), view_dir(poses[b][0])
            ang = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1))))
            key = (min(a, b), max(a, b))
            k = sum(1 for v in verified.values() if key in v)
            if k == 0:
                continue
            # Prefer: angle near target, and roughly half the methods succeeding
            # so the panel grid shows genuine separation rather than all-pass or
            # all-fail.
            score = (-abs(ang - target_angle) / max(target_angle, 1.0)
                     - 1.6 * abs(k / n_methods - 0.6))
            if score > best_score:
                best, best_score = (a, b, ang, k), score
    if best is None:
        return None, None, None, None
    print(f"  selected pair verified by {best[3]}/{n_methods} methods")
    return best[0], best[1], best[2], best_model.parent.parent.name


def read_pair_correspondences(db_path, name0, name1):
    """Verified inlier correspondences for one pair, from a COLMAP database.

    Returns (pts0 [N,2], pts1 [N,2], n_raw_matches) in ORIGINAL image pixels,
    or (None, None, 0) if the pair is absent (verification failed).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        name2id = {r[1]: r[0] for r in
                   conn.execute("SELECT image_id, name FROM images")}
        if name0 not in name2id or name1 not in name2id:
            return None, None, 0
        i0, i1 = name2id[name0], name2id[name1]
        # COLMAP pair convention: smaller image_id first
        a, b = (i0, i1) if i0 < i1 else (i1, i0)
        flipped = i0 > i1
        pair_id = a * _COLMAP_MAX_IMAGE_ID + b

        row = conn.execute(
            "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?",
            (pair_id,)).fetchone()
        n_raw_row = conn.execute(
            "SELECT rows FROM matches WHERE pair_id=?", (pair_id,)).fetchone()
        n_raw = n_raw_row[0] if n_raw_row else 0
        if row is None or row[0] == 0 or row[2] is None:
            return None, None, n_raw
        nrows, ncols, blob = row
        m = np.frombuffer(blob, dtype=np.uint32).reshape(nrows, ncols)

        def kps(image_id):
            r = conn.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                (image_id,)).fetchone()
            if r is None or r[2] is None:
                return None
            kr, kc, kb = r
            return np.frombuffer(kb, dtype=np.float32).reshape(kr, kc)[:, :2]

        ka, kb_ = kps(a), kps(b)
        if ka is None or kb_ is None:
            return None, None, n_raw
        pa, pb = ka[m[:, 0]], kb_[m[:, 1]]
        # Return in (name0, name1) order regardless of COLMAP's internal order
        return (pb, pa, n_raw) if flipped else (pa, pb, n_raw)
    finally:
        conn.close()


def draw_panel(ax, img0, img1, mk0, mk1, label, note, ok=True):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    n_drawn = 0
    if mk0 is not None and len(mk0):
        idx = np.arange(len(mk0))
        if len(idx) > MAX_LINES:
            idx = np.random.default_rng(0).choice(idx, MAX_LINES, replace=False)
        for i in idx:
            p0 = (int(round(mk0[i][0])), int(round(mk0[i][1])))
            p1 = (int(round(mk1[i][0])) + w0, int(round(mk1[i][1])))
            cv2.line(canvas, p0, p1, (60, 220, 60), 1, cv2.LINE_AA)
            cv2.circle(canvas, p0, 2, (60, 220, 60), -1)
            cv2.circle(canvas, p1, 2, (60, 220, 60), -1)
        n_drawn = len(idx)

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=9, fontweight="bold", pad=2,
                 color="#111" if ok else "#b00")
    suffix = f" ({n_drawn} drawn)" if n_drawn else ""
    ax.set_xlabel(note + suffix, fontsize=6.8, labelpad=1.5)


def main(dataset, target_angle, output, max_dim=760, brighten=True,
         methods=None, ncols=3):
    imgdir = DATASET_IMAGES[dataset]
    imgdir = Path(imgdir) if str(imgdir).startswith("/") else Path(imgdir).resolve()

    n0, n1, ang, ref = pick_pair(dataset, target_angle)
    if n0 is None:
        print(f"{dataset}: no reconstruction available"); return
    p0, p1 = imgdir / n0, imgdir / n1
    if not p0.exists() or not p1.exists():
        print(f"{dataset}: images not found: {p0} / {p1}"); return
    print(f"{dataset}: pair {n0} <-> {n1}  baseline {ang:.1f} deg  (ref: {ref})")

    raw0, raw1 = cv2.imread(str(p0)), cv2.imread(str(p1))
    s0 = min(1.0, max_dim / max(raw0.shape[:2]))
    s1 = min(1.0, max_dim / max(raw1.shape[:2]))
    img0 = cv2.resize(raw0, None, fx=s0, fy=s0) if s0 < 1 else raw0
    img1 = cv2.resize(raw1, None, fx=s1, fy=s1) if s1 < 1 else raw1

    # Display-only tone mapping. Deep-sea imagery is close to black in print;
    # without this the reader cannot see the scene the matchers are working on.
    # Applied AFTER matching and purely for visualisation -- every reported
    # number comes from the original imagery -- and disclosed in the caption.
    if brighten:
        def _tone(im):
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        img0, img1 = _tone(img0), _tone(img1)

    panels = ([m for m in PANEL_METHODS if m[0] in methods] if methods
              else PANEL_METHODS)
    if methods:   # preserve the order the caller asked for
        panels = sorted(panels, key=lambda m: list(methods).index(m[0]))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.1 * ncols, 1.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    summary = []
    for ax, (key, pretty) in zip(axes, panels):
        db = DATA / dataset / key / "database.db"
        if not db.exists():
            draw_panel(ax, img0, img1, None, None, pretty, "no database", ok=False)
            continue
        mk0, mk1, n_raw = read_pair_correspondences(db, n0, n1)
        if mk0 is None:
            note = (f"{n_raw} raw matches, verification failed"
                    if n_raw else "no matches")
            draw_panel(ax, img0, img1, None, None, pretty, note, ok=False)
            summary.append((pretty, n_raw, 0, None))
            continue
        n_ver = len(mk0)
        # COLMAP's native SIFT pipeline uses guided matching: after estimating
        # the two-view geometry it re-matches along epipolar lines, so the
        # verified set is NOT a subset of the raw match table and the ratio
        # verified/raw can exceed one. Report it as guided rather than printing
        # a nonsensical percentage.
        if n_raw and n_ver <= n_raw:
            ratio = n_ver / n_raw
            note = f"{n_raw} raw, {n_ver} verified ({ratio:.0%})"
        else:
            ratio = None
            note = f"{n_ver} verified (guided matching)"
        draw_panel(ax, img0, img1, mk0 * s0, mk1 * s1, pretty, note)
        summary.append((pretty, n_raw, n_ver, ratio))
        rtxt = f"({ratio:.1%})" if ratio is not None else "(guided)"
        print(f"  {pretty:<12} raw={n_raw:>6} verified={n_ver:>6} {rtxt}")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{DATASET_TITLE[dataset]} — geometrically verified "
                 f"correspondences at a {ang:.0f}$^\\circ$ viewing-angle baseline",
                 fontsize=11, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_IMAGES))
    ap.add_argument("--target-angle", type=float, default=20.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-brighten", action="store_true",
                    help="Disable display-only CLAHE tone mapping")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Subset of registry keys to show, in display order")
    ap.add_argument("--ncols", type=int, default=3)
    args = ap.parse_args()
    main(args.dataset, args.target_angle,
         args.output or f"paper/figures/matches_{args.dataset}.png",
         brighten=not args.no_brighten, methods=args.methods,
         ncols=args.ncols)
