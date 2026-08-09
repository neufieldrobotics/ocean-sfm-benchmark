#!/usr/bin/env python3
"""Pose-plausibility audit: is a registered camera actually placed correctly?

Registration count alone can overstate reconstruction quality. COLMAP registers
an image whenever it can solve PnP with enough inliers, and wrong-but-consistent
matches can place a camera at a grossly wrong pose. A method can therefore
report more registered images than a competitor while producing a worse
trajectory.

We test this without ground truth by exploiting sequence continuity: all three
datasets are continuous surveys, so two images ADJACENT IN CAPTURE ORDER must
have camera centres close together relative to the survey's own step size. We
measure, over adjacent-in-sequence pairs that both registered:

    step_ratio_max = max(step) / median(step)
    jump_frac      = fraction of steps exceeding 5 x median

Both are scale-invariant and independent of the point cloud, so they are
comparable across methods whose reconstructions have different arbitrary
gauges. A smooth survey gives step_ratio_max of order 2-5. Large values mean
some cameras are placed far from where the sequence says they should be.

Only pairs adjacent in the ORIGINAL image ordering are used, so a method that
registers a sparse subset is not penalised for the gaps it legitimately skipped.

Usage:
    python audit_pose_plausibility.py
"""

import struct
from pathlib import Path

import numpy as np

DATA = Path("/media/goku/data/hamza")

DATASETS = [
    ("MVS", "Glacier", 66),
    ("MVS-HyrdoThermal", "Hydrothermal", 108),
    ("MVS-cityhall", "City Hall", 65),
]

METHODS = [
    ("sift", "SIFT"), ("orb", "ORB"), ("akaze", "AKAZE"),
    ("superpoint+superglue", "SP+SG"), ("aliked", "ALIKED"),
    ("disk+lightglue", "DISK+LG"), ("loftr", "LoFTR"),
    ("roma", "RoMa"), ("dkm", "DKM"),
]

# Re-run RoMa variants, compared against the originals above
EXTRA = [
    ("MVS", "MVS-romahybrid2", "roma", "RoMa*fixed"),
    ("MVS-cityhall", "MVS-cityhall-romahybrid2", "roma", "RoMa*fixed"),
    ("MVS-HyrdoThermal", "MVS-HyrdoThermal-romahybrid2", "roma", "RoMa*fixed"),
]


def read_images_bin(path):
    """Return {name: camera_centre}."""
    out = {}
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
            w, x, y, z = q
            R = np.array([
                [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
            out[nm.decode()] = -R.T @ t
    return out


def count_images(d):
    p = Path(d) / "images.bin"
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def largest_model(method_dir):
    sp = Path(method_dir) / "sparse"
    if not sp.exists():
        return None
    c = [d for d in sp.iterdir() if d.is_dir() and (d / "images.bin").exists()]
    return max(c, key=count_images) if c else None


def all_image_names(dataset):
    """Full capture ordering, taken from whichever model saw the most images."""
    best, n = None, -1
    for k, _ in METHODS:
        m = largest_model(DATA / dataset / k)
        if m is None:
            continue
        names = sorted(read_images_bin(m / "images.bin"))
        if len(names) > n:
            best, n = names, len(names)
    return best or []


def audit(model_dir, order):
    """step_ratio_max and jump_frac over adjacent-in-sequence registered pairs."""
    cams = read_images_bin(Path(model_dir) / "images.bin")
    pos = {i: n for i, n in enumerate(order)}
    steps = []
    for i in range(len(order) - 1):
        a, b = pos[i], pos[i + 1]
        if a in cams and b in cams:          # adjacent in capture order, both registered
            steps.append(float(np.linalg.norm(cams[a] - cams[b])))
    if len(steps) < 5:
        return len(cams), len(steps), None, None
    s = np.array(steps)
    med = np.median(s)
    if med <= 0:
        return len(cams), len(steps), None, None
    return len(cams), len(steps), float(s.max() / med), float(np.mean(s > 5 * med))


if __name__ == "__main__":
    import json
    results = {}
    for ds, pretty, total in DATASETS:
        order = all_image_names(ds)
        if not order:
            continue
        print(f"\n=== {pretty} ({total} images) ===")
        print(f"  {'method':<12}{'Nr':>7}{'adj pairs':>11}{'max/med step':>14}{'jumps>5x':>10}")
        print("  " + "-" * 54)
        rows = []
        for k, name in METHODS:
            m = largest_model(DATA / ds / k)
            if m is None:
                continue
            rows.append((name,) + audit(m, order))
        for base, alt, k, name in EXTRA:
            if base != ds:
                continue
            m = largest_model(DATA / alt / k)
            if m is not None:
                rows.append((name,) + audit(m, order))
        results[pretty] = {}
        for name, nr, npair, ratio, jf in rows:
            r = f"{ratio:>14.1f}" if ratio is not None else f"{'-':>14}"
            j = f"{jf*100:>9.0f}%" if jf is not None else f"{'-':>10}"
            flag = ""
            if ratio is not None and ratio > 10:
                flag = "   <-- implausible"
            print(f"  {name:<12}{nr:>7}{npair:>11}{r}{j}{flag}")
            results[pretty][name] = {
                "registered": nr, "adjacent_pairs": npair,
                "max_over_median_step": ratio,
                "jump_fraction_gt5x": jf,
                "total_images": total,
            }

    out = Path("paper/figures/pose_plausibility.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
