#!/usr/bin/env python3
"""Pose-plausibility audit: is a registered camera actually placed correctly?

Registration count alone can overstate reconstruction quality. COLMAP registers
an image whenever it can solve PnP with enough inliers, and wrong-but-consistent
matches can place a camera at a grossly wrong pose. A method can therefore
report more registered images than a competitor while producing a worse
trajectory.

We test this without ground truth by CROSS-METHOD CONSENSUS. For every pair of
images adjacent in capture order, each reconstruction that registered both gives
a distance between the two recovered camera centres. All reconstructions describe
the same physical camera motion up to a global similarity transform, so after
dividing each method's steps by its own median step the per-pair values should
agree across methods. Disagreement localises to whichever reconstruction placed
the cameras wrongly.

Concretely, for each adjacent pair seen by at least three methods we take the
across-method median as the consensus, form each method's ratio to it,
renormalise by that method's median ratio, and report the dispersion of the log
ratio together with the fraction of pairs deviating by more than 5x and 3x.

Two properties matter:

  * SPEED-INVARIANT. An earlier version of this diagnostic compared each method's
    largest consecutive step to its own median, which conflates bad poses with
    genuinely variable platform speed -- fatal for the hydrothermal survey, where
    HOV Alvin's velocity varies substantially over the dive. Because the
    consensus test compares different reconstructions of the SAME pair, the true
    motion is common to both sides of the comparison and cancels.

  * NON-CIRCULAR. Scoring against one designated reference would make that
    reference perfect by construction. Using the across-method median instead
    means no single reconstruction defines correctness.

Only pairs adjacent in the original ordering are used, so a method that
registers a sparse subset is not penalised for gaps it legitimately skipped;
and because the comparison is per-pair, methods registering different subsets
are still compared only where they overlap.
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


def gauge_normalised_steps(model_dir, order):
    """Consecutive-camera distances, divided by this model's own median step.

    Dividing by the median removes the reconstruction's arbitrary gauge scale, so
    values are comparable across methods. Returns {index: normalised step} keyed
    by position in the capture ordering.
    """
    cams = read_images_bin(Path(model_dir) / "images.bin")
    d = {}
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if a in cams and b in cams:
            d[i] = float(np.linalg.norm(cams[a] - cams[b]))
    if not d:
        return {}, len(cams)
    med = float(np.median(list(d.values())))
    if med <= 0:
        return {}, len(cams)
    return {i: v / med for i, v in d.items()}, len(cams)


def consensus_audit(per_method, min_methods=3):
    """Deviation of each method's steps from the across-method consensus.

    Returns {name: (n_pairs, log_dispersion, frac_gt5x, frac_gt3x)}.
    """
    allp = set().union(*[set(v) for v in per_method.values()]) if per_method else set()
    consensus = {
        i: float(np.median([per_method[n][i] for n in per_method if i in per_method[n]]))
        for i in allp
        if sum(i in per_method[n] for n in per_method) >= min_methods
    }
    out = {}
    for name, d in per_method.items():
        common = [i for i in d if i in consensus and consensus[i] > 0]
        if len(common) < 10:
            out[name] = (len(common), None, None, None)
            continue
        r = np.array([d[i] / consensus[i] for i in common])
        r = r / np.median(r)          # a method may sit at a constant offset
        out[name] = (len(common), float(np.std(np.log(r))),
                     float(np.mean((r > 5) | (r < 0.2))),
                     float(np.mean((r > 3) | (r < 1 / 3))))
    return out


if __name__ == "__main__":
    import json
    results = {}
    for ds, pretty, total in DATASETS:
        order = all_image_names(ds)
        if not order:
            continue

        per_method, nreg = {}, {}
        for k, name in METHODS:
            m = largest_model(DATA / ds / k)
            if m is None:
                continue
            per_method[name], nreg[name] = gauge_normalised_steps(m, order)
        for base, alt, k, name in EXTRA:
            if base != ds:
                continue
            m = largest_model(DATA / alt / k)
            if m is not None:
                per_method[name], nreg[name] = gauge_normalised_steps(m, order)

        # Variant re-runs are diagnostics, not part of the consensus itself
        core = {n: v for n, v in per_method.items() if "*" not in n}
        audited = consensus_audit({**core})
        for n, v in per_method.items():
            if "*" in n:
                audited.update(consensus_audit({**core, n: v}))

        print(f"\n=== {pretty} ({total} images) ===")
        print(f"  {'method':<12}{'Nr':>5}{'pairs':>7}{'log-disp':>10}"
              f"{'>5x off':>9}{'>3x off':>9}")
        print("  " + "-" * 52)
        results[pretty] = {}
        rows = sorted(audited.items(),
                      key=lambda kv: (kv[1][1] is None, kv[1][1] or 0))
        for name, (npair, disp, f5, f3) in rows:
            if disp is None:
                print(f"  {name:<12}{nreg.get(name,0):>5}{npair:>7}   too few pairs")
            else:
                flag = "   <-- inconsistent" if f5 and f5 > 0.15 else ""
                print(f"  {name:<12}{nreg.get(name,0):>5}{npair:>7}{disp:>10.2f}"
                      f"{f5*100:>8.1f}%{f3*100:>8.1f}%{flag}")
            results[pretty][name] = {
                "registered": nreg.get(name, 0), "consensus_pairs": npair,
                "log_dispersion": disp, "frac_gt5x": f5, "frac_gt3x": f3,
                "total_images": total,
            }

    out = Path("paper/figures/pose_plausibility.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
