#!/usr/bin/env python3
"""One representative image per dataset, as a single three-tile figure.

Gives the reader the visual context the difficulty claims rest on: low-texture
ice with angle-dependent illumination, a turbid artificially-lit vent, and a
well-textured terrestrial control.

The three sequences have different native aspect ratios (1.33, 1.78, 1.78), so
each frame is centre-cropped to a common 3:2 before tiling; otherwise the tiles
cannot be laid out uniformly. Deep-sea frames are additionally CLAHE tone-mapped
for display only -- untouched they reproduce as near-black in print -- and this
is stated in the caption.
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, glob, index into the sorted sequence, apply tone mapping)
PANELS = [
    ("Glacier (Svalbard)", "Section-26-PNGs-Win/*.png", 0.5, False),
    ("Hydrothermal vent (Bio9)", "/media/goku/data/2025-Bio9-subset/*.png", 0.5, True),
    ("City Hall (Montreal)", "1_uav_images/*.JPG", 0.5, False),
]

TARGET_ASPECT = 3.0 / 2.0


def centre_crop(im, aspect):
    h, w = im.shape[:2]
    if w / h > aspect:                       # too wide, trim sides
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return im[:, x0:x0 + nw]
    nh = int(round(w / aspect))              # too tall, trim top/bottom
    y0 = (h - nh) // 2
    return im[y0:y0 + nh, :]


def tone_map(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="paper/figures/dataset_samples.png")
    ap.add_argument("--width", type=float, default=2.36,
                    help="Tile width in inches (3 tiles ~= textwidth)")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(PANELS),
                             figsize=(args.width * len(PANELS),
                                      args.width / TARGET_ASPECT + 0.30))
    for ax, (label, pat, frac, tm) in zip(np.atleast_1d(axes).ravel(), PANELS):
        fs = sorted(glob.glob(pat))
        if not fs:
            ax.axis("off"); print(f"  {label}: no images at {pat}"); continue
        p = fs[int(frac * (len(fs) - 1))]
        im = cv2.imread(p)
        im = centre_crop(im, TARGET_ASPECT)
        if tm:
            im = tone_map(im)
        # downsample for a sane file size; print size is ~2.4 in wide
        scale = 900 / im.shape[1]
        if scale < 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=8.5, fontweight="bold", pad=3)
        print(f"  {label}: {Path(p).name}")

    fig.tight_layout(pad=0.35)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image as _I
    w, h = _I.open(out).size
    print(f"Saved: {out}  ({w}x{h}, {7.15*h/w:.2f} in at textwidth)")
