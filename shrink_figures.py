#!/usr/bin/env python3
"""Produce smaller, still-legible copies of the paper figures.

Writes to paper/figures_small/ using IDENTICAL filenames, so the paper can be
switched over by changing one line in main.tex:

    \\graphicspath{{figures/}}        ->  \\graphicspath{{figures_small/}}

Two different treatments, because the figures are two different kinds of image:

  * Plots and point clouds are synthetic line art and scatter on white. They
    downscale cleanly and quantise to a 256-colour palette with no visible loss,
    which is where most of the saving comes from.
  * The correspondence figures are photographs with overlaid lines. Palette
    quantisation would band the imagery, so these are downscaled only and kept
    as full-colour RGB.

Text legibility is the binding constraint, so the default scale is conservative
(0.68 linear, i.e. ~46% of the pixels). The originals in paper/figures/ are left
untouched.
"""

import argparse
from pathlib import Path

from PIL import Image

# Figures that are photographic and must not be palette-quantised
PHOTO = {"matches_MVS.png", "matches_MVS-HyrdoThermal.png",
         "matches_MVS-cityhall.png"}


def shrink(src, dst, scale, quantise):
    im = Image.open(src)
    w, h = im.size
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    im = im.convert("RGB").resize(new, Image.LANCZOS)
    if quantise:
        # Synthetic plots use few distinct colours; a 256-entry palette is
        # visually lossless here and roughly halves the file again.
        im = im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
    im.save(dst, optimize=True)
    return (w, h), new, src.stat().st_size, dst.stat().st_size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.68,
                    help="Linear scale factor (default 0.68)")
    ap.add_argument("--src", default="paper/figures")
    ap.add_argument("--dst", default="paper/figures_small")
    args = ap.parse_args()

    src_dir, dst_dir = Path(args.src), Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tot_a = tot_b = 0
    print(f"{'figure':<44}{'from':>13}{'to':>13}{'KB':>9}{'saved':>8}")
    print("-" * 88)
    for p in sorted(src_dir.glob("*.png")):
        q = p.name not in PHOTO
        (w0, h0), (w1, h1), a, b = shrink(p, dst_dir / p.name, args.scale, q)
        tot_a += a; tot_b += b
        print(f"{p.name:<44}{f'{w0}x{h0}':>13}{f'{w1}x{h1}':>13}"
              f"{b/1024:>9.0f}{100*(1-b/a):>7.0f}%")
    # The photographic figures stay large as PNG because PNG cannot compress
    # photographic content. Also emit JPEG copies of just those three. Both live
    # side by side, so the paper picks whichever its \includegraphics names:
    # keep "name.png" for the drop-in PNG, or drop the extension to let pdflatex
    # prefer the much smaller JPEG.
    print()
    jpg_before = jpg_after = 0
    for name in sorted(PHOTO):
        src = src_dir / name
        if not src.exists():
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im = im.resize((round(w * args.scale), round(h * args.scale)), Image.LANCZOS)
        out = dst_dir / (Path(name).stem + ".jpg")
        im.save(out, quality=88, optimize=True, progressive=True)
        png_equiv = (dst_dir / name).stat().st_size
        jpg_before += png_equiv
        jpg_after += out.stat().st_size
        print(f"  {out.name:<42}{out.stat().st_size/1024:>8.0f} KB "
              f"(vs {png_equiv/1024:>6.0f} KB as PNG)")
    if jpg_before:
        print(f"  -> using JPEG for these three saves a further "
              f"{(jpg_before - jpg_after)/1048576:.1f} MB")

    # carry the JSON sidecars over so the directory is self-contained
    for p in src_dir.glob("*.json"):
        (dst_dir / p.name).write_bytes(p.read_bytes())
    print("-" * 88)
    print(f"{'TOTAL':<44}{'':>13}{'':>13}{tot_b/1024:>9.0f}"
          f"{100*(1-tot_b/max(tot_a,1)):>7.0f}%")
    print(f"\nOriginals: {tot_a/1048576:.1f} MB   Reduced: {tot_b/1048576:.1f} MB")
    print(f"Written to: {dst_dir.resolve()}")
