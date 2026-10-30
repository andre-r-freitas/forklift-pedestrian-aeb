#!/usr/bin/env python3
"""
Plot a depth frame saved by the simulation as a color image with a scale bar,
as in the "Depth capture" panel of Figure 3.x in Chapter 3.

Input: the depth PNG written by the inference actor: 16-bit grayscale,
planar depth along the camera axis in millimeters, 0 = invalid
(see "depth_png" in run_metadata.json).

The depth is converted to meters, clipped to a maximum (40 m by default) so
that the near range, which matters for braking, keeps most of the color
range, and drawn with the perceptually uniform 'viridis' colormap. Invalid
pixels (0) are drawn in a separate color (default white).

Usage
-----
    python plot_depth_frame.py depth_00030.png
    python plot_depth_frame.py depth_00030.png -o Camera_Depth.png --max 40
    python plot_depth_frame.py depth_00030.png --max 25 --cmap magma --dpi 200

Requirements: numpy, Pillow, matplotlib (pip install numpy pillow matplotlib).
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def load_depth_m(path):
    """Load a 16-bit depth PNG (mm) and return depth in meters, NaN where invalid."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype != np.uint16:
        # Pillow may open 16-bit PNGs as int32 ('I' mode); accept that too.
        if arr.dtype.kind in "iu" and arr.max() <= 65535:
            arr = arr.astype(np.uint16)
        else:
            sys.exit(f"{path}: expected a 16-bit depth PNG, got {img.mode} / {arr.dtype}")
    depth = arr.astype(np.float64) / 1000.0
    depth[arr == 0] = np.nan
    return depth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("depth_png", help="16-bit depth PNG in millimeters (0 = invalid)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output image (default: <input>_plot.png)")
    ap.add_argument("--max", type=float, default=40.0,
                    help="Upper end of the color scale in meters; farther depths are "
                         "shown in the top color (default 40)")
    ap.add_argument("--min", type=float, default=0.0,
                    help="Lower end of the color scale in meters (default 0)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default viridis)")
    ap.add_argument("--invalid-color", default="white",
                    help="Color for invalid pixels (default white)")
    ap.add_argument("--label", default="Planar depth (m)", help="Color bar label")
    ap.add_argument("--dpi", type=int, default=150, help="Output resolution (default 150)")
    ap.add_argument("--no-colorbar", action="store_true", help="Omit the color bar")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = load_depth_m(args.depth_png)
    valid = np.isfinite(depth)
    if not valid.any():
        sys.exit("No valid depth pixels in the image.")

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(args.invalid_color)

    h, w = depth.shape
    fig_w = 5.0 if not args.no_colorbar else 4.3
    fig, ax = plt.subplots(figsize=(fig_w, 4.3 * h / w))
    im = ax.imshow(np.ma.masked_invalid(depth), cmap=cmap,
                   vmin=args.min, vmax=args.max, interpolation="nearest")
    ax.set_axis_off()
    if not args.no_colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(args.label)
    fig.tight_layout()

    out = args.output or os.path.splitext(args.depth_png)[0] + "_plot.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    d = depth[valid]
    print(f"{args.depth_png}: {w}x{h} pixels, {np.count_nonzero(~valid)} invalid")
    print(f"depth range {d.min():.2f}-{d.max():.2f} m; "
          f"{100 * np.mean(d > args.max):.1f}% of valid pixels beyond {args.max:g} m "
          f"(shown in the top color)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
