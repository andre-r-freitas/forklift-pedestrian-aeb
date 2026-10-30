#!/usr/bin/env python3
"""
Illustrate how the controller reads the depth of a detected person (Chapter 4,
Section "Depth of a Detected Person"), from one saved frame of a run.

Panels:
  (a) the camera frame, cropped around the detection, with the detection box
      (dashed) and the central region used for the depth (40% of the width,
      50% of the height, solid);
  (b) the rendered depth of the same crop with the same two rectangles;
  (c) the depth values inside the central region (histogram), with the
      25th percentile that the controller uses.

The box is taken from detections.csv of the run (the detection the controller
received for that frame). Depth values outside [min, max] valid depth are
ignored, as in the controller.

Usage
-----
    python plot_depth_reading.py RUN_FOLDER 40
    python plot_depth_reading.py RUN_FOLDER 40 --crop 240 -o DepthReading.pdf

RUN_FOLDER must contain detections.csv and frames/frame_NNNNN.png and
frames/depth_NNNNN.png (or the two PNGs directly in RUN_FOLDER).

Requirements: numpy, pandas, Pillow, matplotlib.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image


def find(run, name):
    for p in (os.path.join(run, "frames", name), os.path.join(run, name)):
        if os.path.exists(p):
            return p
    sys.exit(f"{name} not found in {run} or {run}/frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run folder")
    ap.add_argument("frame", type=int, help="Frame index")
    ap.add_argument("--det", type=int, default=None,
                    help="Row of the detection within the frame if there are several "
                         "(default: the one with the largest box)")
    ap.add_argument("--crop", type=int, default=None,
                    help="Side of the square crop in pixels (default: 1.6 x the box height, at least 120)")
    ap.add_argument("--percentile", type=float, default=25.0)
    ap.add_argument("--min-depth", type=float, default=1.8)
    ap.add_argument("--max-depth", type=float, default=60.0)
    ap.add_argument("--vmin", type=float, default=None,
                    help="Lower end of the depth color scale in m (default: 1 m before the estimate)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Upper end of the depth color scale in m (default: 6 m beyond the estimate)")
    ap.add_argument("-o", "--output", default="DepthReading.pdf")
    args = ap.parse_args()

    dets = pd.read_csv(os.path.join(args.run, "detections.csv"))
    d = dets[dets.frame_index == args.frame]
    if d.empty:
        sys.exit(f"No detection in frame {args.frame}")
    d = d.iloc[args.det] if args.det is not None else d.loc[(d.width * d.height).idxmax()]

    rgb = np.array(Image.open(find(args.run, f"frame_{args.frame:05d}.png")).convert("RGB"))
    dep = np.array(Image.open(find(args.run, f"depth_{args.frame:05d}.png"))).astype(float) / 1000.0
    dep[dep == 0] = np.nan
    H, W = dep.shape

    cx, cy, bw, bh = d.center_x, d.center_y, d.width, d.height
    box = (cx - bw / 2, cy - bh / 2, bw, bh)
    reg = (cx - 0.2 * bw, cy - 0.25 * bh, 0.4 * bw, 0.5 * bh)

    # Central region, pixel-inclusive as in the controller (floor / ceil, clamped).
    x0 = int(np.clip(np.floor(cx - 0.2 * bw), 0, W - 1)); x1 = int(np.clip(np.ceil(cx + 0.2 * bw), 0, W - 1))
    y0 = int(np.clip(np.floor(cy - 0.25 * bh), 0, H - 1)); y1 = int(np.clip(np.ceil(cy + 0.25 * bh), 0, H - 1))
    vals = dep[y0:y1 + 1, x0:x1 + 1].ravel()
    vals = vals[np.isfinite(vals) & (vals > args.min_depth) & (vals < args.max_depth)]
    if vals.size == 0:
        sys.exit("No valid depth values in the central region")
    vs = np.sort(vals)
    est = vs[int(np.floor(args.percentile / 100 * (vs.size - 1)))]

    # Crop around the detection
    s = min(args.crop or max(120, int(1.6 * bh)), W, H)
    l = int(np.clip(cx - s / 2, 0, W - s)); t = int(np.clip(cy - s / 2, 0, H - s))
    rgb_c = rgb[t:t + s, l:l + s]; dep_c = dep[t:t + s, l:l + s]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3),
                             gridspec_kw={"width_ratios": [1.3, 1.45, 1.0]})
    vmin = args.vmin if args.vmin is not None else max(0.0, est - 1.0)
    vmax = args.vmax if args.vmax is not None else est + 6.0

    def rects(ax):
        ax.add_patch(Rectangle((box[0] - l, box[1] - t), box[2], box[3], fill=False,
                               ec="white", lw=1.0, ls="--"))
        ax.add_patch(Rectangle((reg[0] - l, reg[1] - t), reg[2], reg[3], fill=False,
                               ec="#E69F00", lw=1.4))
        ax.set_axis_off()

    axes[0].imshow(rgb_c, interpolation="none"); rects(axes[0]); axes[0].set_title("(a) Camera frame", fontsize=8)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
    im = axes[1].imshow(np.ma.masked_invalid(dep_c), cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="none")
    rects(axes[1]); axes[1].set_title("(b) Rendered depth", fontsize=8)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.04)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Depth (m)")

    ax = axes[2]
    bins = np.linspace(vs.min(), vs.max(), 30)
    ax.hist(vs, bins=bins, color="#3A7DBF", edgecolor="none")
    ax.axvline(est, color="#E69F00", lw=1.2, ls="--")
    ax.text(est, ax.get_ylim()[1] * 0.95, f" {args.percentile:g}th percentile\n {est:.2f} m",
            va="top", ha="left", color="#B07800", fontsize=7)
    ax.set_xlabel("Depth in the region (m)"); ax.set_ylabel("Pixels")
    ax.set_title("(c) Depth values", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(args.output, bbox_inches="tight", dpi=600)
    print(f"frame {args.frame}: box {bw:.0f}x{bh:.0f} px, {vs.size} valid pixels in the region, "
          f"{args.percentile:g}th percentile {est:.3f} m (logged depth_m {d.depth_m:.3f} m); "
          f"saved {args.output}")


if __name__ == "__main__":
    main()
