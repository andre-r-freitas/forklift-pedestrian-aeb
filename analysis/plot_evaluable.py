#!/usr/bin/env python3
"""
Figure for Chapter 5 (Detection Metrics): evaluable people against ignore regions.

For each panel (a run folder and a frame index), the camera frame is cropped
around the person and drawn with
  - the ground-truth box (box of the visible pixels, from ground_truth.csv):
    solid green when the person is evaluable, dashed grey when it is an
    ignore region (visible fraction below --min-visible);
  - the detection box(es) from detections.csv, dashed orange;
  - a label with the visible fraction and how the detection is scored.

The evaluability test here covers the visibility rule only (the 0-30 m band
and the edge rule are not needed for the example frames).

Usage
-----
    python plot_evaluable.py RUN_A:59 RUN_B:61 -o Evaluable.pdf
    python plot_evaluable.py BoxOcluder1.2:59 BoxOcluder1.2:61 --crop 200

Each RUN folder contains detections.csv, ground_truth.csv and frame_NNNNN.png
(directly or in frames/).

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
    ap.add_argument("panels", nargs="+", help="RUN_FOLDER:FRAME for each panel")
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=None,
                    help="Side of the square crop in pixels (default: 2.2 x the largest box height)")
    ap.add_argument("-o", "--output", default="Evaluable.pdf")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    data = []
    for spec in args.panels:
        run, frame = spec.rsplit(":", 1)
        frame = int(frame)
        gt = pd.read_csv(os.path.join(run, "ground_truth.csv"))
        det = pd.read_csv(os.path.join(run, "detections.csv"))
        g = gt[gt.frame_index == frame]
        d = det[det.frame_index == frame]
        if g.empty:
            sys.exit(f"{spec}: no ground truth in this frame")
        img = np.array(Image.open(find(run, f"frame_{frame:05d}.png")).convert("RGB"))
        data.append((spec, img, g, d))

    side = args.crop or int(2.2 * max(g.height.max() for _, _, g, _ in data))
    fig, axes = plt.subplots(1, len(data), figsize=(3.1 * len(data), 3.3))
    axes = np.atleast_1d(axes)
    letters = "abcdefgh"

    for k, (ax, (spec, img, g, d)) in enumerate(zip(axes, data)):
        H, W = img.shape[:2]
        s = min(side, W, H)
        cx, cy = g.center_x.mean(), g.center_y.mean()
        l = int(np.clip(cx - s / 2, 0, W - s)); t = int(np.clip(cy - s / 2, 0, H - s))
        ax.imshow(img[t:t + s, l:l + s], interpolation="none")
        lines = []
        for _, r in g.iterrows():
            vis = r.visible_px / r.full_px if r.full_px else 0.0
            ok = vis >= args.min_visible
            ax.add_patch(Rectangle((r.center_x - r.width / 2 - l, r.center_y - r.height / 2 - t),
                                   r.width, r.height, fill=False,
                                   ec="#009E73" if ok else "#BBBBBB", lw=1.6,
                                   ls="-" if ok else (0, (3, 2))))
            status = "evaluable, true positive" if ok else "ignore region, detection ignored"
            lines.append(f"{100 * vis:.0f}% visible: {status}")
        for _, r in d.iterrows():
            ax.add_patch(Rectangle((r.center_x - r.width / 2 - l, r.center_y - r.height / 2 - t),
                                   r.width, r.height, fill=False, ec="#E69F00", lw=1.2,
                                   ls=(0, (2, 2))))
        ax.set_axis_off()
        ax.set_title(f"({letters[k]}) " + "\n".join(lines), fontsize=7.5)
        print(f"{spec}: " + "; ".join(lines))

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=600)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
