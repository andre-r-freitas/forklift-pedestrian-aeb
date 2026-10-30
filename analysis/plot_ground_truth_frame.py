#!/usr/bin/env python3
"""
Show a ground-truth frame saved by the simulation (frames/gt_NNNNN.png) as a
color overlay on the matching camera frame, for the ground-truth figure in
Chapter 3.

Input: the ground-truth PNG written by the inference actor when
bSaveGroundTruthFrames is on. It holds the raw output of the stencil
post-process material:
    R = visible stencil ID (depth-tested: only pixels where the worker is
        in front of everything else)
    G = full-silhouette stencil ID (no depth test: the whole worker, also
        behind racks and other objects)
    0 = no worker.
The IDs are small numbers (1, 2, 3, ...), so the raw PNG looks black in an
image viewer; this script makes them visible.

For each stencil ID the overlay draws
    - the visible pixels as a solid color,
    - the occluded pixels (in G but not in R) as a striped, lighter color,
    - the outline of the full silhouette,
    - optionally (--boxes) the ground-truth box, which, as in the simulation,
      encloses the visible pixels only.

Usage
-----
    python plot_ground_truth_frame.py gt_00030.png --rgb frame_00030.png
    python plot_ground_truth_frame.py gt_00030.png --rgb frame_00030.png --boxes -o Camera_GT.png
    python plot_ground_truth_frame.py gt_00030.png            # on a black background

Cropped panels for a figure (the same --crop for both, so they line up):
    python plot_ground_truth_frame.py gt_00026.png --rgb frame_00026.png --boxes --no-legend --crop auto
        (prints e.g. "crop: --crop 264,264,260")
    python plot_ground_truth_frame.py gt_00026.png --raw --binary --scale 4 --crop 264,264,260

Requirements: numpy, Pillow, matplotlib (pip install numpy pillow matplotlib).
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Distinct colors per stencil ID (Okabe-Ito, readable for color-blind readers).
PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (255, 255, 255),
]


def load_gt(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32)


def outline(mask):
    """Pixels of the mask that have at least one 4-neighbor outside it."""
    p = np.pad(mask, 1, constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def blend(img, mask, color, alpha):
    c = np.array(color, dtype=np.float64)
    img[mask] = (1 - alpha) * img[mask] + alpha * c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_png", help="Ground-truth PNG (R = visible ID, G = full-silhouette ID)")
    ap.add_argument("--rgb", default=None, help="Matching camera frame (frame_NNNNN.png)")
    ap.add_argument("-o", "--output", default=None, help="Output image (default: <input>_plot.png)")
    ap.add_argument("--visible-alpha", type=float, default=0.6,
                    help="Opacity of the visible pixels (default 0.6)")
    ap.add_argument("--occluded-alpha", type=float, default=0.45,
                    help="Opacity of the occluded stripes (default 0.45)")
    ap.add_argument("--stripe", type=int, default=6, help="Stripe period in pixels (default 6)")
    ap.add_argument("--line", type=int, default=1, help="Outline/box thickness in pixels (default 1)")
    ap.add_argument("--boxes", action="store_true",
                    help="Draw the ground-truth box around the visible pixels of each ID")
    ap.add_argument("--darken", type=float, default=0.0,
                    help="Darken the camera frame by this fraction to make the overlay "
                         "stand out (0-1, default 0)")
    ap.add_argument("--no-legend", action="store_true", help="Omit the legend")
    ap.add_argument("--dpi", type=int, default=150, help="Output resolution (default 150)")
    ap.add_argument("--raw", action="store_true",
                    help="Instead of the overlay, save the stencil image itself with the IDs "
                         "brightened so they are visible: R and G are multiplied by 255/(highest "
                         "ID), so visible pixels look yellow (R+G) and occluded pixels green (G "
                         "only), and higher IDs are brighter. Same pixel size as the input.")
    ap.add_argument("--binary", action="store_true",
                    help="With --raw: every ID at full brightness (255) instead of scaled by ID")
    ap.add_argument("--crop", default=None,
                    help="Crop before drawing, identically for the ground truth and the camera "
                         "frame. Either 'x,y,size' (top-left corner and side of a square, in "
                         "pixels of the original frame) or 'auto' / 'auto:size' (square of the "
                         "given side, default 260, centered on all people in the frame). The "
                         "crop used is printed, so it can be repeated exactly.")
    ap.add_argument("--scale", type=int, default=1,
                    help="With --raw: enlarge the output by this integer factor with "
                         "nearest-neighbor, so edges stay sharp (default 1)")
    args = ap.parse_args()

    vis_id, full_id = load_gt(args.gt_png)
    H, W = vis_id.shape

    crop = None
    if args.crop:
        if args.crop.startswith("auto"):
            size = int(args.crop.split(":")[1]) if ":" in args.crop else 260
            ys, xs = np.nonzero((vis_id > 0) | (full_id > 0))
            if len(xs) == 0:
                sys.exit("--crop auto: no people in this frame; give x,y,size instead")
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        else:
            try:
                x, y, size = (int(v) for v in args.crop.split(","))
            except ValueError:
                sys.exit("--crop must be 'x,y,size' or 'auto[:size]'")
            cx, cy = x + size // 2, y + size // 2
        size = min(size, W, H)
        x0 = int(np.clip(cx - size // 2, 0, W - size))
        y0 = int(np.clip(cy - size // 2, 0, H - size))
        crop = (x0, y0, size)
        vis_id = vis_id[y0:y0 + size, x0:x0 + size]
        full_id = full_id[y0:y0 + size, x0:x0 + size]
        print(f"crop: --crop {x0},{y0},{size}")
    h, w = vis_id.shape

    if args.raw:
        max_id = int(max(vis_id.max(), full_id.max()))
        gain = 255 if args.binary else (255 // max_id if max_id else 0)
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        if args.binary:
            raw[..., 0] = np.where(vis_id > 0, 255, 0)
            raw[..., 1] = np.where(full_id > 0, 255, 0)
        else:
            raw[..., 0] = np.clip(vis_id * gain, 0, 255)
            raw[..., 1] = np.clip(full_id * gain, 0, 255)
        out = args.output or os.path.splitext(args.gt_png)[0] + "_raw.png"
        im = Image.fromarray(raw)
        if args.scale > 1:
            im = im.resize((w * args.scale, h * args.scale), Image.NEAREST)
        im.save(out)
        print(f"{args.gt_png}: {w}x{h}, IDs up to {max_id}, gain {gain}; saved {out}")
        return

    if args.rgb:
        rgb = np.array(Image.open(args.rgb).convert("RGB"))
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
            print(f"note: camera frame resized to the ground-truth size {W}x{H}")
        if crop:
            x0, y0, size = crop
            rgb = rgb[y0:y0 + size, x0:x0 + size]
        img = rgb.astype(np.float64) * (1 - args.darken)
    else:
        img = np.zeros((h, w, 3), dtype=np.float64)

    ids = sorted(set(np.unique(vis_id)) | set(np.unique(full_id)) - {0})
    ids = [i for i in ids if i != 0]
    if not ids:
        print("warning: no stencil IDs in this frame (no worker in view)")

    yy, xx = np.mgrid[0:h, 0:w]
    stripes = ((xx + yy) % args.stripe) < max(1, args.stripe // 2)

    print(f"{args.gt_png}: {w}x{h}")
    print(f"{'ID':>4} {'visible px':>11} {'full px':>9} {'visible %':>10}  box (x0,y0,x1,y1)")
    boxes = []
    for k, sid in enumerate(ids):
        color = PALETTE[k % len(PALETTE)]
        vis = vis_id == sid
        full = full_id == sid
        occl = full & ~vis
        blend(img, vis, color, args.visible_alpha)
        blend(img, occl & stripes, color, args.occluded_alpha)
        if full.any():
            img[dilate(outline(full), args.line - 1)] = color
        n_vis, n_full = int(vis.sum()), int(full.sum())
        frac = 100.0 * n_vis / n_full if n_full else float("nan")
        box = None
        if vis.any():
            ys, xs = np.nonzero(vis)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            boxes.append((box, color))
        print(f"{sid:>4} {n_vis:>11} {n_full:>9} {frac:>9.1f}%  {box}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(6.0, 6.0 * h / w))
    ax.imshow(np.clip(img, 0, 255).astype(np.uint8), interpolation="nearest")
    if args.boxes:
        for (x0, y0, x1, y1), color in boxes:
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                                   fill=False, lw=1.5, ls="--",
                                   ec=np.array(color) / 255.0))
    ax.set_axis_off()

    if not args.no_legend and ids:
        c = np.array(PALETTE[0]) / 255.0
        handles = [Patch(fc=(*c, args.visible_alpha), ec=c, label="Visible (R)"),
                   Patch(fc=(*c, args.occluded_alpha * 0.6), ec=c, hatch="////",
                         label="Occluded (G only)")]
        if args.boxes:
            handles.append(Patch(fc="none", ec=c, ls="--", label="Ground-truth box"))
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)

    fig.subplots_adjust(0, 0, 1, 1)
    out = args.output or os.path.splitext(args.gt_png)[0] + "_plot.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
