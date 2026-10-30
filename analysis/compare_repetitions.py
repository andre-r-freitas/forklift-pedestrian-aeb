#!/usr/bin/env python3
"""
Compare camera frames between repetitions of the same run.

For every frame index present in all repetition folders, the script
  1. loads the frame from each repetition as grayscale (0-255),
  2. computes the absolute pixel difference for every pair of repetitions
     and averages the pairs into one difference map,
  3. reports the mean difference over the whole image, over named
     rectangular regions (ROIs), and over "smooth" pixels, i.e. pixels whose
     local standard deviation in the first repetition is below a threshold.

This is the measurement behind the grain sentence in Section 3.1
("a grainy floor patch differed by about 6 of 255 intensity levels").
The repetitions must use the same seed, so that the camera is at the same
position in the same frame index.

Usage
-----
    python compare_repetitions.py Rep0 Rep1 Rep2
    python compare_repetitions.py Rep0 Rep1 --roi grainy=0.62,0.95,0.70,0.98 \
        --roi lane=0.70,0.98,0.25,0.50 --csv results.csv --heatmap diff.png

ROIs are given as name=top,bottom,left,right in fractions of the image
height (top, bottom) and width (left, right), so they do not depend on the
image resolution. Without --roi, the two default regions below are used.

Requirements: numpy, Pillow (pip install numpy pillow).
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np
from PIL import Image

# Default regions, as fractions (top, bottom, left, right).
# "grainy_floor": floor beside the right-hand racks, where the grain is strongest.
# "lane_floor":   plain floor in the center of the lane.
DEFAULT_ROIS = {
    "grainy_floor": (0.62, 0.95, 0.70, 0.98),
    "lane_floor": (0.70, 0.98, 0.25, 0.50),
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_roi(text):
    """Parse 'name=top,bottom,left,right' into (name, (t, b, l, r))."""
    try:
        name, values = text.split("=", 1)
        t, b, l, r = (float(v) for v in values.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"ROI '{text}' must look like name=top,bottom,left,right")
    if not (0 <= t < b <= 1 and 0 <= l < r <= 1):
        raise argparse.ArgumentTypeError(
            f"ROI '{text}': fractions must satisfy 0 <= top < bottom <= 1 "
            "and 0 <= left < right <= 1")
    return name.strip(), (t, b, l, r)


def load_gray(path):
    """Load an image as a float grayscale array with values 0-255."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def box_mean(img, size):
    """Mean over a size x size window (same output size, edges padded)."""
    pad = size // 2
    p = np.pad(img, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = img.shape
    s = (c[size:size + h, size:size + w] - c[:h, size:size + w]
         - c[size:size + h, :w] + c[:h, :w])
    return s / (size * size)


def local_std(img, size):
    """Local standard deviation over a size x size window."""
    m = box_mean(img, size)
    v = box_mean(img * img, size) - m * m
    return np.sqrt(np.clip(v, 0, None))


def roi_slice(shape, roi):
    h, w = shape
    t, b, l, r = roi
    return slice(int(t * h), int(b * h)), slice(int(l * w), int(r * w))


def common_frames(folders):
    sets = []
    for f in folders:
        names = {n for n in os.listdir(f) if n.lower().endswith(IMAGE_EXTENSIONS)}
        sets.append(names)
    return sorted(set.intersection(*sets))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+",
                    help="Two or more repetition folders with frames of the same names")
    ap.add_argument("--roi", action="append", type=parse_roi, default=None,
                    help="Region as name=top,bottom,left,right (fractions); repeatable")
    ap.add_argument("--smooth-threshold", type=float, default=1.0,
                    help="Local std (gray levels) below which a pixel counts as smooth "
                         "(default 1.0)")
    ap.add_argument("--window", type=int, default=7,
                    help="Window size in pixels for the local std (default 7)")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="Only these frame file names (default: all common frames)")
    ap.add_argument("--csv", default=None, help="Write the per-frame results to this CSV")
    ap.add_argument("--heatmap", default=None,
                    help="Save the difference map of the first frame as a PNG "
                         "(scaled so that 32 levels appear white)")
    args = ap.parse_args()

    if len(args.folders) < 2:
        sys.exit("Need at least two repetition folders.")
    for f in args.folders:
        if not os.path.isdir(f):
            sys.exit(f"Not a folder: {f}")

    rois = dict(args.roi) if args.roi else dict(DEFAULT_ROIS)
    frames = args.frames or common_frames(args.folders)
    if not frames:
        sys.exit("No frame file names are present in all folders.")

    pairs = list(itertools.combinations(range(len(args.folders)), 2))
    columns = ["frame", "whole_image", "smooth_pixels", "smooth_fraction"] + list(rois)
    rows = []

    for i, name in enumerate(frames):
        imgs = [load_gray(os.path.join(f, name)) for f in args.folders]
        shapes = {im.shape for im in imgs}
        if len(shapes) != 1:
            print(f"Skipping {name}: image sizes differ {shapes}", file=sys.stderr)
            continue

        # Mean absolute difference over all pairs of repetitions.
        diff = np.mean([np.abs(imgs[a] - imgs[b]) for a, b in pairs], axis=0)

        smooth = local_std(imgs[0], args.window) < args.smooth_threshold
        row = {
            "frame": name,
            "whole_image": diff.mean(),
            "smooth_pixels": diff[smooth].mean() if smooth.any() else float("nan"),
            "smooth_fraction": smooth.mean(),
        }
        for rname, roi in rois.items():
            ys, xs = roi_slice(diff.shape, roi)
            row[rname] = diff[ys, xs].mean()
        rows.append(row)

        if args.heatmap and i == 0:
            vis = np.clip(diff * (255.0 / 32.0), 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(args.heatmap)

    if not rows:
        sys.exit("No frames could be compared.")

    # Print the per-frame table and a summary.
    head = ["frame", "whole", "smooth"] + list(rois)
    print(f"Repetitions: {len(args.folders)}  pairs: {len(pairs)}  frames: {len(rows)}")
    print("Mean absolute difference between repetitions, in gray levels (0-255)\n")
    print("  ".join(f"{h:>16}" for h in head))
    for r in rows:
        vals = [r["frame"], r["whole_image"], r["smooth_pixels"]] + [r[n] for n in rois]
        print("  ".join(f"{v:>16}" if isinstance(v, str) else f"{v:16.3f}" for v in vals))

    print("\nSummary over frames (mean, min-max):")
    for key in ["whole_image", "smooth_pixels"] + list(rois):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"  {key:>16}: {np.nanmean(v):.3f}  ({np.nanmin(v):.3f}-{np.nanmax(v):.3f})")
    sf = np.mean([r["smooth_fraction"] for r in rows])
    print(f"  smooth pixels make up {100 * sf:.0f}% of the image "
          f"(local std < {args.smooth_threshold} over {args.window}x{args.window})")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nPer-frame results written to {args.csv}")
    if args.heatmap:
        print(f"Difference map of {rows[0]['frame']} saved to {args.heatmap}")


if __name__ == "__main__":
    main()
