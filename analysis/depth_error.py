#!/usr/bin/env python3
"""
How far is an estimated depth source (e.g. Depth Anything V2, written by estimate_depth.py) from Unreal's rendered
depth? Both are read the same way the controller reads depth: for every ground-truth person box, the 25th percentile
of the box's central region (40% of the width, 50% of the height), ignoring values under min_valid_depth_m (1.8 m,
the forklift's own parts) and above --max-depth. The difference is then purely the depth source, not the detector.

Usage:
    python depth_error.py <batch folder> --name da2s [--out <folder>]

Outputs (default <batch>/analysis_depth_<name>/):
    depth_err_boxes.csv     one row per person box: true (Unreal) and estimated depth, both readings of the estimate
    depth_err_summary.csv   per distance band (Unreal depth): median signed / absolute / relative error, p90, bias
    depth_err_image.csv     whole-image metrics per sampled frame: AbsRel, RMSE, delta<1.25, best global scale
    F9_depth_estimate.pdf/.png

Two readings of the estimate are reported, because a monocular network may output distance along the viewing ray
(range) rather than along the camera axis (planar, which is what Unreal writes):
    as_planar       the value as written
    from_range      value x cos(angle of the pixel from the optical axis), i.e. converted from range to planar
The one whose error does not grow towards the image edges is the right reading; with a 90 deg field of view the two
differ by up to 29% in the corners (22% at the left/right edge). The "_scaled" readings remove one global scale
factor per frame (median of true / estimated over the image, true depth <= --image-max-depth): what remains is the
error in shape, not in scale. That scale needs the true depth, so it is an analysis aid, not something a vehicle has.

Requires: numpy, pandas, opencv-python, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [0, 5, 10, 15, 20, 30, 60]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--name", default="da2s")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--percentile", type=float, default=0.25)
    p.add_argument("--max-depth", type=float, default=60.0)
    p.add_argument("--min-vis", type=float, default=0.5)
    p.add_argument("--max-near-frac", type=float, default=0.1,
                   help="Skip a box when more than this share of its center region is closer than the minimum valid "
                        "depth (the person is too close to measure; the percentile would land on the floor behind)")
    p.add_argument("--image-every", type=int, default=5, help="Whole-image metrics on every N-th frame")
    p.add_argument("--image-max-depth", type=float, default=20.0, help="Whole-image metrics only where true depth <= this")
    return p.parse_args()


def read_png_m(path: Path):
    import cv2
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) / 1000.0


def box_value(depth: np.ndarray, cx, cy, w, h, lo, hi, q, cosmap=None):
    H, W = depth.shape
    x0 = int(np.clip(math.floor(cx - 0.2 * w), 0, W - 1)); x1 = int(np.clip(math.ceil(cx + 0.2 * w), 0, W - 1))
    y0 = int(np.clip(math.floor(cy - 0.25 * h), 0, H - 1)); y1 = int(np.clip(math.ceil(cy + 0.25 * h), 0, H - 1))
    v = depth[y0:y1 + 1, x0:x1 + 1]
    if cosmap is not None:
        v = v * cosmap[y0:y1 + 1, x0:x1 + 1]
    v = np.sort(v[(v > lo) & (v < hi)].ravel())
    if not v.size:
        return np.nan
    return float(v[int(np.clip(math.floor(q * (v.size - 1)), 0, v.size - 1))])


def main() -> None:
    args = parse_args()
    root = args.path.resolve()
    out = (args.out or root / f"analysis_depth_{args.name}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in root.rglob(f"depth_{args.name}") if p.is_dir() and (p.parent / "run_metadata.json").exists())
    if not runs:
        raise SystemExit(f"No run folders with depth_{args.name}/ under {root}")

    cosmap = None
    rows, img_rows = [], []
    skipped_near = 0
    for k, rd in enumerate(runs, 1):
        print(f"\r  [{k}/{len(runs)}] {rd.name:<48}", end="", flush=True)
        meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
        lo = max(0.01, float(meta.get("min_valid_depth_m", 0.0)))
        scen = str(meta.get("scenario_id", ""))
        suffix = str(meta.get("variant_suffix", "") or "")
        scen = scen[: -len(suffix)] if suffix and scen.endswith(suffix) else scen
        gt = pd.read_csv(rd / "ground_truth.csv") if (rd / "ground_truth.csv").exists() else pd.DataFrame()
        for fi in sorted(int(p.stem.split("_")[1]) for p in (rd / f"depth_{args.name}").glob("depth_*.png")):
            true = read_png_m(rd / "frames" / f"depth_{fi:05d}.png")
            est = read_png_m(rd / f"depth_{args.name}" / f"depth_{fi:05d}.png")
            if true is None or est is None:
                continue
            if cosmap is None or cosmap.shape != est.shape:
                H, W = est.shape
                f = (0.5 * W) / math.tan(math.radians(0.5 * args.hfov))
                u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
                cosmap = (f / np.sqrt(f * f + u * u + v * v)).astype(np.float32)
            valid = (true > lo) & (true <= args.image_max_depth) & (est > 0.01)
            scale = float(np.median(true[valid] / est[valid])) if valid.sum() > 100 else np.nan
            scale_r = float(np.median(true[valid] / (est * cosmap)[valid])) if valid.sum() > 100 else np.nan
            if fi % args.image_every == 0 and valid.sum() > 100:
                t, e = true[valid], est[valid]
                e_r = (est * cosmap)[valid]
                ratio = np.maximum(t / e, e / t)
                img_rows.append({"run": rd.name, "scenario": scen, "frame_index": fi, "pixels": int(valid.sum()),
                                 "absrel": float(np.mean(np.abs(e - t) / t)),
                                 "absrel_from_range": float(np.mean(np.abs(e_r - t) / t)),
                                 "rmse_m": float(np.sqrt(np.mean((e - t) ** 2))),
                                 "delta1": float(np.mean(ratio < 1.25)), "global_scale": scale,
                                 "global_scale_from_range": scale_r})
            if gt.empty:
                continue
            g = gt[gt.frame_index == fi]
            for r in g.itertuples(index=False):
                vis = r.visible_px / r.full_px if r.full_px else 0.0
                at_edge = (r.center_x - r.width / 2 <= 1) or (r.center_x + r.width / 2 >= est.shape[1] - 1)
                if vis < args.min_vis:
                    continue
                H_, W_ = true.shape
                x0 = int(np.clip(math.floor(r.center_x - 0.2 * r.width), 0, W_ - 1))
                x1 = int(np.clip(math.ceil(r.center_x + 0.2 * r.width), 0, W_ - 1))
                y0 = int(np.clip(math.floor(r.center_y - 0.25 * r.height), 0, H_ - 1))
                y1 = int(np.clip(math.ceil(r.center_y + 0.25 * r.height), 0, H_ - 1))
                region = true[y0:y1 + 1, x0:x1 + 1]
                near_frac = float(np.mean((region > 0) & (region <= lo))) if region.size else 0.0
                if near_frac > args.max_near_frac:
                    skipped_near += 1
                    continue
                z_true = box_value(true, r.center_x, r.center_y, r.width, r.height, lo, args.max_depth, args.percentile)
                if not np.isfinite(z_true):
                    continue
                z_est = box_value(est, r.center_x, r.center_y, r.width, r.height, lo, args.max_depth, args.percentile)
                z_rng = box_value(est, r.center_x, r.center_y, r.width, r.height, lo, args.max_depth, args.percentile,
                                  cosmap)
                off_axis = math.degrees(math.acos(float(cosmap[int(np.clip(r.center_y, 0, est.shape[0] - 1)),
                                                               int(np.clip(r.center_x, 0, est.shape[1] - 1))])))
                rows.append({"run": rd.name, "scenario": scen, "frame_index": fi, "actor": r.actor_name,
                             "box_h": r.height, "vis_frac": vis, "at_edge": at_edge, "off_axis_deg": off_axis,
                             "z_true_m": z_true, "z_est_m": z_est, "z_est_from_range_m": z_rng,
                             "z_est_scaled_m": z_est * scale if np.isfinite(scale) else np.nan,
                             "z_est_from_range_scaled_m": z_rng * scale_r if np.isfinite(scale_r) else np.nan})
    print()
    print(f"  {skipped_near} box(es) skipped: person closer than the minimum valid depth (min_valid_depth_m)")
    b = pd.DataFrame(rows)
    b.to_csv(out / "depth_err_boxes.csv", index=False)
    im = pd.DataFrame(img_rows)
    im.to_csv(out / "depth_err_image.csv", index=False)
    if b.empty:
        print("No person boxes with valid depth.")
        return

    b["band"] = pd.cut(b.z_true_m, BANDS, right=False)
    summ = []
    for reading, col in [("as_planar", "z_est_m"), ("from_range", "z_est_from_range_m"),
                         ("as_planar_scaled", "z_est_scaled_m"), ("from_range_scaled", "z_est_from_range_scaled_m")]:
        e = b[col] - b.z_true_m
        b[f"err_{reading}"] = e
        for band, g in b.groupby("band", observed=True):
            ee = (g[col] - g.z_true_m).dropna()
            if ee.empty:
                continue
            rel = ee / g.loc[ee.index, "z_true_m"]
            summ.append({"reading": reading, "band_m": str(band), "boxes": len(ee), "runs": g.run.nunique(),
                         "median_err_m": ee.median(), "median_abs_err_m": ee.abs().median(),
                         "p90_abs_err_m": ee.abs().quantile(0.9), "median_rel_err": rel.median(),
                         "median_abs_rel_err": rel.abs().median(),
                         "missing": int(g[col].isna().sum())})
    s = pd.DataFrame(summ)
    s.to_csv(out / "depth_err_summary.csv", index=False)

    # which reading is right: does the error grow with the angle off the optical axis?
    near = b[b.z_true_m < 20]
    trend = {}
    for reading in ["as_planar", "from_range"]:
        rel = (near[f"err_{reading}"] / near.z_true_m).dropna()
        x = near.loc[rel.index, "off_axis_deg"]
        trend[reading] = float(np.polyfit(x, rel, 1)[0] * 30) if len(rel) > 20 else np.nan   # change over 30 deg

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(f"depth source '{args.name}' vs Unreal depth: {len(b)} person boxes in {b.run.nunique()} runs -> {out}\n")
        print(s.to_string(index=False))
        if len(im):
            print(f"\nWhole image (true depth <= {args.image_max_depth:g} m, {len(im)} frames): AbsRel "
                  f"{im.absrel.median():.3f} (as planar) / {im.absrel_from_range.median():.3f} (from range), "
                  f"delta<1.25 {im.delta1.median():.3f}, best global scale median {im.global_scale.median():.3f} "
                  f"(IQR {im.global_scale.quantile(0.25):.3f}-{im.global_scale.quantile(0.75):.3f})")
        print(f"Relative error change over 30 deg off axis: as planar {trend['as_planar']:+.3f}, "
              f"from range {trend['from_range']:+.3f} (the reading closer to 0 is the right one)")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
        ax = axes[0]
        lim = min(30.0, float(np.nanpercentile(b.z_true_m, 99)) + 1)
        sub = b[b.z_true_m <= lim]
        ax.scatter(sub.z_true_m, sub.z_est_m, s=3, alpha=0.25, color="#2a78d6", lw=0, label="estimate")
        ax.plot([0, lim], [0, lim], color="#7a7a75", lw=1)
        ax.axvline(20, color="#b3b2ad", lw=0.8, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Unreal depth at the person (m)"); ax.set_ylabel(f"{args.name} depth (m)")
        ax.set_title("Estimated against rendered depth", loc="left", fontsize=9)
        ax = axes[1]
        cols = {"as_planar": "#2a78d6", "from_range": "#eb6834", "as_planar_scaled": "#1baf7a",
                "from_range_scaled": "#7a5cc2"}
        for i, (reading, g) in enumerate(s.groupby("reading", sort=False)):
            mids = [float(x.strip("[)").split(",")[0]) + 2.5 + i * 0.4 for x in g.band_m]
            ax.plot(mids, g.median_rel_err, marker="o", ms=4, color=cols[reading], label=reading.replace("_", " "))
        ax.axhline(0, color="#7a7a75", lw=1)
        ax.set_xlabel("Unreal depth band (m)"); ax.set_ylabel("Median relative error")
        ax.set_title("Bias by distance", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "F9_depth_estimate.pdf"); fig.savefig(out / "F9_depth_estimate.png", dpi=200)
        print("  F9_depth_estimate")
    except Exception as exc:  # figure is optional
        print(f"  (figure skipped: {exc})")


if __name__ == "__main__":
    main()
