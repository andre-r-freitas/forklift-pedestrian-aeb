#!/usr/bin/env python3
"""
Detection metrics for the forklift / pedestrian simulation runs (the detection chapter).

Usage:
    python detection_metrics.py <EvalRuns, batch or run folder> [--detector unreal] [--loop open] [--photometric]

<path> is searched recursively; every folder with run_metadata.json is one run (same convention as
evaluate_runs.py). Results go to <path>/analysis_det (or --out).

What it measures, per frame and per ground-truth (GT) person:
    * matching: greedy by confidence, IoU >= 0.5 (--match-iou), one detection per GT. Detections are person class
      only and above the logged confidence threshold (0.25), so AP / MR are "truncated at 0.25".
    * evaluable GT: visible fraction (visible_px / full_px) >= --min-vis AND --min-range <= distance <= --max-range (distance of the
      body center ahead of the fork tips; below 0 = after contact in open loop) AND not cut off
      by the left/right image edge by more than --max-edge-trunc (full_px only counts in-frame pixels, so a person
      leaving the side of the image keeps a high vis_frac; the cut-off is estimated from box width vs height). Other GT is
      "ignore": missing it is not a miss, and a detection on it is neither TP nor FP (standard ignore handling).
    * recall, precision, false positives per frame, AP@0.5 (COCO 101-point), log-average miss rate MR^-2
      (Caltech pedestrian benchmark, FPPI 1e-2 .. 1e0), mean IoU of matches.
    * recall in bins: distance (fork tips to body center), visible fraction, box height, crouched, and with
      --photometric the brightness and contrast of the person against its surroundings (from the saved RGB frames).
    * depth error by distance band (detector 'unreal' only: Unreal's estimate vs the true distance).
    * frames: by default only while the forklift moves (--include-stopped to keep standstill frames, which repeat
      the same view after a stop and would inflate the counts).

Detectors (--detector):
    unreal   detections.csv as logged by Unreal (also checks this script's matching against Unreal's own)
    <name>   detections_<name>.csv in each run folder, columns frame_index, confidence, center_x, center_y, width,
             height (pixels, 640x640, person class only), e.g. written by a Python YOLO11n/s/m script

Outputs (in --out):
    det_gt.csv            one row per GT person per frame (distance, visibility, box size, photometry, detected, IoU)
    det_detections.csv    one row per detection (TP / FP / ignored, confidence, IoU, position if logged)
    det_summary.csv       per group (ALL, scenario, scenario x speed): recall, precision, FP/frame, AP50, MR-2,
                          with Wilson intervals and optional run-clustered bootstrap intervals (--bootstrap N)
    det_recall_bins.csv   recall per bin and group, with Wilson 95% intervals
    det_depth_error.csv   signed median and mean absolute depth error per distance band (unreal only)
    det_runs.csv          per run: frames used, GT, TP, FP, FN, validity flags

Note on intervals: frames of one run are strongly correlated (the same person a fraction of a second apart), so
per-frame Wilson intervals are too narrow. Use the run-clustered bootstrap (--bootstrap 2000) for numbers you report.

Requires: pandas, numpy; pillow only with --photometric. evaluate_runs.py in the same folder.
"""

from __future__ import annotations

import argparse
import json
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_runs import EXPECTED_CONFIG, EXPECTED_JITTER, MOVING_SPEED_MPS, check_config, read_csv, variant_id

DIST_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 60, np.inf]
VIS_BINS = [0.0, 0.25, 0.5, 0.75, 0.999, 1.001]
VIS_LABELS = ["<25%", "25-50%", "50-75%", "75-99%", "100%"]
HEIGHT_BINS = [0, 10, 20, 40, 80, 160, np.inf]
LUMA_BINS = [0, 40, 80, 120, 160, 256]
CONTRAST_BINS = [0, 0.1, 0.25, 0.5, 1.0, np.inf]
DEPTH_BANDS = [0, 5, 10, 15, 20, 30, np.inf]


# ------------------------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------------------------


def _fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def progress(i: int, n: int, label: str, t0: float, newline: bool = False) -> None:
    """One status line: [ i/n] pct  label  elapsed / remaining. Overwrites itself unless newline=True."""
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0.0
    w = len(str(n))
    line = f"  [{i:>{w}}/{n}] {100 * i / max(1, n):3.0f}%  {label[:48]:<48}  {_fmt_s(el)} elapsed, ~{_fmt_s(eta)} left"
    if newline:
        print(line, flush=True)
    else:
        print("\r" + line, end="\n" if i >= n else "", flush=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--detector", default="unreal")
    p.add_argument("--loop", choices=["open", "closed", "all"], default="open",
                   help="Which runs to use: open loop (braking off, default), closed loop, or all")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--min-vis", type=float, default=0.5, help="Evaluable GT: visible fraction at least this")
    p.add_argument("--max-range", type=float, default=30.0, help="Evaluable GT: at most this far ahead (m)")
    p.add_argument("--min-range", type=float, default=0.0,
                   help="Evaluable GT: at least this far ahead of the fork tips (m). Below 0 the person's center is level "
                        "with or behind the tips, i.e. after contact in open loop: a detection there cannot change the outcome")
    p.add_argument("--min-score", type=float, default=0.25,
                   help="Ignore detections below this (raise it to study a stricter threshold)")
    p.add_argument("--max-edge-trunc", type=float, default=0.5,
                   help="Ignore GT cut off by the left/right image edge by more than this fraction of its width, "
                        "estimated from the box height and a nominal person aspect (w/h) of --person-aspect. "
                        "Needed because full_px only counts pixels inside the frame, so vis_frac misses edge cut-off.")
    p.add_argument("--person-aspect", type=float, default=0.25,
                   help="Narrowest width/height of a whole upright person box (side view, legs together; ~0.22-0.29 in the logs)")
    p.add_argument("--ignore-match", choices=["ioa", "iou"], default="ioa",
                   help="How detections are absorbed by ignored people: 'ioa' = intersection over detection area >= "
                        "--match-iou (Caltech; default), 'iou' = same IoU rule as for evaluable people")
    p.add_argument("--include-stopped", action="store_true", help="Keep frames while the forklift stands still")
    p.add_argument("--include-invalid", action="store_true", help="Keep runs that evaluate_runs would flag")
    p.add_argument("--no-config-check", action="store_true")
    p.add_argument("--photometric", action="store_true",
                   help="Brightness / contrast of every GT box from the saved RGB frames (slower)")
    p.add_argument("--ring", type=float, default=0.5,
                   help="Photometry: ring around the box, as a fraction of box width/height on each side")
    p.add_argument("--bootstrap", type=int, default=0, help="Run-clustered bootstrap resamples (0 = off)")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of a (the detection): how much of each detection lies inside each GT box."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    area = (a[:, 2] * a[:, 3])[:, None]
    return np.where(area > 0, iw * ih / area, 0.0)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (n, 4) as cx, cy, w, h -> (len(a), len(b)) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax1, ay1 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx1, by1 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ap_coco(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: 101-point interpolated precision over recall."""
    if n_gt == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]       # monotone envelope
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, pts, side="left")
    vals = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(vals.mean())


def log_avg_miss_rate(scores: np.ndarray, tp: np.ndarray, n_gt: int, n_frames: int) -> float:
    """Caltech MR^-2: geometric mean of the miss rate at 9 FPPI points log-spaced over [1e-2, 1e0].
    Where the curve has no point at or below a reference FPPI, the miss rate counts as 1 (conservative)."""
    if n_gt == 0 or n_frames == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    tpc = np.cumsum(tp[order]) if len(order) else np.array([0])
    fpc = np.cumsum(~tp[order]) if len(order) else np.array([0])
    mr = 1.0 - tpc / n_gt
    fppi = fpc / n_frames
    refs = np.logspace(-2, 0, 9)
    vals = []
    for ref in refs:
        ok = np.where(fppi <= ref)[0]
        vals.append(mr[ok[-1]] if len(ok) else 1.0)
    vals = np.maximum(np.array(vals, float), 1e-10)
    return float(np.exp(np.mean(np.log(vals))))


def yaw_axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    y = math.radians(yaw_deg)
    return np.array([math.cos(y), math.sin(y)]), np.array([-math.sin(y), math.cos(y)])


# ------------------------------------------------------------------------------------------------
# Per run
# ------------------------------------------------------------------------------------------------

def run_flags(meta: dict, fm: pd.DataFrame | None, args) -> list[str]:
    flags = [] if args.no_config_check else check_config(meta, EXPECTED_CONFIG)
    if not args.no_config_check and meta.get("randomised"):
        flags += check_config(meta, EXPECTED_JITTER)
    stale = meta.get("stale_frames")
    if stale is None and fm is not None and "stale_frame" in fm:
        stale = int(fm.stale_frame.sum())
    if stale:
        flags.append(f"stale_frames={stale}")
    if meta.get("end_reason") in ("Aborted", "Stopped", "EndPlay", "Superseded"):
        flags.append(f"end_reason={meta.get('end_reason')}")
    if meta.get("speed_wait_timed_out"):
        flags.append("speed_wait_timed_out")
    return flags


def load_detections(run_dir: Path, detector: str) -> pd.DataFrame:
    name = "detections.csv" if detector == "unreal" else f"detections_{detector}.csv"
    det = read_csv(run_dir / name)
    if det is None:
        return pd.DataFrame(columns=["frame_index", "confidence", "center_x", "center_y", "width", "height"])
    if "class_id" in det:
        det = det[det.class_id == 0]
    return det.reset_index(drop=True)


def find_frame(run_dir: Path, frame: int) -> Path | None:
    for p in (run_dir / "frames" / f"frame_{frame:05d}.png", run_dir / f"frame_{frame:05d}.png"):
        if p.exists():
            return p
    return None


def photometry(img: np.ndarray, cx, cy, w, h, ring: float) -> tuple[float, float]:
    """Mean luma inside the box, and |inside - ring| / ring luma (Weber contrast against the surroundings)."""
    H, W = img.shape
    x0, x1 = int(max(0, math.floor(cx - w / 2))), int(min(W, math.ceil(cx + w / 2)))
    y0, y1 = int(max(0, math.floor(cy - h / 2))), int(min(H, math.ceil(cy + h / 2)))
    if x1 <= x0 or y1 <= y0:
        return (np.nan, np.nan)
    inside = img[y0:y1, x0:x1]
    rx, ry = max(2, int(round(w * ring))), max(2, int(round(h * ring)))
    X0, X1, Y0, Y1 = max(0, x0 - rx), min(W, x1 + rx), max(0, y0 - ry), min(H, y1 + ry)
    outer = img[Y0:Y1, X0:X1].astype(float)
    n_out = outer.size - inside.size
    if n_out <= 0:
        return (float(inside.mean()), np.nan)
    ring_mean = (outer.sum() - inside.sum()) / n_out
    lin = float(inside.mean())
    return (lin, abs(lin - ring_mean) / max(ring_mean, 1.0))


def evaluate_run(run_dir: Path, root: Path, args) -> tuple[list[dict], list[dict], dict | None]:
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    braking = bool(meta.get("safety", {}).get("enabled", False))
    if (args.loop == "open" and braking) or (args.loop == "closed" and not braking):
        return [], [], None

    fm, gt, ws = read_csv(run_dir / "frame_metrics.csv"), read_csv(run_dir / "ground_truth.csv"), read_csv(run_dir / "world_state.csv")
    run_key = run_dir.relative_to(root).as_posix() if run_dir != root else run_dir.name
    flags = run_flags(meta, fm, args)
    if fm is None or ws is None:
        flags.append("missing_csv")
    if meta.get("full_silhouette_from_g") is False:
        flags.append("no_full_silhouette")          # visible fraction meaningless
    scen = str(meta.get("scenario_id", ""))
    suffix = str(meta.get("variant_suffix", "") or "")
    if suffix and scen.endswith(suffix):                 # batch runs carry the variant in scenario_id
        scen = scen[: -len(suffix)]
    info = {"run_key": run_key, "run_id": meta.get("run_id"), "scenario_id": scen,
            "variant": variant_id(meta.get("run_id", run_dir.name)), "target_kmh": meta.get("forklift_target_kmh"),
            "braking": braking, "seed": meta.get("seed"), "detector": args.detector, "flags": ";".join(flags),
            "valid": not flags}
    if (flags and not args.include_invalid) or fm is None or ws is None:
        return [], [], {**info, "frames_used": 0}

    # ---- per-frame forklift pose, speed, and frames to use ----
    fk = ws[ws.role == "forklift"].set_index("frame_index")
    fwd_speed = {}
    for f, row in fk.iterrows():
        ex, _ = yaw_axes(row.yaw_deg)
        fwd_speed[f] = row.vx_mps * ex[0] + row.vy_mps * ex[1]
    frames = sorted(int(f) for f in fm.frame_index.unique())
    if not args.include_stopped:
        frames = [f for f in frames if fwd_speed.get(f, 0.0) > MOVING_SPEED_MPS]
    frame_set = set(frames)
    front = meta.get("safety", {}).get("front_offset_from_origin_m", 2.58)
    crouched = {p.get("actor_name"): bool(p.get("crouched", False)) for p in meta.get("pedestrians", [])}
    peds = ws[ws.role == "pedestrian"].set_index(["frame_index", "actor_name"])

    det = load_detections(run_dir, args.detector)
    det = det[(det.confidence >= args.min_score) & det.frame_index.isin(frame_set)]
    gt = gt[gt.frame_index.isin(frame_set)] if gt is not None else pd.DataFrame()

    from_img = args.photometric
    img_w = float(meta.get("image_width", 640) or 640)
    gt_rows, det_rows = [], []
    agree = disagree = 0
    for f in frames:
        g = gt[gt.frame_index == f] if len(gt) else gt
        d = det[det.frame_index == f]
        # GT geometry: forward / lateral of the body center from the fork tips
        dist, lat = [], []
        if f in fk.index:
            pose = fk.loc[f]
            ex, ey = yaw_axes(pose.yaw_deg)
            for _, row in g.iterrows():
                key = (f, row.actor_name)
                if key in peds.index:
                    pr = peds.loc[key]
                    v = np.array([pr.x_m - pose.x_m, pr.y_m - pose.y_m])
                    dist.append(float(v @ ex) - front)
                    lat.append(float(v @ ey))
                else:
                    dist.append(np.nan)
                    lat.append(np.nan)
        else:
            dist = lat = [np.nan] * len(g)
        vis = (g.visible_px / g.full_px.replace(0, np.nan)).clip(upper=1.0).to_numpy(float) if len(g) else np.array([])
        dist = np.array(dist, float)
        if len(g):
            x0 = (g.center_x - g.width / 2).to_numpy(float)
            x1 = (g.center_x + g.width / 2).to_numpy(float)
            at_edge = (x0 <= 1.0) | (x1 >= img_w - 1.0)
            width_frac = g.width.to_numpy(float) / np.maximum(1.0, args.person_aspect * g.height.to_numpy(float))
            edge_trunc = at_edge & (width_frac < 1.0 - args.max_edge_trunc)
        else:
            edge_trunc = np.zeros(0, bool)
        evaluable = (vis >= args.min_vis) & (dist <= args.max_range) & (dist >= args.min_range) & ~edge_trunc

        gboxes = g[["center_x", "center_y", "width", "height"]].to_numpy(float) if len(g) else np.zeros((0, 4))
        dboxes = d[["center_x", "center_y", "width", "height"]].to_numpy(float)
        ious = iou_matrix(dboxes, gboxes)
        ioas = ioa_matrix(dboxes, gboxes) if args.ignore_match == "ioa" else ious
        order = np.argsort(-d.confidence.to_numpy(float), kind="mergesort")
        g_match = np.full(len(g), -1)
        d_state = np.full(len(d), "fp", dtype=object)
        d_iou = np.zeros(len(d))
        # Evaluable GT first: one-to-one at IoU >= match_iou. Then ignore regions (Caltech convention): a remaining
        # detection is ignored if it lies mostly inside an ignored person's box (intersection / detection area >=
        # match_iou), and one ignored person may absorb several detections. This keeps a correct box on the visible
        # part of a mostly hidden person (e.g. a head above a rack) from counting as a false positive.
        for di in order:
            cand = [(ious[di, gi], gi) for gi in range(len(g))
                    if g_match[gi] < 0 and evaluable[gi] and ious[di, gi] >= args.match_iou]
            if cand:
                best_iou, gi = max(cand)
                g_match[gi] = di
                d_state[di] = "tp"
                d_iou[di] = best_iou
        for di in order:
            if d_state[di] != "fp":
                continue
            cand = [(ioas[di, gi], gi) for gi in range(len(g)) if not evaluable[gi] and ioas[di, gi] >= args.match_iou]
            if cand:
                best, gi = max(cand)
                if g_match[gi] < 0 and ious[di, gi] >= args.match_iou:   # "detected" keeps the IoU definition
                    g_match[gi] = di
                d_state[di] = "ignored"
                d_iou[di] = ious[di, gi]

        img = None
        if from_img and len(g):
            fp_path = find_frame(run_dir, f)
            if fp_path is not None:
                from PIL import Image
                rgb = np.asarray(Image.open(fp_path).convert("RGB")).astype(float)
                img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        for gi, (_, row) in enumerate(g.iterrows()):
            luma = contrast = np.nan
            if img is not None:
                luma, contrast = photometry(img, row.center_x, row.center_y, row.width, row.height, args.ring)
            di = g_match[gi]
            gt_rows.append({**{k: info[k] for k in ("run_key", "scenario_id", "variant", "target_kmh", "braking", "detector")},
                            "frame_index": f, "actor_name": row.actor_name, "stencil_id": row.stencil_id,
                            "crouched": crouched.get(row.actor_name, False),
                            "dist_m": dist[gi], "lat_m": lat[gi] if gi < len(lat) else np.nan,
                            "vis_frac": vis[gi], "edge_trunc": bool(edge_trunc[gi]), "box_w": row.width, "box_h": row.height,
                            "evaluable": bool(evaluable[gi]), "detected": di >= 0,
                            "confidence": float(d.confidence.iloc[di]) if di >= 0 else np.nan,
                            "iou": float(d_iou[di]) if di >= 0 else np.nan,
                            "luma": luma, "contrast": contrast})
            if args.detector == "unreal" and "detected" in row:
                agree += int(bool(row.detected) == (di >= 0))
                disagree += int(bool(row.detected) != (di >= 0))

        for k, di in enumerate(range(len(d))):
            drow = d.iloc[di]
            matched_gi = int(np.where(g_match == di)[0][0]) if (g_match == di).any() else -1
            det_rows.append({**{k2: info[k2] for k2 in ("run_key", "scenario_id", "variant", "target_kmh", "braking", "detector")},
                             "frame_index": f, "confidence": float(drow.confidence), "state": d_state[di],
                             "iou": float(d_iou[di]) if matched_gi >= 0 else np.nan,
                             "matched_actor": g.actor_name.iloc[matched_gi] if matched_gi >= 0 else "",
                             "gt_dist_m": dist[matched_gi] if matched_gi >= 0 else np.nan,
                             "est_fwd_m": float(drow.fwd_m) if "fwd_m" in d else np.nan,
                             "est_lat_m": float(drow.lat_m) if "lat_m" in d else np.nan,
                             "in_zone": int(drow.in_zone) if "in_zone" in d else np.nan,
                             "box_w": float(drow.width), "box_h": float(drow.height)})

    n_gt = sum(r["evaluable"] for r in gt_rows)
    tp = sum(r["state"] == "tp" for r in det_rows)
    fp = sum(r["state"] == "fp" for r in det_rows)
    info.update({"frames_used": len(frames), "gt_evaluable": n_gt, "tp": tp, "fp": fp, "fn": n_gt - tp,
                 "match_agreement_with_unreal": (agree / (agree + disagree)) if (agree + disagree) else np.nan})
    return gt_rows, det_rows, info


# ------------------------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------------------------

def group_metrics(gt: pd.DataFrame, det: pd.DataFrame, n_frames: int) -> dict:
    ge = gt[gt.evaluable]
    dd = det[det.state != "ignored"]
    n_gt, tp, fp = len(ge), int((dd.state == "tp").sum()), int((dd.state == "fp").sum())
    scores, is_tp = dd.confidence.to_numpy(float), (dd.state == "tp").to_numpy(bool)
    rec_lo, rec_hi = wilson(tp, n_gt)
    pre_lo, pre_hi = wilson(tp, tp + fp)
    near = ge[ge.dist_m <= 10]
    return {"frames": n_frames, "gt_evaluable": n_gt, "tp": tp, "fp": fp, "fn": n_gt - tp,
            "recall": tp / n_gt if n_gt else np.nan, "recall_wilson_lo": rec_lo, "recall_wilson_hi": rec_hi,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan, "precision_wilson_lo": pre_lo,
            "precision_wilson_hi": pre_hi, "fp_per_frame": fp / n_frames if n_frames else np.nan,
            "ap50_trunc": ap_coco(scores, is_tp, n_gt), "mr2_trunc": log_avg_miss_rate(scores, is_tp, n_gt, n_frames),
            "mean_iou_tp": float(dd[dd.state == "tp"].iou.mean()) if tp else np.nan,
            "recall_within_10m": float(near.detected.mean()) if len(near) else np.nan,
            "fp_in_zone": int(((dd.state == "fp") & (dd.in_zone == 1)).sum()) if "in_zone" in dd else np.nan}


def bootstrap(gt: pd.DataFrame, det: pd.DataFrame, frames_by_run: dict, B: int, rng) -> dict:
    runs = np.array(sorted(frames_by_run))
    if len(runs) < 2 or B <= 0:
        return {}
    gt_by = {k: v for k, v in gt[gt.evaluable].groupby("run_key")}
    det_by = {k: v for k, v in det[det.state != "ignored"].groupby("run_key")}
    rec, pre, fpf = [], [], []
    for _ in range(B):
        pick = rng.choice(runs, size=len(runs), replace=True)
        n_gt = sum(len(gt_by.get(r, ())) for r in pick)
        tp = sum(int((det_by[r].state == "tp").sum()) for r in pick if r in det_by)
        fp = sum(int((det_by[r].state == "fp").sum()) for r in pick if r in det_by)
        nf = sum(frames_by_run[r] for r in pick)
        rec.append(tp / n_gt if n_gt else np.nan)
        pre.append(tp / (tp + fp) if (tp + fp) else np.nan)
        fpf.append(fp / nf if nf else np.nan)
    q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
    (r0, r1), (p0, p1), (f0, f1) = q(rec), q(pre), q(fpf)
    return {"recall_boot_lo": r0, "recall_boot_hi": r1, "precision_boot_lo": p0, "precision_boot_hi": p1,
            "fp_per_frame_boot_lo": f0, "fp_per_frame_boot_hi": f1}


def recall_bins(gt: pd.DataFrame, group: str, args) -> list[dict]:
    """Each dimension is binned over the GT that passes the OTHER evaluability criteria."""
    rows = []
    vis_ok = (gt.vis_frac >= args.min_vis) & ~gt.edge_trunc
    dist_ok = (gt.dist_m <= args.max_range) & (gt.dist_m >= args.min_range) & ~gt.edge_trunc
    dims = [("distance_m", gt[vis_ok], "dist_m", DIST_BINS, None),
            ("visible_fraction", gt[dist_ok], "vis_frac", VIS_BINS, VIS_LABELS),
            ("box_height_px", gt[gt.evaluable], "box_h", HEIGHT_BINS, None)]
    if gt.luma.notna().any():
        dims += [("luma_0_255", gt[gt.evaluable], "luma", LUMA_BINS, None),
                 ("contrast_weber", gt[gt.evaluable], "contrast", CONTRAST_BINS, None)]
    for name, sub, col, edges, labels in dims:
        if not len(sub):
            continue
        cats = pd.cut(sub[col], edges, right=False, labels=labels)
        for b, s in sub.groupby(cats, observed=True):
            k, n = int(s.detected.sum()), len(s)
            lo, hi = wilson(k, n)
            rows.append({"group": group, "dimension": name, "bin": str(b), "n": n, "detected": k,
                         "recall": k / n, "wilson_lo": lo, "wilson_hi": hi,
                         "runs": s.run_key.nunique(), "mean_conf_detected": s.confidence.mean()})
    sub = gt[gt.evaluable]
    for flag, s in sub.groupby("crouched"):
        k, n = int(s.detected.sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({"group": group, "dimension": "crouched", "bin": str(bool(flag)), "n": n, "detected": k,
                     "recall": k / n, "wilson_lo": lo, "wilson_hi": hi, "runs": s.run_key.nunique(),
                     "mean_conf_detected": s.confidence.mean()})
    return rows


def depth_error(det: pd.DataFrame) -> pd.DataFrame:
    tp = det[(det.state == "tp") & det.est_fwd_m.notna() & det.gt_dist_m.notna()].copy()
    if not len(tp):
        return pd.DataFrame()
    tp["err_m"] = tp.est_fwd_m - tp.gt_dist_m          # estimate (nearest surface) - body center
    tp["band"] = pd.cut(tp.gt_dist_m, DEPTH_BANDS, right=False)
    out = []
    for b, s in tp.groupby("band", observed=True):
        out.append({"distance_band_m": str(b), "n": len(s), "median_err_m": s.err_m.median(),
                    "mae_m": s.err_m.abs().mean(), "p05_err_m": s.err_m.quantile(0.05),
                    "p95_err_m": s.err_m.quantile(0.95)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = args.path.resolve()
    out = (args.out or root / "analysis_det").resolve()
    run_dirs = sorted(p.parent for p in root.rglob("run_metadata.json") if "analysis" not in p.parts[-2])
    gt_rows, det_rows, infos = [], [], []
    print(f"Matching {len(run_dirs)} run(s){' with photometry' if args.photometric else ''}...", flush=True)
    t0 = time.time()
    for k, rd in enumerate(run_dirs, 1):
        progress(k - 1, len(run_dirs), rd.name, t0)
        g, d, info = evaluate_run(rd, root, args)
        gt_rows += g
        det_rows += d
        if info is not None:
            infos.append(info)
    progress(len(run_dirs), len(run_dirs), "done", t0)
    runs = pd.DataFrame(infos)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "det_runs.csv", index=False)
    if not gt_rows:
        print(f"No usable frames ({len(runs)} run(s) considered, loop={args.loop}). See det_runs.csv for flags.")
        return
    gt = pd.DataFrame(gt_rows)
    det = pd.DataFrame(det_rows) if det_rows else pd.DataFrame(columns=["run_key", "state", "confidence", "iou",
                                                                         "in_zone", "est_fwd_m", "gt_dist_m"])
    gt.to_csv(out / "det_gt.csv", index=False)
    det.to_csv(out / "det_detections.csv", index=False)
    used = runs[runs.frames_used > 0]
    frames_by_run = dict(zip(used.run_key, used.frames_used))

    rng = np.random.default_rng(args.seed)
    groups = [("ALL", used.run_key.tolist())]
    groups += [(f"{s}", list(r.run_key)) for s, r in used.groupby("scenario_id")]
    groups += [(f"{s}@{k:g}kmh", list(r.run_key)) for (s, k), r in used.groupby(["scenario_id", "target_kmh"])]
    summary, bins = [], []
    print(f"Metrics for {len(groups)} group(s){f', {args.bootstrap} bootstrap resamples each' if args.bootstrap else ''}...", flush=True)
    t0 = time.time()
    for k, (name, keys) in enumerate(groups, 1):
        progress(k - 1, len(groups), name, t0)
        g, d = gt[gt.run_key.isin(keys)], det[det.run_key.isin(keys)]
        fbr = {k: frames_by_run[k] for k in keys}
        row = {"group": name, "runs": len(keys), **group_metrics(g, d, sum(fbr.values()))}
        row.update(bootstrap(g, d, fbr, args.bootstrap, rng))
        summary.append(row)
        bins += recall_bins(g, name, args)
    progress(len(groups), len(groups), "done", t0)
    print()
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "det_summary.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "det_recall_bins.csv", index=False)
    de = depth_error(det)
    if len(de):
        de.to_csv(out / "det_depth_error.csv", index=False)

    # ---- console ----
    print(f"detector={args.detector} loop={args.loop} match IoU>={args.match_iou} evaluable: vis>={args.min_vis}, "
          f"edge cut-off<={args.max_edge_trunc:.0%}, {args.min_range:g}<=dist<={args.max_range} m, score>={args.min_score}, {'all frames' if args.include_stopped else 'moving frames only'}")
    print(f"{len(used)} run(s) used, {len(runs) - len(used)} skipped (flags in det_runs.csv) -> {out}\n")
    cols = ["group", "runs", "frames", "gt_evaluable", "recall", "precision", "fp_per_frame", "ap50_trunc",
            "mr2_trunc", "mean_iou_tp", "recall_within_10m"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    if args.detector == "unreal" and "match_agreement_with_unreal" in used:
        a = used.match_agreement_with_unreal.dropna()
        if len(a):
            print(f"\nMatching check vs Unreal's 'detected' column: {a.mean():.4f} agreement "
                  f"(below ~0.99 means the two matchers differ; differences come from ignore regions or ties)")


if __name__ == "__main__":
    main()
