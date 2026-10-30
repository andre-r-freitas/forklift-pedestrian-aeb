#!/usr/bin/env python3
"""
Evaluate forklift / pedestrian simulation runs written by the UE5 EvalRunLogger.

Usage:
    python evaluate_runs.py <path> [options]

<path> can be a single run folder, a batch folder (Saved/EvalRuns/<BatchId>) or the whole EvalRuns folder.
Every folder containing run_metadata.json is treated as one run.

Outputs (written to --out, default: <path>/analysis):
    pedestrian_results.csv  one row per (run, pedestrian): clearance, contact, corridor entry, braking, detection
    run_results.csv         one row per run (worst pedestrian = smallest clearance)
    variant_summary.csv     runs grouped by scenario variant (run id without _repNN)

Geometry (top view, forklift frame, meters; x forward from the actor origin, y lateral):
    body  : x in [-rear_offset, body_front],  |y| <= body_width / 2
    forks : x in [body_front, front_offset],   |y| <= fork_width / 2
    pedestrian = circle of radius ped_radius around the actor location (capsule center)
    clearance  = distance from the circle to the footprint (negative = overlap = contact)

world_state.csv is logged once per perception frame (every 0.2 s), i.e. ~0.56 m apart at 10 km/h. Positions are
linearly interpolated to --interp-hz before computing clearance so contacts between samples are not missed.
(Exact at constant speed; during braking the error is a few cm.)

Requires: pandas, numpy
"""

from __future__ import annotations

import argparse
import json
import time
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------

@dataclass
class Geometry:
    front_offset: float      # origin -> fork tips (m); taken from run metadata when available
    body_front: float        # origin -> front of the body / fork heel (m)
    rear_offset: float       # origin -> rear of the body (m)
    body_width: float
    fork_width: float        # outer width across both forks
    ped_radius: float


# Frozen evaluation configuration: every run must have been recorded with these settings, otherwise it is
# flagged invalid ("config:<key>=<value>"). Experimental variables (reaction latency, model, forklift speed,
# timing shift) are deliberately NOT listed. Override with --expect-json <file> (same structure).
MAX_CAPTURE_GAP_S = 0.25  # capture gaps above this mean the editor dropped frames
MOVING_SPEED_MPS = 0.14   # 0.5 km/h: below this the forklift counts as stopped for contact

EXPECTED_CONFIG = {
    "synchronous_inference": True,
    "forced_scene_capture": True,
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.70,
    "match_iou_threshold": 0.50,
    "inference_interval_s": 0.20,
    "min_valid_depth_m": 1.80,
    "safety": {
        "corridor_half_width_m": 1.50,
        "front_offset_from_origin_m": 2.58,
        "safety_margin_m": 1.00,
        "assumed_deceleration_mps2": 5.97,
        "emergency_brake_input": 1.00,
        "brake_dead_time_s": 0.10,
        "include_detection_delay": True,
        "perception_period_s": 0.20,
        "confirm_frames": 2,
        "window_frames": 3,
        "min_confidence": 0.25,
    },
}

# Checked only for randomised (batch) runs. Written to the metadata from v3.8 on.
EXPECTED_JITTER = {
    "speed_jitter_fraction": 0.10,
    "start_jitter_s": 0.50,
}



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

def check_config(meta: dict, expected: dict, prefix: str = "") -> list[str]:
    """Returns one flag per setting that is missing or differs from the expected value."""
    problems = []
    for key, want in expected.items():
        name = f"{prefix}{key}"
        if key not in meta:
            problems.append(f"config_missing:{name}")
            continue
        have = meta[key]
        if isinstance(want, dict):
            problems += check_config(have if isinstance(have, dict) else {}, want, name + ".")
        elif isinstance(want, bool) or isinstance(have, bool):
            if bool(have) != want:
                problems.append(f"config:{name}={have}")
        elif isinstance(want, (int, float)):
            try:
                if not math.isclose(float(have), float(want), rel_tol=1e-3, abs_tol=1e-3):
                    problems.append(f"config:{name}={have}")
            except (TypeError, ValueError):
                problems.append(f"config:{name}={have}")
        elif have != want:
            problems.append(f"config:{name}={have}")
    return problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--front-offset", type=float, default=None,
                   help="Origin to fork tips (m). Default: safety.front_offset_from_origin_m from metadata, else 2.58")
    p.add_argument("--body-front", type=float, default=1.23, help="Origin to fork heel / body front (m)")
    p.add_argument("--rear-offset", type=float, default=1.23, help="Origin to rear of the body (m)")
    p.add_argument("--body-width", type=float, default=1.73, help="Forklift body width (m), measured from the mesh")
    p.add_argument("--fork-width", type=float, default=1.19, help="Width of the fork section incl. carriage/mast (m), measured from the mesh")
    p.add_argument("--ped-radius", type=float, default=0.42, help="Pedestrian capsule radius (m), BP_WorkerCharacter")
    p.add_argument("--corridor-half-width", type=float, default=None,
                   help="Default: safety.corridor_half_width_m from metadata, else 1.0")
    p.add_argument("--interp-hz", type=float, default=200.0)
    p.add_argument("--expect-json", type=Path, default=None,
                   help="JSON file with the expected configuration (default: EXPECTED_CONFIG in this script)")
    p.add_argument("--no-config-check", action="store_true", help="Skip the configuration check")
    p.add_argument("--near-range", type=float, default=10.0,
                   help="Recall is also reported for ground-truth frames with the pedestrian ahead within this range (m)")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def variant_id(run_id: str) -> str:
    return re.sub(r"_rep\d+$", "", run_id)


def footprint_distance(lx: np.ndarray, ly: np.ndarray, g: Geometry) -> np.ndarray:
    """Signed distance from points (forklift frame) to the footprint (body + forks); negative inside."""

    def rect(xmin, xmax, half_w):
        dx = np.maximum.reduce([xmin - lx, np.zeros_like(lx), lx - xmax])
        dy = np.maximum(np.abs(ly) - half_w, 0.0)
        outside = np.hypot(dx, dy)
        inside_depth = np.minimum.reduce([lx - xmin, xmax - lx, half_w - np.abs(ly)])
        return np.where(outside > 0.0, outside, -inside_depth)

    body = rect(-g.rear_offset, g.body_front, g.body_width / 2.0)
    forks = rect(g.body_front, g.front_offset, g.fork_width / 2.0)
    return np.minimum(body, forks)


def interp_track(t: np.ndarray, src_t: np.ndarray, src: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    return {c: np.interp(t, src_t, src[c].to_numpy(dtype=float)) for c in cols}


def first_true(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def at_time(t: np.ndarray, values: np.ndarray, when: float) -> float:
    if when is None or not np.isfinite(when) or when < t[0] or when > t[-1]:
        return float("nan")
    return float(np.interp(when, t, values))


def r(x, n=3):
    try:
        return round(float(x), n) if x is not None and np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------
# One run
# ------------------------------------------------------------------------------------------------

def evaluate_run(run_dir: Path, root: Path, args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    safety = meta.get("safety") or {}
    run_id = meta.get("run_id", run_dir.name)

    ws = read_csv(run_dir / "world_state.csv")
    if ws is None:
        return [], [f"{run_id}: no world_state.csv"]

    g = Geometry(
        front_offset=args.front_offset if args.front_offset is not None
        else float(safety.get("front_offset_from_origin_m", 2.58)),
        body_front=args.body_front, rear_offset=args.rear_offset,
        body_width=args.body_width, fork_width=args.fork_width, ped_radius=args.ped_radius)
    corridor = args.corridor_half_width if args.corridor_half_width is not None \
        else float(safety.get("corridor_half_width_m", 1.0))

    # ---- forklift track, interpolated ----
    fk = ws[ws.role == "forklift"].sort_values("timestamp_s").drop_duplicates("timestamp_s")
    if len(fk) < 2:
        return [], [f"{run_id}: forklift track too short"]
    fk = fk.assign(yaw_unwrapped=np.degrees(np.unwrap(np.radians(fk.yaw_deg.to_numpy(dtype=float)))))
    t0, t1 = float(fk.timestamp_s.iloc[0]), float(fk.timestamp_s.iloc[-1])
    t = np.arange(t0, t1 + 1e-9, 1.0 / args.interp_hz)
    F = interp_track(t, fk.timestamp_s.to_numpy(float), fk, ["x_m", "y_m", "vx_mps", "vy_mps", "yaw_unwrapped"])
    yaw = np.radians(F["yaw_unwrapped"])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    fk_fwd_speed = F["vx_mps"] * cos_y + F["vy_mps"] * sin_y

    # ---- which pedestrians belong to this run ----
    ped_meta = {p["actor_name"]: p for p in meta.get("pedestrians", [])}
    peds_all = ws[ws.role == "pedestrian"]
    names = list(ped_meta) if ped_meta else sorted(peds_all.actor_name.unique())
    if not ped_meta:
        warnings.append(f"{run_id}: metadata lists no pedestrians; evaluating every logged pedestrian")

    # ---- optional files ----
    fm = read_csv(run_dir / "frame_metrics.csv")
    gt = read_csv(run_dir / "ground_truth.csv")
    det = read_csv(run_dir / "detections.csv")

    stale_frames = meta.get("stale_frames")
    if stale_frames is None and fm is not None and "stale_frame" in fm:
        stale_frames = int(fm.stale_frame.sum())

    run_start = float(meta.get("run_start_s", t0))
    rows = []

    for name in names:
        pm = ped_meta.get(name, {})
        p = peds_all[peds_all.actor_name == name].sort_values("timestamp_s").drop_duplicates("timestamp_s")
        if len(p) < 2:
            warnings.append(f"{run_id}: pedestrian {name} not in world_state")
            continue
        stencil_id = int(p.stencil_id.iloc[0])
        P = interp_track(t, p.timestamp_s.to_numpy(float), p, ["x_m", "y_m", "vx_mps", "vy_mps"])

        # pedestrian in the forklift frame
        dx, dy = P["x_m"] - F["x_m"], P["y_m"] - F["y_m"]
        lx = dx * cos_y + dy * sin_y              # forward of the actor origin
        ly = -dx * sin_y + dy * cos_y             # lateral
        fwd_from_tips = lx - g.front_offset
        clearance = footprint_distance(lx, ly, g) - g.ped_radius

        # Contact only counts while the forklift is moving: a pedestrian who keeps walking into a forklift that has
        # already stopped (S9 oncoming, S10 after catching up) is not a failure of the system.
        moving = fk_fwd_speed > MOVING_SPEED_MPS
        clr_moving = np.where(moving, clearance, np.inf)
        i_min = int(np.argmin(clr_moving)) if moving.any() else int(np.argmin(clearance))
        i_contact = first_true((clearance < 0.0) & moving)
        i_contact_stopped = first_true((clearance < 0.0) & ~moving)

        # corridor entry: pedestrian center inside the corridor and ahead of the fork tips, forklift moving
        in_corridor = (np.abs(ly) <= corridor) & (fwd_from_tips > 0.0)
        i_entry = first_true(in_corridor & (fk_fwd_speed > 0.1))
        entry_t = float(t[i_entry]) if i_entry is not None else float("nan")
        entry_dist = float(fwd_from_tips[i_entry]) if i_entry is not None else float("nan")
        entry_speed = float(fk_fwd_speed[i_entry]) if i_entry is not None else float("nan")

        # ---- safety (closed loop) ----
        confirm_t = float(safety.get("confirm_capture_time_s", -1))
        confirm_t = confirm_t if confirm_t >= 0 else float("nan")
        stopped_t = float(safety.get("stopped_time_s", -1))
        stopped_t = stopped_t if stopped_t >= 0 else float("nan")
        gt_dist_at_confirm = at_time(t, fwd_from_tips, confirm_t)
        speed_at_confirm = at_time(t, fk_fwd_speed, confirm_t)
        clearance_at_stop = at_time(t, clearance, stopped_t)
        gap_at_stop = at_time(t, fwd_from_tips, stopped_t)

        # ---- detection quality for this pedestrian ----
        recall = recall_near = first_det_t = first_det_dist = np.nan
        n_near = n_near_det = 0
        dist_err_med = dist_err_mae = lat_err_med = np.nan
        n_gt = 0
        if gt is not None:
            gp = gt[gt.stencil_id == stencil_id].copy()
            n_gt = len(gp)
            if n_gt:
                gp["fwd"] = np.interp(gp.timestamp_s, t, fwd_from_tips)
                recall = gp.detected.mean()
                near = gp[(gp.fwd > 0) & (gp.fwd <= args.near_range)]
                n_near, n_near_det = len(near), int(near.detected.sum())
                recall_near = near.detected.mean() if len(near) else np.nan
                hits = gp[gp.detected == 1]
                if len(hits):
                    first_det_t = float(hits.timestamp_s.iloc[0])
                    first_det_dist = float(hits.fwd.iloc[0])
        if det is not None and "fwd_m" in det:
            dp = det[det.matched_stencil_id == stencil_id]
            dp = dp[dp.depth_m > 0] if "depth_m" in dp else dp
            if len(dp):
                gt_fwd = np.interp(dp.timestamp_s, t, fwd_from_tips)
                gt_lat = np.interp(dp.timestamp_s, t, ly)
                err = dp.fwd_m.to_numpy(float) - gt_fwd   # estimate (nearest body surface) - GT (body center)
                dist_err_med, dist_err_mae = float(np.median(err)), float(np.mean(np.abs(err)))
                lat_err_med = float(np.median(dp.lat_m.to_numpy(float) - gt_lat))

        # ---- validity checks ----
        flags = [] if args.no_config_check else check_config(meta, args.expected_config)
        if not args.no_config_check and meta.get("randomised"):
            flags += check_config(meta, EXPECTED_JITTER)   # batch runs only; single runs log 0
        if stale_frames is None:
            flags.append("stale_unknown")
        elif stale_frames > 0:
            flags.append(f"stale_frames={stale_frames}")
        if meta.get("end_reason") in ("Aborted", "Stopped", "EndPlay", "Superseded"):
            flags.append(f"end_reason={meta.get('end_reason')}")
        if meta.get("speed_wait_timed_out"):
            flags.append("speed_wait_timed_out")
        if fm is not None and len(fm) > 2:
            gaps = np.diff(np.sort(fm.timestamp_s.to_numpy(float)))
            if gaps.max() > MAX_CAPTURE_GAP_S:          # editor dropped frames (60 fps ticks give 0.200 / 0.217 s)
                flags.append(f"capture_gap_max={gaps.max():.3f}s")
        trig, actual = pm.get("trigger_distance_m", -1), pm.get("forklift_distance_at_start_m", -1)
        if trig is not None and trig > 0 and actual is not None and actual > 0 and abs(trig - actual) > 0.5:
            flags.append(f"start_timing_off={actual - trig:+.2f}m")

        rows.append({
            "run_key": run_dir.relative_to(root).as_posix() if run_dir != root else run_dir.name,
            "run_id": run_id,
            "variant": variant_id(run_id),
            "batch_id": meta.get("batch_id", run_dir.parent.name),
            "scenario_type": meta.get("scenario_type", ""),
            "target_kmh": meta.get("forklift_target_kmh"),
            "timing_shift_s": meta.get("timing_shift_s"),
            "seed": meta.get("seed"),
            "safety_enabled": safety.get("enabled"),
            "end_reason": meta.get("end_reason"),
            "valid": not flags,
            "flags": ";".join(flags),
            "pedestrian": name,
            "stencil_id": stencil_id,
            "start_mode": pm.get("start_mode"),
            "arrival_offset_s": pm.get("arrival_offset_s"),
            # geometry outcome
            "min_clearance_m": r(clearance[i_min]),
            "min_clearance_time_s": r(t[i_min]),
            "contact": i_contact is not None,
            "contact_time_s": r(t[i_contact]) if i_contact is not None else None,
            "contact_speed_kmh": r(fk_fwd_speed[i_contact] * 3.6, 2) if i_contact is not None else None,
            "max_penetration_m": r(-clr_moving.min()) if i_contact is not None else 0.0,
            "contact_with_stopped_forklift": i_contact_stopped is not None and i_contact is None,
            # corridor entry (time available to the system)
            "corridor_entry_time_s": r(entry_t),
            "corridor_entry_dist_m": r(entry_dist),
            "corridor_entry_speed_kmh": r(entry_speed * 3.6, 2),
            "time_available_s": r(entry_dist / entry_speed) if entry_speed > 0.1 else None,
            # closed loop
            "confirm_time_s": r(confirm_t),
            "entry_to_confirm_s": r(confirm_t - entry_t),
            "gt_dist_at_confirm_m": r(gt_dist_at_confirm),
            "est_dist_at_confirm_m": r(safety.get("hazard_distance_at_confirm_m"))
            if safety.get("hazard_distance_at_confirm_m", -1) >= 0 else None,
            "trigger_dist_at_confirm_m": r(safety.get("trigger_distance_at_confirm_m"))
            if safety.get("trigger_distance_at_confirm_m", -1) >= 0 else None,
            "ttc_at_confirm_s": r(gt_dist_at_confirm / speed_at_confirm) if speed_at_confirm > 0.1 else None,
            "braking_distance_m": r(safety.get("braking_distance_m"))
            if safety.get("braking_distance_m", -1) >= 0 else None,
            "stop_time_s": r(stopped_t),
            "gap_tips_to_centre_at_stop_m": r(gap_at_stop),
            "clearance_at_stop_m": r(clearance_at_stop),
            # detection
            "gt_frames": n_gt,
            "recall": r(recall),
            f"recall_within_{args.near_range:g}m": r(recall_near),
            f"gt_frames_within_{args.near_range:g}m": n_near,
            f"detected_within_{args.near_range:g}m": n_near_det,
            "first_detection_time_s": r(first_det_t),
            "first_detection_dist_m": r(first_det_dist),
            "dist_err_median_m": r(dist_err_med),
            "dist_err_mae_m": r(dist_err_mae),
            "lat_err_median_m": r(lat_err_med),
            "mean_inference_ms": r(fm.inference_ms.mean(), 2) if fm is not None else None,
            "stale_frames": stale_frames,
            "run_start_s": r(run_start),
        })

    return rows, warnings


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.expected_config = json.loads(args.expect_json.read_text(encoding="utf-8")) if args.expect_json else EXPECTED_CONFIG
    root: Path = args.path
    run_dirs = sorted({m.parent for m in root.rglob("run_metadata.json")})
    if not run_dirs:
        raise SystemExit(f"No run_metadata.json found under {root}")

    all_rows, all_warn = [], []
    t0 = time.time()
    for k, d in enumerate(run_dirs, 1):
        progress(k - 1, len(run_dirs), d.name, t0)
        try:
            rows, warn = evaluate_run(d, root, args)
        except Exception as exc:  # keep going; report at the end
            rows, warn = [], [f"{d}: FAILED ({exc})"]
        all_rows += rows
        all_warn += warn
    progress(len(run_dirs), len(run_dirs), "done", t0)

    if not all_rows:
        raise SystemExit("No results. " + " | ".join(all_warn))

    out = args.out or (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    peds = pd.DataFrame(all_rows)
    peds.to_csv(out / "pedestrian_results.csv", index=False)

    # one row per run: the pedestrian with the smallest clearance
    runs = peds.loc[peds.groupby("run_key").min_clearance_m.idxmin()].reset_index(drop=True)
    runs.to_csv(out / "run_results.csv", index=False)

    # per variant (only valid runs counted in the outcome columns)
    nr = f"{args.near_range:g}m"

    def summarise(gdf: pd.DataFrame) -> pd.Series:
        v = gdf[gdf.valid]
        near_gt = v[f"gt_frames_within_{nr}"].sum()
        return pd.Series({
            "runs": len(gdf),
            "valid_runs": len(v),
            "contact_rate": v.contact.mean() if len(v) else np.nan,
            "min_clearance_mean_m": v.min_clearance_m.mean(),
            "min_clearance_min_m": v.min_clearance_m.min(),
            "contact_speed_mean_kmh": v.contact_speed_kmh.mean(),
            "time_available_mean_s": v.time_available_s.mean(),
            "entry_to_confirm_mean_s": v.entry_to_confirm_s.mean(),
            "clearance_at_stop_mean_m": v.clearance_at_stop_m.mean(),
            "recall_mean": v.recall.mean(),
            # near range: mean of per-run recalls (runs weighted equally) and pooled over all near frames
            f"recall_within_{nr}_mean": v[f"recall_within_{nr}"].mean(),
            f"recall_within_{nr}_pooled": v[f"detected_within_{nr}"].sum() / near_gt if near_gt else np.nan,
            f"gt_frames_within_{nr}": near_gt,
            f"runs_without_frames_within_{nr}": int((v[f"gt_frames_within_{nr}"] == 0).sum()),
            "first_detection_dist_mean_m": v.first_detection_dist_m.mean(),
        })

    summary = pd.DataFrame([{"variant": k, **summarise(gdf).to_dict()} for k, gdf in runs.groupby("variant")])
    summary.to_csv(out / "variant_summary.csv", index=False)

    # ---- console ----
    print(f"Geometry: front {args.front_offset or 'from metadata'} m, body front {args.body_front} m, rear "
          f"{args.rear_offset} m, body width {args.body_width} m, fork width {args.fork_width} m, "
          f"pedestrian radius {args.ped_radius} m")
    print(f"{len(run_dirs)} run(s) -> {out}\n")
    cols = ["run_key", "valid", "contact", "min_clearance_m", "contact_speed_kmh", "corridor_entry_dist_m",
            "time_available_s", "gt_dist_at_confirm_m", "clearance_at_stop_m", "recall",
            f"recall_within_{args.near_range:g}m", "flags"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(runs[cols].to_string(index=False))
    if all_warn:
        print("\nWarnings:")
        for w in all_warn:
            print("  -", w)


if __name__ == "__main__":
    main()
