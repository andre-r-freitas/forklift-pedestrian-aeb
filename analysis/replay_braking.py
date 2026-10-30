#!/usr/bin/env python3
"""
Offline replay of the safety controller on open-loop runs (braking off in Unreal).

Pedestrians never react to the forklift, so an open-loop run contains everything needed to ask: "if the controller
had been on, with latency L (and detector D, depth source S), when would it have braked and what would have happened?"
The forklift follows its logged path until the brake time, then stops with the calibrated vehicle model; pedestrians
follow their logged paths. This gives braking results for every scenario and latency without extra Unreal runs.

Usage:
    python replay_braking.py <EvalRuns or batch folder> [--latencies 0.05 0.1 0.2] [--detector unreal] [--depth ideal]
    python replay_braking.py <open-loop folder> --validate <closed-loop folder>

Detector sources (--detector):
    unreal          detections.csv as logged by Unreal (positions from Unreal's ideal depth)
    unreal-recomp   Unreal's boxes, positions recomputed in Python from frames/depth_*.png (checks the Python geometry)
    <name>          detections_<name>.csv in each run folder (e.g. written by a Python YOLO script), columns
                    frame_index, confidence, center_x, center_y, width, height  (pixels, 640x640, person class only)
Depth sources (--depth), for detectors other than 'unreal':
    ideal           frames/depth_NNNNN.png (uint16 mm, planar depth) saved by Unreal
    <name>          depth_<name>/depth_NNNNN.png (uint16 mm, as written by estimate_depth.py) or .npy (float meters),
                    planar depth, e.g. Depth Anything V2. Runs without a depth_<name> folder are skipped.

Controllers (--controllers, any combination; default zone):
    zone            the Unreal controller: position-only corridor + trigger distance, 2 of 3 frames (validated)
    ttc             tracking + prediction: constant-velocity Kalman tracks in world coordinates; brake when waiting one
                    more perception frame would let a predicted person get within the margin before the stop
    hybrid          zone OR ttc (prediction can only add earlier stops, never remove a zone stop)
    See tracking.py for the tracker and the prediction rule. --dump-tracks writes every track to <out>/tracks/.

Outputs (in --out, default <path>/analysis):
    replay_results.csv      one row per (run, detector, depth, controller, latency): brake decision, stop, clearance,
                            outcome, and for ttc/hybrid the trigger reason, predicted gap and tracker velocity error
    replay_paired.csv       min clearance of the same seed/timing across forklift speeds (paired comparison)
    replay_validation.csv   with --validate: replay vs the matching closed-loop Unreal run

Requires: pandas, numpy, opencv-python (only for depth PNGs); evaluate_runs.py in the same folder.
"""

from __future__ import annotations

import argparse
import json
import time
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_runs import MOVING_SPEED_MPS, EXPECTED_CONFIG, Geometry, footprint_distance, read_csv, variant_id
from tracking import Tracker, TrackerParams, predicted_conflict

TICK = 1.0 / 60.0


# ------------------------------------------------------------------------------------------------
# Settings
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
    p.add_argument("--latencies", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--detector", default="unreal")
    p.add_argument("--depth", default="ideal")
    p.add_argument("--depth-is-range", action="store_true",
                   help="The --depth source stores distance along the viewing ray; convert to planar depth "
                        "(x cos of the pixel's angle off the optical axis). Use if depth_error.py says 'from range'")
    p.add_argument("--validate", type=Path, default=None,
                   help="Folder with closed-loop runs; each is compared with the replay of the open-loop run "
                        "with the same scenario variant and seed, at the closed-loop run's latency")
    p.add_argument("--include-closed-loop", action="store_true",
                   help="Also replay runs that had braking on (only meaningful up to their brake time; for testing)")
    # controller (defaults = frozen configuration; a run's metadata is NOT used, so every replay uses the same rule)
    s = EXPECTED_CONFIG["safety"]
    p.add_argument("--corridor-half-width", type=float, default=s["corridor_half_width_m"])
    p.add_argument("--front-offset", type=float, default=s["front_offset_from_origin_m"])
    p.add_argument("--margin", type=float, default=s["safety_margin_m"])
    p.add_argument("--assumed-decel", type=float, default=s["assumed_deceleration_mps2"])
    p.add_argument("--controller-dead-time", type=float, default=s["brake_dead_time_s"])
    p.add_argument("--confirm-frames", type=int, default=s["confirm_frames"])
    p.add_argument("--window-frames", type=int, default=s["window_frames"])
    p.add_argument("--min-confidence", type=float, default=s["min_confidence"])
    p.add_argument("--perception-period", type=float, default=s["perception_period_s"])
    p.add_argument("--no-detection-delay", action="store_true", help="Old trigger formula (v L + v^2/2a + margin)")
    # simulated vehicle (dynamics test), used to compute the stop after the brake command
    p.add_argument("--vehicle-dead-time", type=float, default=0.10)
    p.add_argument("--vehicle-decel", default="5:5.45,10:5.97,15:6.55",
                   help="Steady deceleration per speed (km/h:m/s2), linearly interpolated")
    # geometry (same defaults as evaluate_runs.py)
    p.add_argument("--body-front", type=float, default=1.23)
    p.add_argument("--rear-offset", type=float, default=1.23)
    p.add_argument("--body-width", type=float, default=1.73)
    p.add_argument("--fork-width", type=float, default=1.19)
    p.add_argument("--ped-radius", type=float, default=0.42)
    p.add_argument("--hfov", type=float, default=90.0, help="Camera horizontal FOV (deg)")
    p.add_argument("--max-depth", type=float, default=60.0, help="Depths beyond this are invalid (m)")
    p.add_argument("--min-depth", type=float, default=None,
                   help="Depths nearer than this are ignored (m); default = min_valid_depth_m from run_metadata.json, else 0")
    p.add_argument("--depth-percentile", type=float, default=0.25)
    p.add_argument("--interp-hz", type=float, default=200.0)
    # predictive controllers (tracking.py)
    p.add_argument("--controllers", nargs="+", default=["zone"], choices=["zone", "ttc", "hybrid"])
    p.add_argument("--track-meas-sigma", type=float, default=0.20, help="Observation position noise (m)")
    p.add_argument("--track-accel-sigma", type=float, default=2.0, help="Process noise, white acceleration (m/s2)")
    p.add_argument("--track-init-vel-sigma", type=float, default=1.5, help="Velocity uncertainty of a new track (m/s)")
    p.add_argument("--track-gate", type=float, default=9.21, help="Association gate, squared Mahalanobis (99%%, 2 dof)")
    p.add_argument("--track-min-hits", type=int, default=2, help="Detections before a track is used")
    p.add_argument("--track-max-miss", type=int, default=2, help="Frames a track may coast without a detection")
    p.add_argument("--ttc-dt", type=float, default=0.05, help="Prediction time step (s)")
    p.add_argument("--dump-tracks", action="store_true", help="Write every track to <out>/tracks/<run_id>.csv")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------
# Geometry helpers (Unreal: X forward, Y right, Z up; yaw about Z, pitch about Y; meters)
# ------------------------------------------------------------------------------------------------

def rot_axes(yaw_deg: float, pitch_deg: float):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([-math.sin(y), math.cos(y), 0.0])
    up = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)])
    return fwd, right, up


def to_world(origin, yaw, pitch, local):
    f, r, u = rot_axes(yaw, pitch)
    return origin + local[0] * f + local[1] * r + local[2] * u


def to_local(origin, yaw, pitch, world):
    f, r, u = rot_axes(yaw, pitch)
    d = world - origin
    return np.array([d @ f, d @ r, d @ u])


# ------------------------------------------------------------------------------------------------
# Detections -> observations (forward distance from the fork tips, lateral offset)
# ------------------------------------------------------------------------------------------------

def load_depth(run_dir: Path, frame: int, source: str, max_depth_m: float):
    if source == "ideal":
        import cv2
        path = run_dir / "frames" / f"depth_{frame:05d}.png"
        if not path.exists():
            return None
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32) / 1000.0
    folder = run_dir / f"depth_{source}"
    png, npy = folder / f"depth_{frame:05d}.png", folder / f"depth_{frame:05d}.npy"
    if png.exists():                                  # uint16 mm, same format as Unreal's (estimate_depth.py)
        import cv2
        d = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        d = d.astype(np.float32) / 1000.0
    elif npy.exists():                                # float meters
        d = np.load(npy).astype(np.float32)
    else:
        return None
    d[(d <= 0.01) | (d >= max_depth_m)] = 0.0
    return d


def estimate_observations(dets: pd.DataFrame, ws: pd.DataFrame, run_dir: Path, args, depth_source: str,
                          img_w: int = 640, img_h: int = 640) -> pd.DataFrame:
    """Same as AYOLOInferenceActor::EstimatePerson: percentile depth of the box's central region, pinhole back-projection."""
    cam = ws[ws.role == "camera"].set_index("frame_index")
    fk = ws[ws.role == "forklift"].set_index("frame_index")
    focal = (0.5 * img_w) / math.tan(math.radians(0.5 * args.hfov))
    min_depth = args.min_depth
    if min_depth is None:
        try:
            meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            min_depth = float(meta.get("min_valid_depth_m", 0.0))
        except (OSError, ValueError):
            min_depth = 0.0
    out = []
    cache: dict[int, np.ndarray | None] = {}
    for r in dets.itertuples(index=False):
        f = int(r.frame_index)
        if f not in cache:
            cache[f] = load_depth(run_dir, f, depth_source, args.max_depth)
            if cache[f] is not None and depth_source != "ideal" and getattr(args, "depth_is_range", False):
                H, W = cache[f].shape
                u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
                fpx = (0.5 * W) / math.tan(math.radians(0.5 * args.hfov))
                cache[f] = cache[f] * (fpx / np.sqrt(fpx * fpx + u * u + v * v)).astype(np.float32)
        depth = cache[f]
        fwd = lat = np.nan
        if depth is not None and f in cam.index and f in fk.index:
            h, w = depth.shape[:2]
            sx, sy = w / img_w, h / img_h
            x0 = int(np.clip(math.floor((r.center_x - 0.2 * r.width) * sx), 0, w - 1))
            x1 = int(np.clip(math.ceil((r.center_x + 0.2 * r.width) * sx), 0, w - 1))
            y0 = int(np.clip(math.floor((r.center_y - 0.25 * r.height) * sy), 0, h - 1))
            y1 = int(np.clip(math.ceil((r.center_y + 0.25 * r.height) * sy), 0, h - 1))
            vals = depth[y0:y1 + 1, x0:x1 + 1].ravel()
            vals = np.sort(vals[(vals > max(0.01, min_depth)) & (vals < args.max_depth)])
            if vals.size:
                z = float(vals[int(np.clip(math.floor(args.depth_percentile * (vals.size - 1)), 0, vals.size - 1))])
                right = (r.center_x - 0.5 * img_w) / focal * z
                down = (r.center_y - 0.5 * img_h) / focal * z
                c, k = cam.loc[f], fk.loc[f]
                world = to_world(np.array([c.x_m, c.y_m, c.z_m]), c.yaw_deg, c.pitch_deg, np.array([z, right, -down]))
                local = to_local(np.array([k.x_m, k.y_m, k.z_m]), k.yaw_deg, k.pitch_deg, world)
                fwd, lat = local[0] - args.front_offset, local[1]
        out.append({"frame_index": f, "confidence": r.confidence, "fwd_m": fwd, "lat_m": lat})
    return pd.DataFrame(out, columns=["frame_index", "confidence", "fwd_m", "lat_m"])


def observations_for_run(run_dir: Path, ws: pd.DataFrame, args) -> pd.DataFrame | None:
    if args.detector == "unreal":
        det = read_csv(run_dir / "detections.csv")
        if det is None:
            return pd.DataFrame(columns=["frame_index", "confidence", "fwd_m", "lat_m"])
        if "fwd_m" not in det:
            return None
        d = det[["frame_index", "confidence", "fwd_m", "lat_m"]].copy()
        if "depth_m" in det:
            d.loc[det.depth_m <= 0, ["fwd_m", "lat_m"]] = np.nan
        return d
    if args.detector == "unreal-recomp":
        det = read_csv(run_dir / "detections.csv")
        if det is None:
            return pd.DataFrame(columns=["frame_index", "confidence", "fwd_m", "lat_m"])
        return estimate_observations(det, ws, run_dir, args, "ideal")
    det = read_csv(run_dir / f"detections_{args.detector}.csv")
    if det is None:
        return None
    return estimate_observations(det, ws, run_dir, args, args.depth)


# ------------------------------------------------------------------------------------------------
# Controller replay (USafetyBrakeController)
# ------------------------------------------------------------------------------------------------

def replay_controller(frames: pd.DataFrame, obs: pd.DataFrame, latency: float, args) -> dict:
    """frames: frame_index, t, v (forward speed m/s) for every perception frame, in order."""
    grouped = {k: g for k, g in obs.groupby("frame_index")} if len(obs) else {}
    history: list[bool] = []
    prev_t = None
    for fr in frames.itertuples(index=False):
        period = (fr.t - prev_t) if prev_t is not None and fr.t > prev_t else args.perception_period
        prev_t = fr.t
        v = max(0.0, fr.v)
        delay = latency + (0.0 if args.no_detection_delay
                           else args.confirm_frames * period + args.controller_dead_time)
        trigger = v * delay + v * v / (2.0 * max(0.1, args.assumed_decel)) + args.margin

        hazard, closest = False, np.nan
        g = grouped.get(fr.frame_index)
        if g is not None:
            ok = (g.confidence >= args.min_confidence) & g.fwd_m.notna() & (g.lat_m.abs() <= args.corridor_half_width) \
                 & (g.fwd_m > -0.5) & (g.fwd_m <= trigger)
            if ok.any():
                hazard, closest = True, float(g.fwd_m[ok].min())

        history.append(hazard)
        history = history[-args.window_frames:]
        if hazard and sum(history) >= args.confirm_frames:
            brake_t = math.ceil((fr.t + latency) / TICK - 1e-6) * TICK
            return {"braked": True, "confirm_time_s": fr.t, "confirm_frame": int(fr.frame_index),
                    "est_dist_at_confirm_m": closest, "trigger_dist_at_confirm_m": trigger,
                    "speed_at_confirm_kmh": v * 3.6, "brake_time_s": brake_t}
    return {"braked": False}


def run_tracker(frames: pd.DataFrame, obs: pd.DataFrame, args) -> dict[int, list]:
    """Runs the tracker once per run (independent of latency and controller). Returns, per frame index, the live
    tracks as (tid, x, y, vx, vy, hits, misses) in world coordinates."""
    params = TrackerParams(args.track_meas_sigma, args.track_accel_sigma, args.track_init_vel_sigma, args.track_gate,
                           args.track_min_hits, args.track_max_miss)
    trk = Tracker(params)
    ok = obs[(obs.confidence >= args.min_confidence) & obs.fwd_m.notna()] if len(obs) else obs
    grouped = {k: g for k, g in ok.groupby("frame_index")} if len(ok) else {}
    states = {}
    for fr in frames.itertuples(index=False):
        f_x, f_y = math.cos(math.radians(fr.yaw)), math.sin(math.radians(fr.yaw))
        r_x, r_y = -f_y, f_x
        meas = []
        g = grouped.get(fr.frame_index)
        if g is not None:
            for o in g.itertuples(index=False):
                ahead = o.fwd_m + args.front_offset
                meas.append((fr.x + ahead * f_x + o.lat_m * r_x, fr.y + ahead * f_y + o.lat_m * r_y, o.confidence))
        live = trk.step(fr.t, meas)
        states[int(fr.frame_index)] = [(tr.tid, *tr.x, tr.hits, tr.misses) for tr in live]
    return states


def replay_predictive(frames: pd.DataFrame, obs: pd.DataFrame, tracks: dict[int, list], latency: float, args,
                      mode: str) -> dict:
    """mode 'ttc' (prediction only) or 'hybrid' (zone OR prediction). Same latency, dead time, margin, corridor and
    assumed deceleration as the zone controller."""
    grouped = {k: g for k, g in obs.groupby("frame_index")} if len(obs) else {}
    history: list[bool] = []
    prev_t = None
    for fr in frames.itertuples(index=False):
        period = (fr.t - prev_t) if prev_t is not None and fr.t > prev_t else args.perception_period
        prev_t = fr.t
        v = max(0.0, fr.v)

        # zone part (identical to replay_controller)
        zone_fire, closest = False, np.nan
        if mode == "hybrid":
            delay = latency + (0.0 if args.no_detection_delay else args.confirm_frames * period + args.controller_dead_time)
            trigger = v * delay + v * v / (2.0 * max(0.1, args.assumed_decel)) + args.margin
            hazard = False
            g = grouped.get(fr.frame_index)
            if g is not None:
                okm = (g.confidence >= args.min_confidence) & g.fwd_m.notna() & (g.lat_m.abs() <= args.corridor_half_width) \
                      & (g.fwd_m > -0.5) & (g.fwd_m <= trigger)
                if okm.any():
                    hazard, closest = True, float(g.fwd_m[okm].min())
            history.append(hazard)
            history = history[-args.window_frames:]
            zone_fire = hazard and sum(history) >= args.confirm_frames

        # predictive part: would waiting one more frame let a tracked person get inside the margin before the stop?
        ttc_fire, best_gap, best_tid = False, math.inf, None
        f_x, f_y = math.cos(math.radians(fr.yaw)), math.sin(math.radians(fr.yaw))
        r_x, r_y = -f_y, f_x
        wait = period + latency + args.controller_dead_time
        for tid, x, y, vx, vy, hits, misses in tracks.get(int(fr.frame_index), []):
            if hits < args.track_min_hits:
                continue
            dx, dy = x - fr.x, y - fr.y
            p_fwd = dx * f_x + dy * f_y - args.front_offset
            p_lat = dx * r_x + dy * r_y
            conflict, gap = predicted_conflict(p_fwd, p_lat, vx * f_x + vy * f_y, vx * r_x + vy * r_y, v,
                                               max(0.1, args.assumed_decel), wait, args.corridor_half_width,
                                               args.margin, args.ttc_dt)
            if gap < best_gap:
                best_gap, best_tid = gap, tid
            ttc_fire |= conflict

        if zone_fire or ttc_fire:
            brake_t = math.ceil((fr.t + latency) / TICK - 1e-6) * TICK
            reason = "both" if (zone_fire and ttc_fire) else ("zone" if zone_fire else "ttc")
            return {"braked": True, "confirm_time_s": fr.t, "confirm_frame": int(fr.frame_index),
                    "est_dist_at_confirm_m": closest, "trigger_reason": reason,
                    "pred_gap_at_confirm_m": round(best_gap, 3) if math.isfinite(best_gap) else None,
                    "track_id": best_tid, "speed_at_confirm_kmh": v * 3.6, "brake_time_s": brake_t}
    return {"braked": False}


def track_velocity_error(tracks: dict[int, list], frames: pd.DataFrame, ws: pd.DataFrame, args) -> dict:
    """Median |v_track - v_true| over confirmed, freshly updated tracks within 1 m of a true person."""
    peds = ws[ws.role == "pedestrian"]
    by_f = {k: g for k, g in peds.groupby("frame_index")}
    errs = []
    for f, live in tracks.items():
        g = by_f.get(f)
        if g is None:
            continue
        P = g[["x_m", "y_m", "vx_mps", "vy_mps"]].to_numpy(float)
        for tid, x, y, vx, vy, hits, misses in live:
            if hits < args.track_min_hits or misses > 0:
                continue
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            i = int(np.argmin(d))
            if d[i] <= 1.0:
                errs.append(math.hypot(vx - P[i, 2], vy - P[i, 3]))
    return {"track_vel_err_med_mps": round(float(np.median(errs)), 3) if errs else None,
            "track_vel_err_p90_mps": round(float(np.percentile(errs, 90)), 3) if errs else None}


# ------------------------------------------------------------------------------------------------
# Kinematics: logged path, then the calibrated stop
# ------------------------------------------------------------------------------------------------

def make_decel(table: str):
    pts = sorted((float(a), float(b)) for a, b in (kv.split(":") for kv in table.split(",")))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return lambda v_mps: float(np.interp(v_mps * 3.6, xs, ys))


class ForkliftPath:
    """Forklift pose as a function of distance travelled along its logged path (extrapolated straight at the end)."""

    def __init__(self, fk: pd.DataFrame):
        self.t = fk.timestamp_s.to_numpy(float)
        self.x = fk.x_m.to_numpy(float)
        self.y = fk.y_m.to_numpy(float)
        self.yaw = np.degrees(np.unwrap(np.radians(fk.yaw_deg.to_numpy(float))))
        yaw_r = np.radians(self.yaw)
        self.v = fk.vx_mps.to_numpy(float) * np.cos(yaw_r) + fk.vy_mps.to_numpy(float) * np.sin(yaw_r)
        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])

    def s_at_time(self, t):
        return np.interp(t, self.t, self.s)

    def speed_at_time(self, t):
        return float(np.interp(t, self.t, self.v))

    def pose_at_s(self, s):
        s = np.asarray(s, float)
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        yaw = np.interp(s, self.s, self.yaw)
        beyond = s > self.s[-1]
        if beyond.any():
            e = s[beyond] - self.s[-1]
            yr = math.radians(self.yaw[-1])
            x[beyond] = self.x[-1] + e * math.cos(yr)
            y[beyond] = self.y[-1] + e * math.sin(yr)
            yaw[beyond] = self.yaw[-1]
        return x, y, yaw


def braked_distance(tau, v_b, a, td):
    """Distance travelled tau seconds after the brake command: dead time at v_b, then constant deceleration."""
    tau = np.maximum(tau, 0.0)
    t_stop = v_b / a
    during = np.minimum(tau, td) * v_b
    tb = np.clip(tau - td, 0.0, t_stop)
    return during + v_b * tb - 0.5 * a * tb ** 2


# ------------------------------------------------------------------------------------------------
# One run
# ------------------------------------------------------------------------------------------------

def replay_run(run_dir: Path, args, decel_fn) -> list[dict]:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    safety = meta.get("safety") or {}
    if safety.get("enabled") and not args.include_closed_loop:
        return []
    if meta.get("end_reason") in ("Aborted", "Stopped", "EndPlay", "Superseded"):
        return []          # interrupted run (e.g. a stopped batch); its resumed copy is in another batch folder
    if args.detector not in ("unreal", "unreal-recomp") and args.depth != "ideal" and not (run_dir / f"depth_{args.depth}").is_dir():
        return []          # estimated depth exists only for a subset of runs: skip the others instead of replaying blind
    ws = read_csv(run_dir / "world_state.csv")
    if ws is None:
        return []

    fk = ws[ws.role == "forklift"].sort_values("timestamp_s").drop_duplicates("timestamp_s")
    path = ForkliftPath(fk)
    frames = pd.DataFrame({"frame_index": fk.frame_index.to_numpy(int), "t": path.t, "v": path.v,
                           "x": path.x, "y": path.y, "yaw": path.yaw})

    obs = observations_for_run(run_dir, ws, args)
    if obs is None:
        return []

    names = [p["actor_name"] for p in meta.get("pedestrians", [])] or sorted(ws[ws.role == "pedestrian"].actor_name.unique())
    peds = ws[ws.role == "pedestrian"]
    g = Geometry(args.front_offset, args.body_front, args.rear_offset, args.body_width, args.fork_width, args.ped_radius)

    t = np.arange(path.t[0], path.t[-1] + 1e-9, 1.0 / args.interp_hz)
    s_log = path.s_at_time(t)

    def clearance_series(s_track):
        x, y, yaw = path.pose_at_s(s_track)
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        best = np.full_like(t, np.inf)
        for n in names:
            p = peds[peds.actor_name == n].sort_values("timestamp_s").drop_duplicates("timestamp_s")
            if len(p) < 2:
                continue
            px = np.interp(t, p.timestamp_s, p.x_m)
            py = np.interp(t, p.timestamp_s, p.y_m)
            dx, dy = px - x, py - y
            lx, ly = dx * cy + dy * sy, -dx * sy + dy * cy
            best = np.minimum(best, footprint_distance(lx, ly, g) - g.ped_radius)
        return best

    # without braking = the open-loop run as logged
    base = clearance_series(s_log)
    base_contact = bool((base < 0).any())
    base_contact_t = float(t[np.argmax(base < 0)]) if base_contact else np.nan

    tracks, track_diag = None, {}
    if any(c != "zone" for c in args.controllers):
        tracks = run_tracker(frames, obs, args)
        track_diag = track_velocity_error(tracks, frames, ws, args)
        if args.dump_tracks:
            tdir = (args.out or (args.path / "analysis")) / "tracks"
            tdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"frame_index": f, "tid": tid, "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": vy,
                           "hits": h, "misses": m} for f, live in tracks.items()
                          for tid, x, y, vx, vy, h, m in live]).to_csv(tdir / f"{meta.get('run_id', run_dir.name)}.csv",
                                                                       index=False)

    rows = []
    for controller in args.controllers:
        for L in args.latencies:
            if controller == "zone":
                dec = replay_controller(frames, obs, L, args)
                dec.setdefault("trigger_reason", "zone" if dec["braked"] else None)
            else:
                dec = replay_predictive(frames, obs, tracks, L, args, controller)
            s_track = s_log.copy()
            row = {"run_dir": run_dir.as_posix(), "run_id": meta.get("run_id"), "variant": variant_id(meta.get("run_id", "")),
                   "scenario_type": meta.get("scenario_type"), "seed": meta.get("seed"),
                   "target_kmh": meta.get("forklift_target_kmh"), "timing_shift_s": meta.get("timing_shift_s"),
                   "detector": args.detector, "depth": "unreal" if args.detector == "unreal" else args.depth + ("+range" if args.depth != "ideal" and args.depth_is_range else ""),
                   "controller": controller, "latency_s": L, "conflict_without_braking": base_contact,
                   "min_clearance_without_braking_m": round(float(base.min()), 3), **dec,
                   **(track_diag if controller != "zone" else {})}
            if dec["braked"]:
                tb = dec["brake_time_s"]
                v_b = path.speed_at_time(tb)
                a = decel_fn(v_b)
                s_b = float(path.s_at_time(tb))
                after = t >= tb
                s_track[after] = s_b + braked_distance(t[after] - tb, v_b, a, args.vehicle_dead_time)
                row.update({"speed_at_brake_kmh": round(v_b * 3.6, 2), "braking_distance_m":
                            round(float(braked_distance(1e9, v_b, a, args.vehicle_dead_time)), 3),
                            "stop_time_s": round(tb + args.vehicle_dead_time + v_b / a, 3),
                            "brake_before_conflict_s": round(base_contact_t - tb, 3) if base_contact else None})
            speed_track = np.gradient(s_track, t)
            clr = np.where(speed_track > MOVING_SPEED_MPS, clearance_series(s_track), np.inf)   # contact only while moving
            contact = bool((clr < 0).any())
            row.update({"min_clearance_m": round(float(clr.min()), 3), "contact": contact,
                        "contact_speed_kmh": round(float(speed_track[np.argmax(clr < 0)] * 3.6), 2) if contact else None})
            if base_contact:
                row["outcome"] = "avoided" if not contact else ("mitigated" if dec["braked"] and
                                                                 dec["brake_time_s"] < base_contact_t else "missed")
            else:
                row["outcome"] = "unnecessary_stop" if dec["braked"] else "no_stop_correct"
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return rows


# ------------------------------------------------------------------------------------------------
# Validation against closed-loop Unreal runs
# ------------------------------------------------------------------------------------------------

def validate(replay: pd.DataFrame, closed_root: Path, args) -> pd.DataFrame:
    from evaluate_runs import evaluate_run, EXPECTED_CONFIG as EC
    ev_args = argparse.Namespace(front_offset=args.front_offset, body_front=args.body_front, rear_offset=args.rear_offset,
                                 body_width=args.body_width, fork_width=args.fork_width, ped_radius=args.ped_radius,
                                 corridor_half_width=args.corridor_half_width, interp_hz=args.interp_hz, near_range=10.0,
                                 no_config_check=True, expected_config=EC)
    rows = []
    metas = sorted(closed_root.rglob("run_metadata.json"))
    print(f"Validating against {len(metas)} closed-loop run(s) in {closed_root}...", flush=True)
    t0 = time.time()
    for k, m in enumerate(metas, 1):
        progress(k, len(metas), m.parent.name, t0)
        meta = json.loads(m.read_text(encoding="utf-8"))
        safety = meta.get("safety") or {}
        if not safety.get("enabled"):
            continue
        L = round(float(safety.get("reaction_latency_s", 0.1)), 3)
        zone = replay[replay.controller == "zone"] if "controller" in replay else replay   # Unreal runs the zone rule
        match = zone[(zone.variant == variant_id(meta.get("run_id", ""))) & (zone.seed == meta.get("seed"))
                     & (zone.latency_s.round(3) == L)]
        ev, _ = evaluate_run(m.parent, m.parent, ev_args)
        if match.empty or not ev:
            rows.append({"closed_run": m.parent.as_posix(), "matched": False})
            continue
        r = match.iloc[0]
        actual = min(ev, key=lambda e: e["min_clearance_m"])
        actual_braked = safety.get("brake_applied_time_s", -1) >= 0
        rows.append({
            "closed_run": m.parent.as_posix(), "matched": True, "variant": r.variant, "seed": r.seed, "latency_s": L,
            "braked_replay": r.braked, "braked_unreal": actual_braked,
            "est_dist_at_confirm_replay_m": r.get("est_dist_at_confirm_m"),
            "est_dist_at_confirm_unreal_m": safety.get("hazard_distance_at_confirm_m"),
            "braking_distance_replay_m": r.get("braking_distance_m"),
            "braking_distance_unreal_m": safety.get("braking_distance_m"),
            "min_clearance_replay_m": r.min_clearance_m, "min_clearance_unreal_m": actual["min_clearance_m"],
            "clearance_diff_m": round(r.min_clearance_m - actual["min_clearance_m"], 3),
            "contact_replay": r.contact, "contact_unreal": actual["contact"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def paired_table(res: pd.DataFrame) -> pd.DataFrame:
    """Same pedestrian (seed) and timing at every forklift speed: one row per pair, one column per speed."""
    if res.empty:
        return res
    r = res.copy()
    r["family"] = r.variant.str.replace(r"_v\d+(\.\d+)?", "", regex=True)
    keys = ["family", "timing_shift_s", "seed", "detector", "depth", "controller", "latency_s"]
    out = r.pivot_table(index=keys, columns="target_kmh", values="min_clearance_m", aggfunc="first")
    out.columns = [f"min_clearance_{c:g}kmh" for c in out.columns]
    return out.reset_index()


def main() -> None:
    args = parse_args()
    decel_fn = make_decel(args.vehicle_decel)
    run_dirs = sorted({m.parent for m in args.path.rglob("run_metadata.json")})
    rows, skipped = [], 0
    print(f"Replaying {len(run_dirs)} run(s): controllers {' '.join(args.controllers)}, latencies "
          f"{' '.join(f'{l:g}' for l in args.latencies)} s, detector {args.detector}", flush=True)
    t0 = time.time()
    for k, d in enumerate(run_dirs, 1):
        progress(k - 1, len(run_dirs), d.name, t0)
        try:
            r = replay_run(d, args, decel_fn)
        except Exception as exc:
            print(f"\n  ! {d}: {exc}")
            r = []
        skipped += 0 if r else 1
        rows += r
    progress(len(run_dirs), len(run_dirs), "done", t0)
    print()
    out = args.out or (args.path / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "replay_results.csv", index=False)
    if not res.empty:
        paired_table(res).to_csv(out / "replay_paired.csv", index=False)
    print(f"{len(run_dirs)} run(s), {len(run_dirs) - skipped} replayed (closed-loop or incomplete runs skipped) -> {out}")
    if not res.empty:
        cols = ["run_id", "controller", "latency_s", "braked", "trigger_reason", "est_dist_at_confirm_m",
                "pred_gap_at_confirm_m", "speed_at_brake_kmh",
                "min_clearance_without_braking_m", "min_clearance_m", "contact", "outcome"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(res[[c for c in cols if c in res]].to_string(index=False))
    if args.validate:
        v = validate(res, args.validate, args)
        v.to_csv(out / "replay_validation.csv", index=False)
        print("\nValidation against closed-loop runs:")
        print(v.to_string(index=False) if not v.empty else "  no closed-loop runs found")


if __name__ == "__main__":
    main()
