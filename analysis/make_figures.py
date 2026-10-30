#!/usr/bin/env python3
"""
Thesis figures and statistical tests from the analysis CSVs.

Usage (every input is optional; each figure is made when its inputs are given):
    python make_figures.py --out figures \\
        --det  <batch1>/analysis_det                       (detection_metrics.py output, detector unreal)
        --models <batch1>/analysis_det_yolo11n ...          (more detection_metrics outputs, one per detector; optional)
        --timing <batch1>/analysis_det/detector_timing_*.csv
        --replay <batch1>/analysis/replay_results.csv       (replay_braking.py with --controllers zone ttc hybrid)
        --closed <batch2>/analysis/pedestrian_results.csv   (evaluate_runs.py on closed-loop batches)
        --validation <batch1>/analysis/replay_validation.csv

Figures (PDF for the thesis + PNG for checking), all in --out:
    F1_recall_vs_distance      recall per distance band, one panel per scenario (detector unreal)
    F2_clearance_vs_time       minimum clearance against time available (corridor entry to contact), one panel
                               per controller, colour + marker = forklift speed; contact is below 0
    F3_contact_rate            contact rate per scenario and controller (no braking / zone / ttc / hybrid), Wilson 95%
    F4_outcomes                outcome shares per controller and scenario (avoided, mitigated, missed, unnecessary stop)
    F5_depth_error             median depth error per distance band with 5-95% whiskers
    F6_model_tradeoff          recall within 10 m against inference time per detector (needs --models + --timing)
    F7_replay_validation       replay vs Unreal minimum clearance for the closed-loop runs (needs --validation)
    F10_relative_motion        S9 vs S10, zone vs ttc minimum clearance (offline replay)
    F11_detection_rate         same closed-loop runs at 5 Hz vs 10 Hz (needs --rate base.csv fast.csv)
    F12_depth_source           contacts per controller with rendered vs monocular depth (needs --depth-replay)
    F13_latency                contacts and conflict-free stops against replay latency (from --replay, all latencies)

Statistics (CSV in --out, also printed):
    stats_contact_rates.csv    contacts / runs with Wilson 95% intervals per scenario x speed x controller
    stats_unnecessary_stops.csv  stop (false-alarm) rate in conflict-free runs per scenario x timing x speed x controller
    F8_unnecessary_stops_by_timing  that rate against the timing shift (ApproachAndStop, LateralCrossing +2/+3 s)
    stats_paired_tests.csv     same runs, controller A vs B: exact McNemar test on contact and on unnecessary stops,
                               median paired clearance difference with a bootstrap 95% interval and a Wilcoxon
                               signed-rank p-value
    Runs are independent repetitions, so run-level Wilson intervals and paired tests are valid; frame-level numbers
    are not (use detection_metrics.py --bootstrap for those).

Requires: pandas, numpy, matplotlib, scipy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

# ---- palette (validated: first three categorical slots pass all-pairs CVD on white; aqua < 3:1 contrast, so every
#      colour is paired with a marker shape and a legend) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]              # blue, orange, aqua
MARKERS = ["o", "s", "^"]
VAL_COLORS = CAT + ["#8a5cd1", "#9a6b2f", "#d6458f"]   # F7 groups beyond the three controller colours
NEUTRAL = "#b3b2ad"                                   # "no braking" reference
STATUS = {"avoided": "#0ca30c", "mitigated": "#fab219", "missed": "#d03b3b", "unnecessary_stop": "#ec835a",
          "no_stop_correct": "#d9d8d2"}
OUTCOME_LABEL = {"avoided": "Avoided", "mitigated": "Mitigated (contact, slower)", "missed": "Missed (no braking before contact)",
                 "unnecessary_stop": "Unnecessary stop", "no_stop_correct": "Correctly no stop"}
CONTROLLER_LABEL = {"none": "No braking", "zone": "Zone (position only)", "ttc": "Predictive (tracking + TTC)",
                    "hybrid": "Hybrid (zone or predictive)"}
SPEED_ORDER = [5.0, 10.0, 15.0]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def family(variant: str) -> str:
    """LateralCrossing_v10_t+2.0 -> LateralCrossing."""
    import re
    return re.sub(r"(_v\d+(\.\d+)?)?(_t[+-]\d+(\.\d+)?)?$", "", str(variant))


# ------------------------------------------------------------------------------------------------
# F1 recall vs distance, F5 depth error, F6 model trade-off
# ------------------------------------------------------------------------------------------------

def fig_recall_vs_distance(det_dir: Path, out: Path):
    b = pd.read_csv(det_dir / "det_recall_bins.csv")
    b = b[(b.dimension == "distance_m") & ~b.group.str.contains("@") & (b.group != "ALL")]
    if b.empty:
        return
    b["lo_m"] = b.bin.str.extract(r"\[([\d.]+),").astype(float)
    b["hi_m"] = b.bin.str.extract(r", ([\d.inf]+)\)")[0].replace("inf", np.nan).astype(float)
    b["mid"] = np.where(b.hi_m.notna(), (b.lo_m + b.hi_m) / 2, b.lo_m + 5)
    groups = sorted(b.group.unique())
    n = len(groups)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.9 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, g in zip(axes.flat, groups):
        s = b[b.group == g].sort_values("mid")
        # clip: a Wilson bound can land 1e-16 on the wrong side of a recall of exactly 1.0 (floating point)
        yerr = [(s.recall - s.wilson_lo).clip(lower=0), (s.wilson_hi - s.recall).clip(lower=0)]
        ax.errorbar(s.mid, s.recall, yerr=yerr, color=CAT[0],
                    marker="o", ms=4, lw=2, elinewidth=1, capsize=0)
        ax.set_title(g, loc="left")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, 60)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Distance ahead of fork tips (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall")
    fig.suptitle("Detection recall by distance (people at least 50% visible; bars: per-frame Wilson 95%)",
                 x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "F1_recall_vs_distance")


def fig_depth_error(det_dir: Path, out: Path):
    p = det_dir / "det_depth_error.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["lo"] = d.distance_band_m.str.extract(r"\[([\d.]+),").astype(float)
    d["hi"] = d.distance_band_m.str.extract(r", ([\d.inf]+)\)")[0].replace("inf", np.nan).astype(float)
    d["mid"] = np.where(d.hi.notna(), (d.lo + d.hi) / 2, d.lo + 5)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.axhline(0, color=AXIS, lw=1)
    ax.errorbar(d.mid, d.median_err_m, yerr=[d.median_err_m - d.p05_err_m, d.p95_err_m - d.median_err_m],
                color=CAT[0], marker="o", ms=5, lw=0, elinewidth=1.2, capsize=0)
    for _, r in d.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.mid, r.p95_err_m), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=7, color=MUTED)
    ax.set_xlabel("True distance ahead of fork tips (m)")
    ax.set_ylabel("Estimate - body center (m)")
    ax.set_title("Depth error by distance: median, 5-95% whiskers", loc="left")
    ax.set_xlim(0, d.mid.max() + 2.5)
    ax.set_ylim(top=0.04)
    save(fig, out, "F5_depth_error")


def fig_model_tradeoff(model_dirs: list[Path], timing_files: list[Path], out: Path):
    if not model_dirs or not timing_files:
        return
    # Unreal's YOLO11x detections and the Python yolo11x timing are the same model (pipeline check), so both map to
    # "yolo11x"; Python-run detectors keep their own names (yolo11n, yolo11s, ...).
    alias = lambda n: "yolo11x" if str(n) in ("unreal", "yolo11x_py") else str(n)
    t = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)
    t["detector"] = t.detector.map(alias)
    t = t.drop_duplicates("detector", keep="last")
    rows = []
    for d in model_dirs:
        s = pd.read_csv(d / "det_summary.csv")
        runs = pd.read_csv(d / "det_runs.csv") if (d / "det_runs.csv").exists() else pd.DataFrame()
        if len(runs) and "detector" in runs:
            det = runs.detector.iloc[0]
        else:                                          # fall back to the folder name: analysis_det_<name> / analysis_det
            det = d.name.split("analysis_det_", 1)[1] if d.name.startswith("analysis_det_") else "unreal"
        a = s[s.group == "ALL"].iloc[0]
        row = {"detector": alias(det)}
        for c in ("recall", "recall_within_10m", "precision", "fp_per_frame", "ap50_trunc", "mr2_trunc", "mean_iou_tp",
                  "recall_boot_lo", "recall_boot_hi", "recall_within_10m_boot_lo", "recall_within_10m_boot_hi"):
            if c in a:
                row[c] = a[c]
        rows.append(row)
    m = pd.DataFrame(rows).merge(t[["detector", "inference_ms_mean", "inference_ms_p95", "total_ms_mean"]],
                                 on="detector", how="left")
    m = m.sort_values("inference_ms_mean")
    m.to_csv(out / "model_comparison.csv", index=False)
    missing = m[m.inference_ms_mean.isna()].detector.tolist()
    if missing:
        print(f"  F6: no timing for {missing} (pass their detector_timing_*.csv with --timing)")
    m = m.dropna(subset=["inference_ms_mean"])
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(m.inference_ms_mean, m.recall_within_10m, "-s", color=CAT[1], ms=6, lw=1, label="Recall within 10 m")
    ax.plot(m.inference_ms_mean, m.recall, "-o", color=CAT[0], ms=6, lw=1, label="Recall within 30 m")
    for _, r in m.iterrows():                         # size letter above the upper (10 m) point, clear of the markers
        short = str(r.detector).replace("yolo11", "")
        ax.annotate(short, (r.inference_ms_mean, max(r.recall, r.recall_within_10m)), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=INK2)
    ax.set_ylim(min(0.7, m.recall.min() - 0.03), 1.02)
    ax.set_xlabel("Inference time per frame, RTX 3060, batch 1 (ms)")
    ax.set_ylabel("Recall")
    ax.legend(loc="lower right")
    ax.set_title("YOLO11 model size (n, s, m, l, x): recall against inference time", loc="left")
    save(fig, out, "F6_model_tradeoff")


# ------------------------------------------------------------------------------------------------
# Braking outcomes
# ------------------------------------------------------------------------------------------------

def load_outcomes(replay_files: list[Path], closed_files: list[Path], latency: float) -> pd.DataFrame:
    """One row per (run, controller): family, speed, timing, contact, braked, outcome, clearance, time available."""
    parts = []
    for f in replay_files:
        r = pd.read_csv(f)
        r = r[np.isclose(r.latency_s, latency)] if "latency_s" in r else r
        if "controller" not in r:
            r["controller"] = "zone"
        r["family"] = r.variant.map(family)
        base = r.drop_duplicates("run_id").copy()
        base["controller"] = "none"
        base["contact"] = base.conflict_without_braking
        base["min_clearance_m"] = base.min_clearance_without_braking_m
        base["braked"] = False
        base["outcome"] = np.where(base.contact, "missed", "no_stop_correct")
        parts += [r.assign(source="replay"), base.assign(source="replay")]
    for f in closed_files:
        c = pd.read_csv(f)
        c = c[c.safety_enabled.astype(bool)] if "safety_enabled" in c else c
        c = c[c.valid.astype(bool)] if "valid" in c else c
        c = c.loc[c.groupby("run_key").min_clearance_m.idxmin()]
        c["family"] = c.variant.map(family)
        c["controller"] = "zone"
        c["braked"] = c.confirm_time_s.notna()
        parts.append(c.assign(source="unreal_closed_loop"))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["contact"] = d.contact.astype(bool)
    return d


def contact_rate_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, fam, spd, ctl), g in d.groupby(["source", "family", "target_kmh", "controller"]):
        k, n = int(g.contact.sum()), len(g)
        lo, hi = wilson(k, n)
        unn = int((g.get("outcome", pd.Series(dtype=str)) == "unnecessary_stop").sum())
        rows.append({"source": src, "family": fam, "target_kmh": spd, "controller": ctl, "runs": n, "contacts": k,
                     "contact_rate": k / n, "wilson_lo": lo, "wilson_hi": hi, "unnecessary_stops": unn,
                     "min_clearance_median_m": g.min_clearance_m.median()})
    return pd.DataFrame(rows)


def fig_contact_rate(d: pd.DataFrame, out: Path):
    """Dot + Wilson interval per scenario and controller (dots, not bars, so a rate of 0 stays visible)."""
    # Only runs that end in contact without braking: otherwise runs where the worker passes safely anyway (shifted
    # timings, StationaryNearPath) dilute the rate. Unnecessary stops in the other runs are shown in F4.
    r = d[(d.source == "replay") & (d.controller != "none")]
    if "conflict_without_braking" in r:
        r = r[r.conflict_without_braking.astype(bool)]
    if r.empty:
        return
    ctls = [c for c in ["zone", "ttc", "hybrid"] if c in r.controller.unique()]
    colors = {"none": NEUTRAL, "zone": CAT[0], "ttc": CAT[1], "hybrid": CAT[2]}
    marks = {"none": "D", "zone": "o", "ttc": "s", "hybrid": "^"}
    fams = sorted(r.family.unique())
    speeds = [s for s in SPEED_ORDER if s in r.target_kmh.unique()]
    fig, axes = plt.subplots(1, len(speeds), figsize=(2.1 * len(speeds), 0.2 * len(fams) * len(ctls) + 1.3),
                             sharey=True, squeeze=False)
    step = 0.8 / len(ctls)
    for ax, spd in zip(axes[0], speeds):
        for fi in range(len(fams)):
            if fi % 2 == 1:
                ax.axhspan(fi - 0.5, fi + 0.5, color="#f4f4f1", lw=0, zorder=0)
        for ci, ctl in enumerate(ctls):
            for fi, fam in enumerate(fams):
                g = r[(r.family == fam) & (r.controller == ctl) & (r.target_kmh == spd)]
                if g.empty:
                    continue
                k, n = int(g.contact.sum()), len(g)
                a, b = wilson(k, n)
                y = fi + (ci - (len(ctls) - 1) / 2) * step
                ax.plot([a, b], [y, y], color=colors[ctl], lw=1.2, solid_capstyle="round", zorder=2)
                ax.plot(k / n, y, marker=marks[ctl], color=colors[ctl], ms=6, mec="white", mew=0.8, ls="none",
                        zorder=3, label=CONTROLLER_LABEL[ctl] if fi == 0 else None)
        ax.set_title(f"{spd:g} km/h", loc="left")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(True, axis="x", which="minor", color=GRID)
        ax.set_xlabel("Contact rate")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(fams)), fams)
    axes[0][0].set_ylim(len(fams) - 0.5, -0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=len(ctls))
    fig.suptitle("Contact rate in runs that collide without braking (offline replay; lines: Wilson 95% over runs)", x=0.0,
                 ha="left", fontsize=9, color=INK)
    for ax in axes[0][1:]:
        ax.tick_params(axis="y", left=False)
    fig.tight_layout()
    save(fig, out, "F3_contact_rate")


def fig_outcomes(d: pd.DataFrame, out: Path):
    r = d[(d.source == "replay") & (d.controller != "none")]
    if r.empty:
        return
    ctls = [c for c in ["zone", "ttc", "hybrid"] if c in r.controller.unique()]
    fams = sorted(r.family.unique())
    order = ["avoided", "mitigated", "missed", "unnecessary_stop", "no_stop_correct"]
    fig, axes = plt.subplots(1, len(ctls), figsize=(2.4 * len(ctls), 0.32 * len(fams) + 1.3), sharey=True,
                             squeeze=False)
    for ax, ctl in zip(axes[0], ctls):
        left = np.zeros(len(fams))
        for oc in order:
            share = np.array([((r.family == f) & (r.controller == ctl) & (r.outcome == oc)).sum() /
                              max(1, ((r.family == f) & (r.controller == ctl)).sum()) for f in fams])
            ax.barh(range(len(fams)), share, left=left, height=0.7, color=STATUS[oc], label=OUTCOME_LABEL[oc],
                    edgecolor="white", lw=1.5)
            left += share
        ax.set_title(CONTROLLER_LABEL[ctl], loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(True, axis="x", which="minor", color=GRID)
        ax.set_xlabel("Share of runs")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(fams)), fams)
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3)
    fig.suptitle("Braking outcome per scenario (all speeds and timings; offline replay)", x=0.0, ha="left",
                 fontsize=9, color=INK)
    for ax in axes[0][1:]:
        ax.tick_params(axis="y", left=False)
    fig.tight_layout()
    save(fig, out, "F4_outcomes")


def stop_rate_table(d: pd.DataFrame) -> pd.DataFrame:
    """False-alarm rate: share of runs WITHOUT a conflict (no contact without braking) in which a controller braked,
    per family x timing shift x speed (and pooled over speeds, target_kmh = NaN)."""
    r = d[(d.source == "replay") & (d.controller != "none")]
    if r.empty or "conflict_without_braking" not in r or "timing_shift_s" not in r:
        return pd.DataFrame()
    r = r[~r.conflict_without_braking.astype(bool)]
    rows = []
    for keys, g in r.groupby(["family", "timing_shift_s", "controller"]):
        for spd, h in [(np.nan, g)] + list(g.groupby("target_kmh")):
            k, n = int(h.braked.astype(bool).sum()), len(h)
            lo, hi = wilson(k, n)
            rows.append({"family": keys[0], "timing_shift_s": keys[1], "controller": keys[2], "target_kmh": spd,
                         "runs_without_conflict": n, "stops": k, "stop_rate": k / n if n else np.nan,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "clearance_without_braking_median_m": h.min_clearance_without_braking_m.median()
                         if "min_clearance_without_braking_m" in h else np.nan})
    return pd.DataFrame(rows)


def fig_relative_motion(d: pd.DataFrame, out: Path):
    """F10: person walking along the lane towards the forklift (S9) or away from it (S10). Zone treats the person as
    standing still, so it brakes too late for one and too early for the other; ttc uses the tracked velocity."""
    fams = [("RelativeTowards", "Walking towards the forklift (S9)"),
            ("RelativeSameDirection", "Walking away, forklift catches up (S10)")]
    r = d[(d.source == "replay") & d.controller.isin(["zone", "ttc"]) & d.family.isin([f for f, _ in fams])]
    if r.family.nunique() < 2:
        return
    colors = {"zone": CAT[0], "ttc": CAT[1]}
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.axvspan(-1.5, 0, color="#f7e1e1", lw=0, zorder=0)
    rng = np.random.default_rng(7)
    ticks, labels, y = [], [], 0
    for fam, title in fams:
        for ctl in ["zone", "ttc"]:
            g = r[(r.family == fam) & (r.controller == ctl)]
            x = g.min_clearance_m.to_numpy(float)
            ax.scatter(x, y + rng.uniform(-0.18, 0.18, len(x)), s=14, color=colors[ctl], edgecolor="white", lw=0.5,
                       zorder=3, label=CONTROLLER_LABEL[ctl] if fam == fams[0][0] else None)
            ax.plot([np.median(x)] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.5, zorder=4)
            k = int(g.contact.sum())
            ax.text(ax.get_xlim()[1] if False else 3.05, y, f"{k}/{len(g)} contact", va="center", fontsize=7, color=INK2)
            ticks.append(y)
            labels.append(f"{title}" if ctl == "zone" else "")
            y += 1
        y += 0.6
    ax.set_yticks(ticks, labels, fontsize=7.5)
    ax.set_ylim(y - 0.9, -0.7)
    ax.set_xlim(-1.0, 3.6)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Minimum clearance (m); shaded = contact; black tick = median")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2)
    fig.suptitle("Person walking along the lane: zone ignores the person's motion (offline replay)",
                 x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "F10_relative_motion")


def fig_rate_pairs(files: list[Path], out: Path):
    """F11: the same closed-loop runs (same seeds) at two detection rates. One point per run pair: minimum clearance
    at the base rate (x) against the change at the faster rate (y). Above zero = the faster rate kept more clearance."""
    if len(files) != 2:
        return
    cols = ["variant", "seed", "target_kmh", "min_clearance_m", "contact", "contact_speed_kmh", "entry_to_confirm_s",
            "gt_dist_at_confirm_m"]
    a, b = (pd.read_csv(f) for f in files)
    a, b = a[a.valid.astype(bool)], b[b.valid.astype(bool)]
    m = a[cols].merge(b[cols], on=["variant", "seed", "target_kmh"], suffixes=("_base", "_fast"))
    if m.empty:
        print("  F11: no matching variant + seed between the two files")
        return
    m["family"] = m.variant.map(family)
    m["clearance_diff_m"] = m.min_clearance_m_fast - m.min_clearance_m_base
    m.to_csv(out / "stats_rate_pairs.csv", index=False)
    fams = sorted(m.family.unique())
    speeds = sorted(m.target_kmh.unique())
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axvspan(-1.2, 0, color="#f7e1e1", lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    for i, fam in enumerate(fams):
        for j, v in enumerate(speeds):
            g = m[(m.family == fam) & (m.target_kmh == v)]
            if g.empty:
                continue
            ax.scatter(g.min_clearance_m_base, g.clearance_diff_m, s=24, color=CAT[i % 3], marker=MARKERS[j % 3],
                       edgecolor="white", lw=0.5, zorder=3, label=f"{fam}, {v:g} km/h")
    hits = m[m.contact_base | m.contact_fast].sort_values("clearance_diff_m")
    for n, (_, r) in enumerate(hits.iterrows()):
        sb = f"{r.contact_speed_kmh_base:.1f} km/h" if r.contact_base else "no contact"
        sf = f"{r.contact_speed_kmh_fast:.1f} km/h" if r.contact_fast else "no contact"
        ax.annotate(f"{sb} \u2192 {sf}", (r.min_clearance_m_base, r.clearance_diff_m),
                    xytext=(1.45, 0.06 + 0.13 * n), textcoords=("data", "data"), fontsize=6.5, color=INK2,
                    va="center", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))
    ax.set_xlim(-1.2, 4.0)
    ax.set_xlabel("Minimum clearance at 5 Hz (m); shaded = contact at 5 Hz")
    ax.set_ylabel("Change at 10 Hz, same run (m)")
    k_a, k_b = int(m.contact_base.sum()), int(m.contact_fast.sum())
    ax.set_title(f"Same runs at 5 Hz and 10 Hz ({len(m)} pairs): contacts {k_a} \u2192 {k_b}. Late crossings gain clearance;\n"
                 f"runs with time to spare stop closer, because the trigger distance shrinks with the frame interval",
                 loc="left", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, out, "F11_detection_rate")
    print(f"     pairs {len(m)}, contacts {k_a} -> {k_b}, median clearance change "
          f"{m.clearance_diff_m.median():+.3f} m")


DEPTH_LABEL = {"unreal": "Rendered depth", "ideal": "Rendered depth", "da2s+range": "Depth Anything V2 Small", "da2l+range": "Depth Anything V2 Large",
               "da2b+range": "Depth Anything V2 Base"}


def fig_depth_braking(files: list[Path], latency: float, out: Path):
    """F12: contacts per controller on the same runs with only the depth source swapped. The first file is the
    reference (rendered depth); its rows are cut to the runs present in the other files."""
    if len(files) < 2:
        return
    frames = []
    for f in files:
        x = pd.read_csv(f)
        if "latency_s" in x:
            x = x[np.isclose(x.latency_s, latency)]
        frames.append(x)
    det = set(frames[1].detector) if "detector" in frames[1] else set()
    if "detector" in frames[0] and frames[0].detector.nunique() > 1 and det:
        frames[0] = frames[0][frames[0].detector.isin(det)]
    print(f"     reference: detector {sorted(set(frames[0].detector))}, depth {sorted(set(frames[0].depth))}")
    runs = set.intersection(*[set(x.run_id) for x in frames[1:]])
    frames[0] = frames[0][frames[0].run_id.isin(runs)]
    frames = [x[x.run_id.isin(set(frames[0].run_id))] for x in frames]
    ctls = [c for c in ["zone", "ttc", "hybrid"] if all(c in set(x.controller) for x in frames)]
    rows = []
    for x in frames:
        name = str(x.depth.iloc[0])
        for c in ctls:
            g = x[x.controller == c]
            k, n = int(g.contact.sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"depth": name, "controller": c, "contacts": k, "runs": n, "rate": k / n if n else np.nan,
                         "wilson_lo": lo, "wilson_hi": hi, "median_clearance_m": g.min_clearance_m.median()})
    t = pd.DataFrame(rows)
    t.to_csv(out / "stats_depth_contacts.csv", index=False)
    depths = list(dict.fromkeys(t.depth))
    colors = [NEUTRAL] + ["#7fb0ea", CAT[0], "#174a87"][: len(depths) - 1]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    w = 0.8 / len(depths)
    for i, dep in enumerate(depths):
        s = t[t.depth == dep].set_index("controller").loc[ctls]
        xs = np.arange(len(ctls)) + (i - (len(depths) - 1) / 2) * w
        ax.bar(xs, s.contacts, width=w * 0.92, color=colors[i % len(colors)], label=DEPTH_LABEL.get(dep, dep), zorder=3)
        for xv, k in zip(xs, s.contacts):
            ax.text(xv, k + 0.6, str(int(k)), ha="center", va="bottom", fontsize=7, color=INK2)
    n = int(t.runs.iloc[0])
    ax.set_xticks(np.arange(len(ctls)), [CONTROLLER_LABEL[c].replace(" (", "\n(") for c in ctls])
    ax.set_ylabel(f"Contacts (of {n} runs)")
    ax.set_ylim(0, t.contacts.max() * 1.18 + 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.24), ncol=len(depths), fontsize=7)
    ax.set_title(f"Only the depth source changes: contacts in {n} Batch 1 runs (offline replay, latency {latency:g} s)",
                 loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "F12_depth_source")


def fig_latency(replay_files: list[Path], out: Path):
    """F13: the same replayed runs at every latency in the files (0.05 / 0.1 / 0.2 s). Left: contacts among the runs
    that collide without braking. Right: stops in conflict-free runs. Only (run, controller) pairs present at every
    latency are used, so each line compares the same runs."""
    parts = [pd.read_csv(f) for f in replay_files]
    parts = [p for p in parts if "latency_s" in p and p.latency_s.nunique() > 1]
    if not parts:
        return
    r = pd.concat(parts, ignore_index=True)
    if "controller" not in r:
        r["controller"] = "zone"
    lats = sorted(r.latency_s.unique())
    n_lat = r.groupby(["run_id", "controller"]).latency_s.transform("nunique")
    r = r[n_lat == len(lats)].copy()
    n_ctl = r.groupby("run_id").controller.transform("nunique")
    r = r[n_ctl == r.controller.nunique()].copy()            # same runs for every controller
    r["contact"] = r.contact.astype(bool)
    r["conflict"] = r.conflict_without_braking.astype(bool)
    r["braked"] = r.braked.astype(bool)
    rows = []
    for (ctl, lat), g in r.groupby(["controller", "latency_s"]):
        cf, nc = g[g.conflict], g[~g.conflict]
        rows.append({"controller": ctl, "latency_s": lat, "conflict_runs": len(cf), "contacts": int(cf.contact.sum()),
                     "contact_speed_mean_kmh": cf.loc[cf.contact, "contact_speed_kmh"].mean(),
                     "conflict_free_runs": len(nc), "stops_conflict_free": int(nc.braked.sum()),
                     "trigger_dist_median_m": g.trigger_dist_at_confirm_m.dropna().median()
                     if "trigger_dist_at_confirm_m" in g and g.trigger_dist_at_confirm_m.notna().any() else np.nan})
    t = pd.DataFrame(rows)
    # paired exact McNemar, lowest vs highest latency, on contact in conflict runs
    tests = []
    for ctl, g in r[r.conflict].groupby("controller"):
        p = g.pivot_table(index="run_id", columns="latency_s", values="contact", aggfunc="first").astype(bool)
        worse = int((~p[lats[0]] & p[lats[-1]]).sum())
        better = int((p[lats[0]] & ~p[lats[-1]]).sum())
        tests.append({"controller": ctl, "worse_at_high_latency": worse, "better_at_high_latency": better,
                      "mcnemar_p": mcnemar_exact(worse, better)})
    t = t.merge(pd.DataFrame(tests), on="controller", how="left")
    t.to_csv(out / "stats_latency.csv", index=False)
    ctls = [c for c in ["zone", "ttc", "hybrid"] if c in set(t.controller)]
    colors = {"zone": CAT[0], "ttc": CAT[1], "hybrid": CAT[2]}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for i, c in enumerate(ctls):
        s = t[t.controller == c].sort_values("latency_s")
        # ttc and hybrid often coincide: offset the hybrid marker slightly so both stay visible
        dx = 0.004 * (i - 1)
        axes[0].plot(s.latency_s + dx, s.contacts, marker=MARKERS[i], color=colors[c], label=CONTROLLER_LABEL[c])
        axes[1].plot(s.latency_s + dx, s.stops_conflict_free, marker=MARKERS[i], color=colors[c])
    n_cf, n_nc = int(t.conflict_runs.iloc[0]), int(t.conflict_free_runs.iloc[0])
    axes[0].set_title(f"Contacts\n(of {n_cf} runs that collide without braking)", loc="left")
    axes[1].set_title(f"Stops where none was needed\n(of {n_nc} conflict-free runs)", loc="left")
    for ax in axes:
        ax.set_xticks(lats, [f"{l:g}" for l in lats])
        ax.set_xlabel("Latency (s)")
        ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "F13_latency")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t[["controller", "latency_s", "contacts", "conflict_runs", "stops_conflict_free", "conflict_free_runs",
                 "mcnemar_p"]].to_string(index=False))


def fig_stop_rate_by_timing(t: pd.DataFrame, out: Path):
    """F8: unnecessary-stop (false-alarm) rate against timing shift, one panel per family that has timing variants
    with conflict-free runs (e.g. ApproachAndStop, LateralCrossing +2/+3 s). Pooled over speeds, Wilson 95%."""
    if t.empty:
        return
    t = t[t.target_kmh.isna()]
    fams = [f for f, g in t.groupby("family") if g.timing_shift_s.nunique() > 1 or f.startswith("ApproachAndStop")]
    if not fams:
        return
    ctls = [c for c in ["zone", "ttc", "hybrid"] if c in t.controller.unique()]
    colors = {"zone": CAT[0], "ttc": CAT[1], "hybrid": CAT[2]}
    marks = {"zone": "o", "ttc": "s", "hybrid": "^"}
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 2.6), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        g = t[t.family == fam]
        xs = sorted(g.timing_shift_s.unique())
        off = {c: (i - (len(ctls) - 1) / 2) * 0.06 * max(1.0, (max(xs) - min(xs)) if len(xs) > 1 else 1.0)
               for i, c in enumerate(ctls)}
        for c in ctls:
            h = g[g.controller == c].sort_values("timing_shift_s")
            x = h.timing_shift_s + off[c]
            ax.errorbar(x, h.stop_rate, yerr=[(h.stop_rate - h.wilson_lo).clip(lower=0), (h.wilson_hi - h.stop_rate).clip(lower=0)], color=colors[c],
                        marker=marks[c], ms=5, lw=1.2, elinewidth=1, capsize=0, label=CONTROLLER_LABEL[c])
        ax.set_xticks(xs, [f"+{x:g}" for x in xs])
        ax.set_title(fam, loc="left")
        ax.set_xlabel("Nominal timing shift (s)")
        ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel("Share of runs with a stop")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=len(ctls))
    fig.suptitle("Unnecessary stops in runs that would not collide without braking (offline replay; Wilson 95%)",
                 x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "F8_unnecessary_stops_by_timing")


def fig_clearance_vs_time(d: pd.DataFrame, pedestrian_files: list[Path], out: Path):
    """x = time from corridor entry to the (unbraked) contact, from evaluate_runs on the open-loop batch."""
    r = d[(d.source == "replay")]
    if r.empty or not pedestrian_files:
        return
    ta = pd.concat([pd.read_csv(f) for f in pedestrian_files], ignore_index=True)
    ta = ta[~ta.safety_enabled.astype(bool)] if "safety_enabled" in ta else ta
    ta = ta.loc[ta.groupby("run_id").min_clearance_m.idxmin()][["run_id", "time_available_s"]]
    # d already has a time_available_s column (all empty for replay rows) when --closed is given, because the
    # closed-loop pedestrian_results.csv carries it; drop it so the merge does not produce _x/_y columns.
    r = r.drop(columns=["time_available_s"], errors="ignore").merge(ta, on="run_id", how="inner")
    r = r[r.time_available_s.notna() & (r.family != "Baseline")]   # Baseline: in the path from the start (15-44 s)
    if r.empty:
        return
    ctls = [c for c in ["none", "zone", "ttc", "hybrid"] if c in r.controller.unique()]
    ncols = 2 if len(ctls) > 2 else len(ctls)
    nrows = math.ceil(len(ctls) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows + 0.3), sharex=True, sharey=True,
                             squeeze=False)
    axes[0][0].set_xlim(0, max(1.0, r.time_available_s.max() * 1.05))
    for ax in axes.flat[len(ctls):]:
        ax.axis("off")
    for ax, ctl in zip(axes.flat, ctls):
        g = r[r.controller == ctl]
        ax.axhspan(-5, 0, color="#fbe9e9", lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        for si, spd in enumerate(SPEED_ORDER):
            s = g[g.target_kmh == spd]
            ax.scatter(s.time_available_s, s.min_clearance_m, s=18, color=CAT[si], marker=MARKERS[si],
                       edgecolor="white", linewidth=0.6, label=f"{spd:g} km/h", zorder=3)
        ax.set_title(CONTROLLER_LABEL[ctl], loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Time available (s)")
    lo = max(-1.5, r.min_clearance_m.min() - 0.2)
    axes[0][0].set_ylim(lo, max(3.0, r.min_clearance_m.max() + 0.2))
    for ax in axes[:, 0]:
        ax.set_ylabel("Min. clearance (m)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3)
    fig.suptitle("Minimum clearance against time available (corridor entry to contact without braking);\n"
                 "shaded = contact; Baseline not shown", x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + 0.25)
    save(fig, out, "F2_clearance_vs_time")


def fig_validation(val_files: list[Path], out: Path):
    if not val_files:
        return
    v = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)
    v = v[v.matched.astype(bool)] if "matched" in v else v
    if v.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    lim = [min(v.min_clearance_unreal_m.min(), v.min_clearance_replay_m.min()) - 0.2,
           max(v.min_clearance_unreal_m.max(), v.min_clearance_replay_m.max()) + 0.2]
    ax.plot(lim, lim, color=AXIS, lw=1)
    ax.axhline(0, color=AXIS, lw=0.6, ls=":")
    ax.axvline(0, color=AXIS, lw=0.6, ls=":")
    # One colour/marker per (scenario, latency), each labelled with its own contact agreement, so a scenario where
    # the replay is biased (e.g. an oncoming person) stays visible instead of disappearing into the pooled number.
    v = v.copy()
    v["fam"] = v.variant.map(family) if "variant" in v else ""
    v["lat"] = v.latency_s.round(3) if "latency_s" in v else np.nan
    groups = list(v.groupby(["fam", "lat"], sort=True))
    marks = ["o", "s", "^", "D", "v", "P"]
    for i, ((fam, L), g) in enumerate(groups):
        ag = int((g.contact_replay.astype(bool) == g.contact_unreal.astype(bool)).sum()) if "contact_replay" in g else 0
        ax.scatter(g.min_clearance_unreal_m, g.min_clearance_replay_m, s=18, color=VAL_COLORS[i % len(VAL_COLORS)],
                   marker=marks[i % len(marks)], edgecolor="white", lw=0.6,
                   label=f"{fam}, {L:g} s: contact agrees {ag}/{len(g)}")
    if len(groups) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.5, ncol=1, frameon=False)
    diff = v.min_clearance_replay_m - v.min_clearance_unreal_m
    agree = int((v.contact_replay.astype(bool) == v.contact_unreal.astype(bool)).sum()) if "contact_replay" in v else None
    ax.set_xlabel("Unreal closed loop: min clearance (m)")
    ax.set_ylabel("Replay: min clearance (m)")
    ax.set_title(f"Replay vs Unreal (n={len(v)}): contact agrees {agree}/{len(v)}, "
                 f"median |diff| {diff.abs().median():.3f} m, max {diff.abs().max():.3f} m", loc="left", fontsize=7)
    save(fig, out, "F7_replay_validation")


# ------------------------------------------------------------------------------------------------
# Paired tests
# ------------------------------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)


def paired_tests(d: pd.DataFrame, rng) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    r = d[d.source == "replay"]
    ctls = [c for c in ["none", "zone", "ttc", "hybrid"] if c in r.controller.unique()]
    pairs = [(a, b) for i, a in enumerate(ctls) for b in ctls[i + 1:] if a != "none" or b == "zone"]
    rows = []
    # "ALL" pools the Batch 1 scenario families only. ApproachAndStop (S12) is a designed stress test of the
    # predictor's false alarms with 90 runs; pooling it would let its size, not the warehouse mix, decide the
    # overall stop comparison. It is reported as its own row, like every family.
    core = r[~r.family.str.startswith("ApproachAndStop")]
    for fam_key, sub in [("ALL", core)] + [(f, g) for f, g in r.groupby("family")]:
        for a, b in pairs:
            A = sub[sub.controller == a].set_index("run_id")
            B = sub[sub.controller == b].set_index("run_id")
            common = A.index.intersection(B.index)
            if len(common) < 2:
                continue
            A, B = A.loc[common], B.loc[common]
            ca, cb = A.contact.astype(bool), B.contact.astype(bool)
            only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
            ua = (A.outcome == "unnecessary_stop") if "outcome" in A else pd.Series(False, index=common)
            ub = (B.outcome == "unnecessary_stop") if "outcome" in B else pd.Series(False, index=common)
            diff = (B.min_clearance_m - A.min_clearance_m).to_numpy(float)
            boots = [np.median(rng.choice(diff, len(diff))) for _ in range(2000)]
            try:
                p_w = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p_w = np.nan
            rows.append({"scenario": fam_key, "A": a, "B": b, "paired_runs": len(common),
                         "contacts_A": int(ca.sum()), "contacts_B": int(cb.sum()),
                         "contact_only_A": only_a, "contact_only_B": only_b,
                         "mcnemar_p_contact": mcnemar_exact(only_a, only_b),
                         "unnecessary_A": int(ua.sum()), "unnecessary_B": int(ub.sum()),
                         "mcnemar_p_unnecessary": mcnemar_exact(int((ua & ~ub).sum()), int((~ua & ub).sum())),
                         "median_clearance_diff_B_minus_A_m": float(np.median(diff)),
                         "boot95_lo_m": float(np.percentile(boots, 2.5)), "boot95_hi_m": float(np.percentile(boots, 97.5)),
                         "wilcoxon_p": p_w})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--det", type=Path, help="detection_metrics.py output folder (detector unreal)")
    p.add_argument("--models", type=Path, nargs="*", default=[], help="detection_metrics.py outputs, one per detector")
    p.add_argument("--timing", type=Path, nargs="*", default=[], help="detector_timing_*.csv from detect_frames.py")
    p.add_argument("--replay", type=Path, nargs="*", default=[], help="replay_results.csv file(s)")
    p.add_argument("--open", type=Path, nargs="*", default=[],
                   help="pedestrian_results.csv of the open-loop batch(es) (for time available in F2)")
    p.add_argument("--closed", type=Path, nargs="*", default=[], help="pedestrian_results.csv of closed-loop batches")
    p.add_argument("--validation", type=Path, nargs="*", default=[], help="replay_validation.csv file(s)")
    p.add_argument("--rate", type=Path, nargs="*", default=[],
                   help="two pedestrian_results.csv, same scenarios + seeds: base rate first, faster rate second (F11)")
    p.add_argument("--depth-replay", type=Path, nargs="*", default=[],
                   help="replay_results.csv per depth source, rendered depth first, then monocular models (F12)")
    p.add_argument("--latency", type=float, default=0.1, help="Replay latency used for the figures (s)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    rng = np.random.default_rng(12345)
    print(f"figures -> {args.out}")
    if args.det:
        fig_recall_vs_distance(args.det, args.out)
        fig_depth_error(args.det, args.out)
    fig_model_tradeoff(args.models, args.timing, args.out)
    d = load_outcomes(args.replay, args.closed, args.latency)
    if not d.empty:
        rates = contact_rate_table(d)
        rates.to_csv(args.out / "stats_contact_rates.csv", index=False)
        fig_contact_rate(d, args.out)
        fig_outcomes(d, args.out)
        stops = stop_rate_table(d)
        if len(stops):
            stops.to_csv(args.out / "stats_unnecessary_stops.csv", index=False)
            fig_stop_rate_by_timing(stops, args.out)
        fig_clearance_vs_time(d, args.open, args.out)
        fig_relative_motion(d, args.out)
        tests = paired_tests(d, rng)
        if len(tests):
            tests.to_csv(args.out / "stats_paired_tests.csv", index=False)
            with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
                print("\nPaired tests (same runs, controller B vs A):")
                print(tests[tests.scenario == "ALL"][["A", "B", "paired_runs", "contacts_A", "contacts_B",
                                                       "mcnemar_p_contact", "unnecessary_A", "unnecessary_B",
                                                       "median_clearance_diff_B_minus_A_m", "wilcoxon_p"]].to_string(index=False))
    fig_validation(args.validation, args.out)
    fig_rate_pairs(args.rate, args.out)
    fig_latency(args.replay, args.out)
    fig_depth_braking(args.depth_replay, args.latency, args.out)


if __name__ == "__main__":
    main()
