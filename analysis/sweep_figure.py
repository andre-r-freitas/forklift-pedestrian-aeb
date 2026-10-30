#!/usr/bin/env python3
"""
Summary table and trade-off figure of the controller sweep (outputs of run_sweep.py).

Reads <folder>/<set>_w<half-width>_m<margin>/replay_results.csv (the older <set>_w<half-width> folders of the first
corridor sweep are read as margin 1.0 m), and for each controller, set, half-width and margin counts contacts and
unnecessary stops at the given latency. Writes <folder>/sweep_summary.csv and the figure <out>.pdf/.png:
(a) batch 1, contacts against unnecessary stops, one curve per controller and swept parameter (solid: corridor
half-width, labeled; dashed: safety margin, unlabeled, it grows from left to right); (b), (c) S12, unnecessary stops against each parameter. S12 is not pooled with
batch 1, because its number of runs would then decide the overall count of unnecessary stops.

Usage:
    python sweep_figure.py sweep2                       # needs sets named batch1 and s12
    python sweep_figure.py sweep2 --out images/Results/controller_tradeoff
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CTRL = {"zone": ("#2a78d6", "o", "Zone (position only)"),
        "ttc": ("#eb6834", "s", "Predictive (tracking + TTC)"),
        "hybrid": ("#1baf7a", "^", "Hybrid")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "legend.frameon": False})


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load(folder: Path, latency: float) -> pd.DataFrame:
    rows = []
    for d in sorted(folder.iterdir()):
        m = re.fullmatch(r"([A-Za-z0-9]+?)_w([\d.]+)(?:_m([\d.]+))?", d.name)
        f = d / "replay_results.csv"
        if not (d.is_dir() and m and f.exists()):
            continue
        r = pd.read_csv(f)
        r = r[(r.latency_s - latency).abs() < 1e-6]
        for ctrl, g in r.groupby("controller"):
            rows.append({"set": m.group(1), "controller": ctrl, "half_width_m": float(m.group(2)),
                         "margin_m": float(m.group(3) or 1.0), "runs": len(g),
                         "contacts": int(as_bool(g.contact).sum()),
                         "unnecessary_stops": int((g.outcome == "unnecessary_stop").sum())})
    if not rows:
        raise SystemExit(f"No <set>_w<width>[_m<margin>]/replay_results.csv found in {folder}")
    df = pd.DataFrame(rows)
    # the same setting may exist twice (old and new folder names): keep one
    return df.drop_duplicates(["set", "controller", "half_width_m", "margin_m"], keep="last")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--latency", type=float, default=0.1)
    ap.add_argument("--b1", default="batch1", help="Name of the batch 1 set")
    ap.add_argument("--s12", default="s12", help="Name of the S12 set")
    ap.add_argument("--default-width", type=float, default=1.5)
    ap.add_argument("--default-margin", type=float, default=1.0)
    ap.add_argument("--out", default="controller_tradeoff")
    a = ap.parse_args()

    df = load(a.folder, a.latency)
    key = ["controller", "half_width_m", "margin_m"]
    wide = df.pivot_table(index=key, columns="set", values=["runs", "contacts", "unnecessary_stops"], aggfunc="sum")
    wide.columns = [f"{v}_{s}" for v, s in wide.columns]
    table = wide.reset_index()   # S12 is kept separate, never pooled with batch 1 (as in Chapter 6)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(table.to_string(index=False))
    table.to_csv(a.folder / "sweep_summary.csv", index=False)
    print(f"\nsaved {a.folder / 'sweep_summary.csv'}")

    def sweep(t, by):
        """Rows of one controller for the corridor sweep (by='half_width_m') or the margin sweep (by='margin_m')."""
        if by == "half_width_m":
            return t[(t.margin_m - a.default_margin).abs() < 1e-6].sort_values(by)
        return t[(t.half_width_m - a.default_width).abs() < 1e-6].sort_values(by)

    ctrls = [c for c in CTRL if c in set(table.controller)] + sorted(set(table.controller) - set(CTRL))
    has_s12 = f"unnecessary_stops_{a.s12}" in table
    fig = plt.figure(figsize=(6.3, 2.9))
    gs = fig.add_gridspec(1, 3 if has_s12 else 1, width_ratios=[2, 1, 1] if has_s12 else [1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    handles = {}

    # (a) batch 1: contacts against unnecessary stops
    xc, yc = f"unnecessary_stops_{a.b1}", f"contacts_{a.b1}"
    ax.set_title(f"(a) Batch 1 ({int(table[f'runs_{a.b1}'].max())} runs)", fontsize=9, color=INK, loc="left")
    for ctrl in ctrls:
        t = table[table.controller == ctrl]
        color, marker, label = CTRL.get(ctrl, (MUTED, "x", ctrl))
        w = sweep(t, "half_width_m")
        h, = ax.plot(w[xc], w[yc], "-", marker=marker, color=color, ms=5, lw=1.4)
        handles[label] = h
        for _, r in w.iterrows():
            ax.annotate(f"{r.half_width_m:g}", (r[xc], r[yc]), textcoords="offset points", xytext=(3, 4),
                        fontsize=7, color=color)
        m = sweep(t, "margin_m")
        if len(m) > 1:
            ax.plot(m[xc], m[yc], "--", marker=marker, color=color, ms=5, lw=1.1, mfc="white")
    ax.set_xlabel("Unnecessary stops")
    ax.set_ylabel("Contacts")
    ax.set_xlim(-4, table[xc].max() * 1.12 + 4)
    ax.set_ylim(0, table[yc].max() * 1.15 + 1)

    # (b), (c) S12: unnecessary stops against each swept parameter (S12 has no contacts by design)
    if has_s12:
        sc, n12 = f"unnecessary_stops_{a.s12}", int(table[f"runs_{a.s12}"].max())
        axb = fig.add_subplot(gs[1])
        axc = fig.add_subplot(gs[2], sharey=axb)
        for axx, by, ls, title, xl in [(axb, "half_width_m", "-", "(b) S12, width", "Corridor half-width (m)"),
                                       (axc, "margin_m", "--", "(c) S12, margin", "Safety margin (m)")]:
            for ctrl in ctrls:
                color, marker, _ = CTRL.get(ctrl, (MUTED, "x", ctrl))
                r = sweep(table[table.controller == ctrl], by)
                axx.plot(r[by], r[sc], ls, marker=marker, color=color, ms=5, lw=1.3 if ls == "-" else 1.1,
                         mfc=color if ls == "-" else "white")
            axx.set_title(title, fontsize=9, color=INK, loc="left")
            axx.set_xlabel(xl)
            axx.set_xticks(sorted(table[by].unique()))
            axx.set_xticklabels([f"{v:g}" for v in sorted(table[by].unique())])
        axb.set_ylabel(f"Unnecessary stops (of {n12})")
        axb.set_ylim(-4, n12 * 1.08)
        plt.setp(axc.get_yticklabels(), visible=False)

    style = [plt.Line2D([], [], color=INK2, lw=1.4, marker="o", label="solid: corridor half-width (m)"),
             plt.Line2D([], [], color=INK2, lw=1.1, ls="--", marker="o", mfc="white",
                        label="dashed: safety margin (m)")]
    fig.legend(list(handles.values()) + style, list(handles) + [s.get_label() for s in style],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.17))
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
