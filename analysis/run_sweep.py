#!/usr/bin/env python3
"""
Parameter sweep of the braking controllers in the offline replay (calls replay_braking.py once per setting).

The zone and predictive (ttc) controllers share the same two parameters, the corridor half-width and the safety
margin, so both are swept on the same values:
    corridor half-width  1.5, 2.0, 2.5, 3.0 m   (margin 1.0 m)
    safety margin        0.5, 1.0, 1.5, 2.0 m   (half-width 1.5 m)
The default setting (1.5 m, 1.0 m) is shared by both sweeps and run once.

Output: one folder per (set, half-width, margin), e.g. sweep2/batch1_w2.0_m1.0/replay_results.csv.
Settings whose replay_results.csv already exists are skipped, so an interrupted sweep can be restarted.

Usage (from the folder that holds replay_braking.py):
    python run_sweep.py --set batch1=<batch 1 folder> --set s12=<S12 folder> --out sweep2
    python run_sweep.py --set batch1=<...> --set s12=<...> --out sweep2 --margins 1.0     # corridor sweep only
    python run_sweep.py ... -- --detector unreal --depth ideal     # anything after -- goes to replay_braking.py
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=FOLDER",
                    help="A replay input folder and its short name (letters/digits only), repeatable")
    ap.add_argument("--out", type=Path, default=Path("sweep2"))
    ap.add_argument("--widths", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--default-width", type=float, default=1.5)
    ap.add_argument("--default-margin", type=float, default=1.0)
    ap.add_argument("--controllers", nargs="+", default=["zone", "ttc"])
    ap.add_argument("--latency", type=float, default=0.1)
    ap.add_argument("--replay", type=Path, default=Path(__file__).with_name("replay_braking.py"))
    args = ap.parse_args(argv)

    sets = []
    for s in args.set:
        name, _, folder = s.partition("=")
        if not name.isalnum() or not folder:
            ap.error(f"--set must be NAME=FOLDER with an alphanumeric NAME, got {s!r}")
        sets.append((name, Path(folder)))

    settings = [(w, args.default_margin) for w in args.widths]
    settings += [(args.default_width, m) for m in args.margins if (args.default_width, m) not in settings]

    jobs = [(name, folder, w, m) for name, folder in sets for w, m in settings]
    for k, (name, folder, w, m) in enumerate(jobs, 1):
        out = args.out / f"{name}_w{w:.1f}_m{m:.1f}"
        if (out / "replay_results.csv").exists():
            print(f"[{k}/{len(jobs)}] {out} exists, skipped")
            continue
        cmd = [sys.executable, str(args.replay), str(folder), "--out", str(out),
               "--latencies", str(args.latency), "--controllers", *args.controllers,
               "--corridor-half-width", str(w), "--margin", str(m), *extra]
        print(f"[{k}/{len(jobs)}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    print(f"\ndone; summarize with: python sweep_figure.py {args.out}")


if __name__ == "__main__":
    main()
