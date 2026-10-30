#!/usr/bin/env python3
"""
Run an Ultralytics YOLO model on the clean frames Unreal saved, and write detections in the format the evaluation
scripts read, so other model sizes go through exactly the same metrics and braking replay as Unreal's YOLO11x.

Usage:
    python detect_frames.py <EvalRuns, batch or run folder> --model yolo11n.pt [--name yolo11n] [--device 0]
    python detect_frames.py <folder> --model yolo11x.pt --name yolo11x_py --compare-unreal      (pipeline check)

For every run folder with frames/frame_NNNNN.png it writes detections_<name>.csv next to detections.csv:
    frame_index, timestamp_s, class_id, confidence, center_x, center_y, width, height   (pixels in the 640x640 frame)
Then:
    python detection_metrics.py <folder> --detector <name>
    python replay_braking.py <folder> --detector <name> --depth ideal --controllers zone ttc hybrid

Settings match Unreal's inference (run_metadata.json): 640x640 input, confidence 0.25, NMS IoU 0.70, person class
only. The frames are already 640x640, so no letterboxing happens.

PIPELINE CHECK (do this first): run yolo11x.pt with --compare-unreal on a few runs. Each Python box is matched to
Unreal's box in the same frame (IoU >= 0.5). If Python and Unreal agree (almost all boxes matched, mean IoU > 0.95,
confidence differences of a few hundredths), the model-size comparison is fair. If they disagree, differences between
n/s/m and Unreal's x would mix model size with pipeline differences (preprocessing, FP16 vs FP32, NMS), and the fair
comparison is n/s/m/x ALL run in Python.

Speed: per-image latency is measured separately on --timing-frames frames with batch size 1 after a warm-up, and
written to <out>/detector_timing_<name>.csv (model-only time on this PC; not a Jetson number).

Requires: ultralytics (pip install ultralytics; with a CUDA build of torch for GPU), pandas, numpy.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd



def _fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def progress(i: int, n: int, label: str, t0: float, newline: bool = False) -> None:
    """One status line: [ i/n] pct  label  elapsed / remaining. Overwrites itself unless newline=True."""
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0.0
    w = len(str(n))
    line = f"  [{i:>{w}}/{n}] {100 * i / max(1, n):3.0f}%  {label[:48]:<48}  {_fmt_s(el)} elapsed, ~{_fmt_s(eta)} left"
    if newline:                                     # full label, nothing truncated
        print(f"  [{i:>{w}}/{n}] {100 * i / max(1, n):3.0f}%  {_fmt_s(el)} elapsed, ~{_fmt_s(eta)} left  {label}", flush=True)
    else:
        print("\r" + line, end="\n" if i >= n else "", flush=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--model", required=True, help="Weights, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11x.pt")
    p.add_argument("--name", default=None, help="Detector name for the output file (default: weights file stem)")
    p.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu' for CPU (default: Ultralytics picks)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument("--batch", type=int, default=16, help="Images per forward pass for the bulk run")
    p.add_argument("--overwrite", action="store_true", help="Redo runs that already have detections_<name>.csv")
    p.add_argument("--compare-unreal", action="store_true", help="Compare with Unreal's detections.csv per run")
    p.add_argument("--timing-frames", type=int, default=200, help="Frames for the batch-1 latency measurement (0 = off)")
    p.add_argument("--out", type=Path, default=None, help="Where timing / comparison summaries go (default <path>/analysis_det)")
    return p.parse_args()


def frame_files(run_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for f in sorted((run_dir / "frames").glob("frame_*.png")):
        m = re.match(r"frame_(\d+)\.png$", f.name)
        if m:
            files.append((int(m.group(1)), f))
    return files


def boxes_to_rows(result, frame_index: int, timestamp: float | None) -> list[dict]:
    rows = []
    b = result.boxes
    if b is None or len(b) == 0:
        return rows
    xywh = b.xywh.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    for (cx, cy, w, h), c, k in zip(xywh, conf, cls):
        if k != 0:
            continue
        rows.append({"frame_index": frame_index, "timestamp_s": timestamp, "class_id": 0, "confidence": float(c),
                     "center_x": float(cx), "center_y": float(cy), "width": float(w), "height": float(h)})
    return rows


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def compare(py: pd.DataFrame, ue: pd.DataFrame, run_key: str) -> dict:
    """Greedy one-to-one matching per frame at IoU >= 0.5 between Python and Unreal boxes."""
    if "class_id" in ue:
        ue = ue[ue.class_id == 0]
    matched, ious, dconf = 0, [], []
    frames = sorted(set(py.frame_index) | set(ue.frame_index))
    for f in frames:
        a = py[py.frame_index == f]
        b = ue[ue.frame_index == f]
        m = iou_matrix(a[["center_x", "center_y", "width", "height"]].to_numpy(float),
                       b[["center_x", "center_y", "width", "height"]].to_numpy(float))
        used_a, used_b = set(), set()
        for i, j in sorted(((i, j) for i in range(m.shape[0]) for j in range(m.shape[1]) if m[i, j] >= 0.5),
                           key=lambda ij: -m[ij]):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += 1
            ious.append(m[i, j])
            dconf.append(float(a.confidence.iloc[i]) - float(b.confidence.iloc[j]))
    return {"run_key": run_key, "python_boxes": len(py), "unreal_boxes": len(ue), "matched": matched,
            "python_only": len(py) - matched, "unreal_only": len(ue) - matched,
            "mean_iou_matched": float(np.mean(ious)) if ious else np.nan,
            "mean_abs_conf_diff": float(np.mean(np.abs(dconf))) if dconf else np.nan,
            "mean_conf_diff_py_minus_ue": float(np.mean(dconf)) if dconf else np.nan}


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    name = args.name or Path(args.model).stem
    model = YOLO(args.model)
    root = args.path.resolve()
    out = (args.out or root / "analysis_det").resolve()
    run_dirs = sorted(p.parent for p in root.rglob("run_metadata.json"))
    common = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, classes=[0], verbose=False, half=args.half)
    if args.device is not None:
        common["device"] = args.device

    comparisons, done, skipped, timing_pool = [], 0, 0, []
    print(f"{len(run_dirs)} run folder(s) under {root}, model {args.model} -> detections_{name}.csv")
    t0 = time.time()
    for k, rd in enumerate(run_dirs, 1):
        files = frame_files(rd)
        target = rd / f"detections_{name}.csv"
        if not files:
            skipped += 1
            progress(k, len(run_dirs), f"{rd.name}: no frames/, skipped", t0, newline=True)
            continue
        try:
            meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
            if int(meta.get("image_width", 640)) != args.imgsz or int(meta.get("image_height", 640)) != args.imgsz:
                print(f"  ! {rd.name}: frames are {meta.get('image_width')}x{meta.get('image_height')}, not {args.imgsz}")
        except (OSError, ValueError):
            pass
        ts = {}
        fm = rd / "frame_metrics.csv"
        if fm.exists():
            m = pd.read_csv(fm, usecols=["frame_index", "timestamp_s"])
            ts = dict(zip(m.frame_index, m.timestamp_s))
        if target.exists() and not args.overwrite:
            det = pd.read_csv(target)
            status = f"reused ({len(det)} boxes)"
        else:
            rows = []
            for i in range(0, len(files), args.batch):
                chunk = files[i:i + args.batch]
                results = model.predict([str(p) for _, p in chunk], **common)
                for (fi, _), r in zip(chunk, results):
                    rows += boxes_to_rows(r, fi, ts.get(fi))
            det = pd.DataFrame(rows, columns=["frame_index", "timestamp_s", "class_id", "confidence",
                                              "center_x", "center_y", "width", "height"])
            det.to_csv(target, index=False)
            done += 1
            status = f"{len(files)} frames, {len(det)} boxes"
        timing_pool += [p for _, p in files[:: max(1, len(files) // 10)]]
        if args.compare_unreal and (rd / "detections.csv").exists():
            ue = pd.read_csv(rd / "detections.csv")
            c = compare(det, ue, rd.relative_to(root).as_posix() if rd != root else rd.name)
            comparisons.append(c)
            status += f", matched {c['matched']}/{c['unreal_boxes']} Unreal"
        progress(k, len(run_dirs), f"{rd.name}: {status}", t0, newline=True)

    out.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} name={name}: {done} run(s) detected, {len(run_dirs) - done - skipped} reused, "
          f"{skipped} without frames/ (closed-loop batches do not save frames)")

    # ---- latency, batch size 1 ----
    if args.timing_frames > 0 and timing_pool:
        sample = timing_pool[: args.timing_frames]
        print(f"\nMeasuring latency on {len(sample)} frames (batch size 1, after 10 warm-up frames)...", flush=True)
        for p in sample[:10]:                                   # warm-up
            model.predict(str(p), **common)
        ms = []
        for p in sample:
            r = model.predict(str(p), **common)[0]
            sp = r.speed                                        # ms: preprocess, inference, postprocess
            ms.append((sp.get("preprocess", np.nan), sp.get("inference", np.nan), sp.get("postprocess", np.nan)))
        ms = np.array(ms, float)
        tdf = pd.DataFrame([{"detector": name, "model": args.model, "half": args.half, "frames": len(ms),
                             "preprocess_ms_mean": ms[:, 0].mean(), "inference_ms_mean": ms[:, 1].mean(),
                             "inference_ms_p95": np.percentile(ms[:, 1], 95), "postprocess_ms_mean": ms[:, 2].mean(),
                             "total_ms_mean": ms.sum(axis=1).mean()}])
        tdf.to_csv(out / f"detector_timing_{name}.csv", index=False)
        print(f"latency (batch 1, {len(ms)} frames): inference {ms[:, 1].mean():.1f} ms mean, "
              f"{np.percentile(ms[:, 1], 95):.1f} ms p95; total incl. pre/post {ms.sum(axis=1).mean():.1f} ms")

    # ---- pipeline check against Unreal ----
    if comparisons:
        c = pd.DataFrame(comparisons)
        c.to_csv(out / f"compare_unreal_{name}.csv", index=False)
        tot_py, tot_ue, tot_m = c.python_boxes.sum(), c.unreal_boxes.sum(), c.matched.sum()
        w = c.matched.clip(lower=1)
        print(f"\nPython {name} vs Unreal detections.csv over {len(c)} run(s):")
        print(f"  boxes: python {tot_py}, unreal {tot_ue}, matched {tot_m} "
              f"({tot_m / max(1, tot_ue):.1%} of Unreal's, {tot_m / max(1, tot_py):.1%} of Python's)")
        print(f"  matched boxes: mean IoU {np.average(c.mean_iou_matched.fillna(0), weights=w):.3f}, "
              f"mean |conf difference| {np.average(c.mean_abs_conf_diff.fillna(0), weights=w):.3f}, "
              f"mean (python - unreal) {np.average(c.mean_conf_diff_py_minus_ue.fillna(0), weights=w):+.3f}")
        print("  Same model (yolo11x) should give >98% matched, IoU > 0.95, |conf diff| < 0.03; otherwise compare all "
              "model sizes in Python.")


if __name__ == "__main__":
    main()
