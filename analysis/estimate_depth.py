#!/usr/bin/env python3
"""
Monocular metric depth on the clean frames Unreal saved, written in the same format as Unreal's own depth frames, so
the braking replay and depth_error.py can use it in place of the perfect rendered depth.

Usage:
    python estimate_depth.py <batch folder> [--reps 0 1 2] [--model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf]
                             [--name da2s] [--device 0] [--fp16] [--batch 8]

For each selected run with frames/frame_NNNNN.png it writes depth_<name>/depth_NNNNN.png next to frames/:
uint16, millimeters, 0 = invalid, 640 x 640 (the model output is resized to the frame size). Runs that already have
all their depth files are skipped unless --overwrite.

Then:
    python depth_error.py <batch folder> --name da2s                       (estimate vs Unreal's depth, per person box)
    python replay_braking.py <batch folder> --detector yolo11x_py --depth da2s --controllers zone ttc hybrid --out <...>
        (YOLO11x boxes from Python are identical to Unreal's; only the depth source changes. Runs without a
         depth_da2s folder are skipped, so the replay covers exactly the subset processed here.)

Model: Depth Anything V2, metric fine-tune for indoor scenes (Hypersim, depths up to about 20 m). The model predicts
depth in meters from the image alone; no camera parameters are given to it. Install: pip install transformers
(PyTorch with CUDA as for Ultralytics). The weights download on first use and are cached.

Subset: --reps selects repetitions (default 0 1 2 = 3 per variant, 81 runs for batch 1), --scenarios limits the
scenario families. Full-resolution float depth for all 270 runs would be tens of GB; uint16 PNG is about 0.3-0.6 MB
per frame.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    p.add_argument("--name", default="da2s", help="Depth source name: output folder depth_<name>")
    p.add_argument("--reps", type=int, nargs="*", default=[0, 1, 2], help="Repetitions to process (empty = all)")
    p.add_argument("--scenarios", nargs="*", default=[], help="Only these scenario ids (prefix match), e.g. LateralCrossing")
    p.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    p.add_argument("--fp16", action="store_true", help="Half precision on the GPU (faster; check results once against FP32)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-depth", type=float, default=60.0, help="Values above this are written as 0 (invalid), m")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def frame_files(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for f in sorted((run_dir / "frames").glob("frame_*.png")):
        m = re.match(r"frame_(\d+)\.png$", f.name)
        if m:
            out.append((int(m.group(1)), f))
    return out


def selected(run_dir: Path, args) -> bool:
    try:
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (meta.get("safety") or {}).get("enabled"):
        return False                                   # closed-loop runs have no clean frames anyway
    rep = int(meta.get("repetition", -1))
    if args.reps and rep not in args.reps:
        return False
    if args.scenarios and not any(str(meta.get("scenario_id", "")).startswith(s) for s in args.scenarios):
        return False
    return True


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    root = args.path.resolve()
    runs = sorted(p.parent for p in root.rglob("run_metadata.json"))
    runs = [r for r in runs if selected(r, args) and (r / "frames").is_dir()]
    print(f"{len(runs)} run(s) selected (reps {args.reps or 'all'}"
          f"{', scenarios ' + ' '.join(args.scenarios) if args.scenarios else ''}), model {args.model} -> depth_{args.name}/",
          flush=True)

    t0, n_frames, infer_s = time.time(), 0, 0.0
    for k, rd in enumerate(runs, 1):
        files = frame_files(rd)
        out_dir = rd / f"depth_{args.name}"
        todo = [(i, f) for i, f in files if args.overwrite or not (out_dir / f"depth_{i:05d}.png").exists()]
        if not todo:
            print(f"  [{k:>{len(str(len(runs)))}}/{len(runs)}] {rd.name}: reused ({len(files)} frames)", flush=True)
            continue
        out_dir.mkdir(exist_ok=True)
        import cv2
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            imgs = [Image.open(f).convert("RGB") for _, f in chunk]
            sizes = [(im.height, im.width) for im in imgs]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            ts = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_s += time.time() - ts
            pred = outputs.predicted_depth.float()                         # (B, h, w) meters at model resolution
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=sizes[0], mode="bilinear",
                                                   align_corners=False).squeeze(1)
            for (fi, _), dt in zip(chunk, pred):
                d = dt.cpu().numpy()                                        # meters, frame resolution
                d[~np.isfinite(d) | (d <= 0) | (d > args.max_depth)] = 0.0
                cv2.imwrite(str(out_dir / f"depth_{fi:05d}.png"), np.round(d * 1000.0).astype(np.uint16))
            n_frames += len(chunk)
        el = time.time() - t0
        eta = el / k * (len(runs) - k)
        print(f"  [{k:>{len(str(len(runs)))}}/{len(runs)}] {100 * k / len(runs):3.0f}%  {_fmt_s(el)} elapsed, "
              f"~{_fmt_s(eta)} left  {rd.name}: {len(todo)} frames", flush=True)

    if n_frames:
        print(f"\n{n_frames} frames, model time {1000 * infer_s / n_frames:.1f} ms/frame "
              f"(batch {args.batch}, {'FP16' if dtype == torch.float16 else 'FP32'}, {device})")
    (root / f"depth_{args.name}_info.json").write_text(json.dumps(
        {"model": args.model, "name": args.name, "reps": args.reps, "scenarios": args.scenarios,
         "fp16": dtype == torch.float16, "frames": n_frames,
         "model_ms_per_frame": 1000 * infer_s / n_frames if n_frames else None}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
