#!/usr/bin/env python3
"""
Write the class labels of every YOLO11 ONNX export in this folder to <model>.labels.json,
in the format {"labels": {"0": "person", "1": "bicycle", ...}}, and check that all models
share the same labels.

Usage (from the folder with the .onnx files):
    python labels.py
"""
import glob
import json
from pathlib import Path

from ultralytics import YOLO

files = sorted(glob.glob("yolo11*.onnx"))
if not files:
    raise SystemExit("No yolo11*.onnx files in this folder")

reference = None
for f in files:
    names = YOLO(f, task="detect").names
    out = Path(f).with_suffix(".labels.json")
    with open(out, "w") as labels_file:
        json.dump({"labels": names}, labels_file)
    same = "" if reference is None else ("  same as " + files[0] if names == reference else "  DIFFERENT from " + files[0])
    print(f"{f}: {len(names)} classes, class 0 = {names[0]!r} -> {out.name}{same}")
    reference = reference or names
