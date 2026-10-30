#!/usr/bin/env python3
"""
Export YOLO11 models to ONNX with the settings used for the model in Unreal Engine:
image size 640, opset 18, batch 1, FP32, static shapes, no NMS in the graph.

Ultralytics downloads missing .pt weights to the current folder and writes each .onnx next to its .pt,
replacing an existing .onnx of the same name.

Usage (from the models folder):
    python export_onnx.py                 # exports all five sizes: n, s, m, l and x
    python export_onnx.py s               # exports only YOLO11s (also accepts yolo11s)
    python export_onnx.py n x             # exports YOLO11n and YOLO11x
Then check the shapes with:
    python check_onnx_exports.py
"""
import sys

from ultralytics import YOLO

ALL_SIZES = ["n", "s", "m", "l", "x"]

# Accept "s" as well as "yolo11s"; no arguments means every size.
sizes = [a.lower().removeprefix("yolo11") for a in sys.argv[1:]] or ALL_SIZES
unknown = [s for s in sizes if s not in ALL_SIZES]
if unknown:
    raise SystemExit(f"Unknown size(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_SIZES)}")

for s in sizes:
    model = YOLO(f"yolo11{s}.pt")
    path = model.export(format="onnx", opset=18, imgsz=640, batch=1, half=False, dynamic=False)
    print(f"yolo11{s}: {path}")
