#!/usr/bin/env python3
"""
Check the YOLO11 ONNX exports in this folder: print the input and output shapes and the opset of each model.

Every model used in the closed loop must have the interface that the inference actor in Unreal Engine expects:
    input   [1, 3, 640, 640]   one RGB image, 640 x 640 pixels (batch, channels, height, width)
    output  [1, 84, 8400]      8400 candidate boxes, each with 4 box values and 80 class scores (no NMS)
    opset   18                 the ONNX version supported by the runtime in Unreal Engine

Usage (from the folder with the .onnx files, after export_onnx.py):
    python check_onnx_exports.py
"""
import glob

import onnx


def shape(tensor):
    """Dimensions of an ONNX graph input or output, e.g. [1, 3, 640, 640] (0 = dynamic dimension)."""
    return [d.dim_value for d in tensor.type.tensor_type.shape.dim]


for f in sorted(glob.glob("yolo11*.onnx")):
    m = onnx.load(f)
    print(f, shape(m.graph.input[0]), shape(m.graph.output[0]), "opset", m.opset_import[0].version)
