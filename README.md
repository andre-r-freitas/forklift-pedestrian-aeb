# Camera-Based Pedestrian Detection and Emergency Braking for Forklifts

Code and run data of the MSc dissertation *Simulation-Based Evaluation of Camera-Based Pedestrian Detection and
Emergency Braking for Forklifts* (André Rodrigues Freitas, University of Minho, 2026).

A forklift drives through a simulated factory in Unreal Engine 5.6 while pedestrians follow scripted scenarios. A
pre-trained YOLO11 detector runs on the forklift's camera, the distance to each detected person is read from depth,
and a braking controller decides whether to stop. Every frame is logged with exact ground truth (person pixels,
visible fraction, depth, position), so detection, distance estimation and braking can be measured. An offline replay
in Python re-applies other braking controllers, detectors, depth sources and latencies to the recorded open-loop runs.

Thesis: *[link to RepositóriUM after the deposit]*

## Contents

```
unreal/     Unreal Engine 5.6 C++ module: scenario manager, forklift route and dynamics test, pedestrians,
            YOLO inference actor, safety brake controller, ground truth and run logging
analysis/   Python: detection metrics, run evaluation, offline braking replay, tracking, sweeps, figures
models/     YOLO11 ONNX export and checks (the weights themselves are not included)
data/       Per-run CSV/JSON output of every batch used in the thesis (frames and depth maps not included)
```

### Not included

- **The factory level, Blueprints and 3D assets.** The environment, forklift and character assets come from
  third-party packs whose licenses do not allow redistribution. The C++ module builds on its own, but running the
  scenarios needs a level with a forklift, pedestrians and a camera set up as described in Chapters 3 and 4 of the
  thesis.
- **Camera frames, depth maps and ground-truth images** (about 25 GB for batch 1 alone). All figures and tables in the
  thesis can be regenerated from the CSV/JSON files in `data/`, except the few that show individual frames.
- **Model weights.** See [Models](#models).

The Unreal project is named `MeshEditor` for historical reasons: it began as a project for editing the forklift
meshes and grew into the full simulation.

## Unreal Engine module

Requirements: Unreal Engine 5.6, Visual Studio 2022 with the *Game development with C++* workload, and the engine
plugins Neural Network Engine (runtime `NNERuntimeORTDml`), Chaos Vehicles and StateTree.

1. Copy `unreal/` into a folder of your choice.
2. Right-click `MeshEditor.uproject` > *Generate Visual Studio project files*.
3. Open the solution and build the `MeshEditorEditor` target (Development Editor, Win64).

Runs are written to `Saved/EvalRuns/<BatchId>/<RunId>/`, one folder per run with `run_metadata.json`, the
per-frame CSV files and (optionally) the frames. The format is described in Appendix C of the thesis.

## Python analysis

Python 3.13 was used. Install PyTorch with CUDA first (see [pytorch.org](https://pytorch.org)), then:

```
pip install -r requirements.txt
```

Every script documents its inputs and options in its docstring (`python <script>.py --help`). The main steps:

| Script | Purpose |
|---|---|
| `evaluate_runs.py` | Per-run outcomes: contact, minimum clearance, stop distance, time available |
| `detection_metrics.py` | Recall, precision, AP<sub>50</sub>, MR<sup>-2</sup> and recall per distance / visibility bin |
| `replay_braking.py` | Offline replay of the zone, predictive and hybrid controllers on open-loop runs |
| `tracking.py` | Kalman-filter tracker and time-to-collision rule used by the predictive controller |
| `run_sweep.py`, `sweep_figure.py` | Corridor-width and safety-margin sweep, and its trade-off figure |
| `detect_frames.py` | Runs other YOLO11 sizes on the saved frames (needs the frames) |
| `estimate_depth.py`, `depth_error.py` | Depth Anything V2 depth on the saved frames and its error (needs the frames) |
| `compare_repetitions.py` | Checks that repetitions of a run render identical frames (needs the frames) |
| `make_figures.py` | Figures and statistical tests of the results chapter |
| `plot_*.py`, `depth_frames_figure.py` | Single-frame illustrations (need the frames) |

## Models

The detector is YOLO11x from Ultralytics, exported to ONNX (opset 18, 640 x 640, batch 1, FP32, no NMS). From
`models/`:

```
python export_onnx.py          # downloads the .pt weights and exports all five sizes
python check_onnx_exports.py   # checks input/output shapes and opset
python labels.py               # writes <model>.labels.json next to each export
```

## Data

| Folder | Batch | Type | Scenarios | Runs |
|---|---|---|---|---|
| `data/batch1` | 1 | Open loop | S1–S6 | 270 |
| `data/batch1b` | 1b | Open loop | S12 | 90 |
| `data/batch2` | 2 | Closed loop, latency 0.1 s | S2+S3, S5 | 90 |
| `data/batch3` | 3 | Closed loop, latency 0.2 s | S2+S3 | 27 |
| `data/batch4` | 4 | Open loop | S8, S9, S11 | 70 |
| `data/batch4_s7` | 4 | Open loop | S7a, S7b | 9 |
| `data/batch5` | 5 | Closed loop, latency 0.1 s | S9 | 15 |
| `data/batch6` | 6 | Open loop | S10 | 20 |
| `data/batch7` | 7 | Closed loop, 10 Hz | S2+S3, S5 | 60 |
| `data/pilot_walled_section` | – | Closed loop, single run | S1 at 15 km/h in the walled section | 1 |
| `data/sweep` | – | Replay outputs of the controller sweep | | |

## Third-party software

This repository is MIT-licensed (see `LICENSE`). It does not include third-party code or weights, but it uses:

| Component | License |
|---|---|
| Ultralytics / YOLO11 | AGPL-3.0 |
| Depth Anything V2 Small | Apache-2.0 |
| Depth Anything V2 Large | CC BY-NC 4.0 (non-commercial) |
| Unreal Engine | Unreal Engine EULA |
| PyTorch, NumPy, pandas, SciPy, Matplotlib, OpenCV, Pillow, Transformers, ONNX Runtime | BSD / Apache / MIT-style |

Use of these components is governed by their own licenses.

## Citation

> André Rodrigues Freitas. *Simulation-Based Evaluation of Camera-Based Pedestrian Detection and Emergency Braking for
> Forklifts.* MSc dissertation, University of Minho, 2026.
