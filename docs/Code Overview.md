# Code Overview

This project is a modular real-time computer vision pipeline. The main runtime entry point is `src/cli/run.py`, which connects camera capture, inference, tracking, visualization, and serial output.

## Runtime Flow

1. `CameraManager` creates one or more `CameraStream` instances from `src/camera.py`.
2. `DetectorManager` in `src/inference.py` loads one or more YOLO detectors and an optional face detector.
3. `InferenceWorker` runs prediction in a background thread so the UI stays responsive.
4. `Tracker` in `src/tracker.py` assigns stable IDs to detections and keeps short motion history.
5. `src/visualization.py` draws boxes, track labels, panels, and crosshairs on the frame.
6. `SerialInterface` in `src/serial_comm.py` sends the primary track to Arduino-compatible hardware.

## Main Modules

- `src/cli/run.py` installs bootstrap logging before heavy imports, then starts the application, applies CLI overrides, opens the window, and drives the main loop.
- `src/cli/scripts.py` provides the interactive, capability-aware launcher for `src/scripts/` diagnostics.
- `src/config.py` loads YAML config into dataclasses and normalizes model paths under `models/`.
- `src/camera.py` handles real cameras, video files, and synthetic simulation frames.
- `src/inference.py` wraps Ultralytics YOLO, optional face detection, and detection filtering.
- `src/tracker.py` keeps persistent track IDs and a simple velocity estimate.
- `src/serial_comm.py` handles serial output, optional CRC, and simulation mode.
- `src/visualization.py` draws the UI overlay and diagnostic panels.

Runtime logging starts with the project formatter before OpenCV or optional ML
libraries load. Once YAML configuration is available, the logger is replaced
with the configured level, output file, colors, and verbose context. Native
stderr diagnostics are forwarded through `nirt_shooterbot.library`.

## Detection Pipeline

The detector layer supports multiple models:

- Primary YOLO models for general object detection.
- A face-specific YOLO model if configured.
- An OpenCV DNN or Haar cascade face fallback when a YOLO face model is not available.

Detections are filtered by confidence, class allow/deny lists, minimum area, and non-maximum suppression-style IoU pruning. The tracker then consumes the filtered detections and keeps the primary target stable across frames.

## Tracking Behavior

`Tracker` uses IoU matching to connect new detections to existing tracks. If SciPy is installed, it can use Hungarian assignment. Otherwise it falls back to a greedy matcher.
Tracks become eligible for use after they reach the configured `min_hits` threshold, and they are removed when they exceed the configured age or lost-frame limit.

Each track keeps:

- A persistent numeric ID.
- The latest bounding box.
- Confidence and class ID.
- A short center history for drawing trajectories.
- A velocity estimate used for future-point extrapolation.

## Serial Output

The main loop sends telemetry for the primary track only. That telemetry is meant for an external controller, usually an Arduino or similar microcontroller, to turn object coordinates into motion.

The application currently treats the serial link as optional. If serial is disabled or unavailable, it falls back to simulation mode so the UI still runs.
