"""Benchmark every locally installed YOLO model on one CUDA camera frame."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.paths import models_dir
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)

import cv2

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Requires PyTorch and Ultralytics for CUDA benchmarks.") from exc

IMG_SIZE = 640
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def capture_frame() -> object:
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Failed to capture frame")
        return frame
    finally:
        cap.release()


def benchmark_model(model_path: Path, frame: object, runs: int) -> None:
    device = "cuda:0"
    print("Loading model:", model_path.name)
    model = YOLO(str(model_path))
    for _ in range(WARMUP_RUNS):
        model.predict(source=frame, imgsz=IMG_SIZE, device=device, half=True, verbose=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(source=frame, imgsz=IMG_SIZE, device=device, half=True, verbose=False)
    torch.cuda.synchronize()
    average_ms = (time.perf_counter() - start) / float(runs) * 1000.0
    fps = 1000.0 / average_ms if average_ms > 0.0 else 0.0
    print(f"{model_path.name}: {average_ms:.2f} ms | {fps:.2f} FPS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=positive_int, default=BENCHMARK_RUNS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires CUDA.")
    model_paths = sorted(models_dir().glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError("No .pt models were found in root models/.")
    print("CUDA device:", torch.cuda.get_device_name(0) or "unknown")
    print("Models to benchmark:", ", ".join(path.name for path in model_paths))
    frame = capture_frame()
    print("Captured frame:", f"{frame.shape[1]}x{frame.shape[0]}")
    failed: list[str] = []
    for model_path in model_paths:
        try:
            benchmark_model(model_path, frame, args.runs)
        except Exception as exc:
            failed.append(model_path.name)
            print(f"{model_path.name}: benchmark failed ({exc})")
    if failed:
        raise RuntimeError("Models that failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
