from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

try:
    import torch
except ImportError:
    raise SystemExit(
        "Requires PyTorch with CUDA support for this benchmark."
    )

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scripts.paths import model_path
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)

MODEL_PATH = model_path("yolov8n.pt")

IMG_SIZE = 640
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "This benchmark requires a CUDA-capable PyTorch installation."
        )

    device_index = 0
    device = f"cuda:{device_index}"

    device_name = torch.cuda.get_device_name(device_index)

    print(
        "CUDA device:",
        device_name if device_name else "unknown",
    )

    print(
        "Loading model:",
        str(MODEL_PATH),
    )

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open camera"
            )

        print("Capturing frame...")

        success, captured_frame = cap.read()

        if not success or captured_frame is None:
            raise RuntimeError(
                "Failed to capture frame"
            )

        frame = captured_frame

    finally:
        cap.release()

    frame_height, frame_width = frame.shape[:2]

    print(
        f"Captured frame: "
        f"{frame_width}x{frame_height}"
    )

    print("Warming up...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            device=device,
            half=True,
            verbose=False,
        )

    torch.cuda.synchronize()

    print("Warmup complete")
    print("Benchmarking...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(BENCHMARK_RUNS):
        model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            device=device,
            half=True,
            verbose=False,
        )

    torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed_seconds = end - start

    average_ms: float = (
            elapsed_seconds
            / float(BENCHMARK_RUNS)
            * 1000.0
    )

    fps: float = (
        1000.0 / average_ms
        if average_ms > 0.0
        else 0.0
    )

    print(
        f"Average inference: {average_ms:.2f} ms"
    )

    print(
        f"FPS: {fps:.2f}"
    )


if __name__ == "__main__":
    main()
