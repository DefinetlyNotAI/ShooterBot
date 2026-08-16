import sys
import time
from pathlib import Path

import cv2

try:
    import torch
except Exception:
    exit("Requires CUDA support for this test - Needs torch installed")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "yolov8n.pt"

model = YOLO(MODEL_PATH)
model.to("cuda")

# Enable FP16
if hasattr(model, "model"):
    model.model.half()

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Capturing frame...")

ret, frame = cap.read()

cap.release()

if not ret:
    raise RuntimeError("Failed to capture frame")

print(f"Captured frame: {frame.shape}")

IMG_SIZE = 640

# Warmup
print("Warming up...")

for _ in range(10):
    model(source=frame, imgsz=IMG_SIZE, device="cuda", verbose=False)

torch.cuda.synchronize()

print("Warmup complete")

# Benchmark
print("Benchmarking...")

runs = 100

torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(runs):
    model(source=frame, imgsz=IMG_SIZE, device="cuda", verbose=False)

torch.cuda.synchronize()
end = time.perf_counter()

ms = ((end - start) / runs) * 1000
fps = 1000 / ms

print(f"Average inference: {ms:.2f} ms")
print(f"FPS: {fps:.2f}")
