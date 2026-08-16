import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
import cv2
import numpy as np

cfg = load_config(str(ROOT / "configs" / "default.yaml"))

# enable simulated camera for smoke testing so a fake face is injected
cfg.debug.simulate_camera = True
cfg.debug.inject_fake_face = True
print("Config loaded (simulate_camera=True)")
dm = DetectorManager(cfg)
print("Detectors:", list(dm.detectors.keys()))
cap = cv2.VideoCapture(0)
frame = None
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if not ret:
        frame = None
else:
    frame = None
# if camera not available or failed, synthesize a blank frame with default config resolution
if frame is None:
    w = getattr(cfg.camera, "width", 1280)
    h = getattr(cfg.camera, "height", 720)
    frame = 255 * np.ones((h, w, 3), dtype="uint8")
    print("Using synthetic frame for simulation", frame.shape)
else:
    print("Got frame", frame.shape)
res = dm.predict(frame)
print("Detections:", res)
