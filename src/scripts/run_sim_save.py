import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.config import load_config
from src.inference import DetectorManager
from src.serial_comm import SerialInterface
from src.tracker import Tracker
from src.visualization import (
    draw_detection,
    draw_track,
    show_info,
    draw_center_ui,
    draw_top_left_panel,
)

# Load config and enable simulation
cfg = load_config(str(ROOT / "configs" / "default.yaml"))
cfg.debug.simulate_camera = True
cfg.debug.inject_fake_face = True

# Prepare
out_dir = ROOT / "files"
out_dir.mkdir(exist_ok=True)
img_path = out_dir / f"sim_output_{int(time.time())}.png"

# create components
dm = DetectorManager(cfg)
tracker = Tracker(
    max_lost=cfg.tracking.max_lost,
    iou_threshold=cfg.tracking.iou_threshold,
    track_only=getattr(cfg.tracking, "track_only", None),
    use_kalman=getattr(cfg.tracking, "use_kalman", True),
)
serial = SerialInterface(simulation=True)

# Create synthetic frame
w, h = cfg.camera.width, cfg.camera.height
frame = np.full((h, w, 3), 120, dtype=np.uint8)
# optional: draw subtle background
cv2.putText(
    frame,
    "SIMULATION MODE",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (200, 200, 200),
    2,
    cv2.LINE_AA,
)

# run prediction
t0 = time.time()
dets = dm.predict(frame)
t1 = time.time()

# update tracker
tracks = tracker.update(dets, t1)

# choose primary
primary = None
if tracks:
    primary = max(tracks, key=lambda t: t.confidence)

# draw all detections
for d in dets:
    draw_detection(
        frame,
        d["bbox"],
        label=str(d.get("class_name") or d.get("label") or ""),
        confidence=float(d.get("confidence", 0.0)),
        color=(200, 200, 200),
    )

# draw tracks
if primary:
    draw_track(frame, primary, color=(0, 180, 80))
    tracked_info = {
        "class_name": "face",
        "confidence": primary.confidence,
        "bbox": list(primary.bbox),
    }
else:
    tracked_info = None

# draw panels and center UI
draw_top_left_panel(
    frame, tracked_info, looking_for=getattr(cfg.tracking, "track_only", None)
)

try:
    import torch

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
except Exception:
    device = "cpu"

show_info(frame, fps=30.0, inference_time_ms=(t1 - t0) * 1000.0, device=device)
draw_center_ui(frame, serial_center=None)

# save image
cv2.imwrite(str(img_path), frame)

# send telemetry for primary
if primary:
    # noinspection DuplicatedCode
    h, w = frame.shape[:2]
    cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
    cy = (primary.bbox[1] + primary.bbox[3]) / 2.0
    nc = [cx / w, cy / h]
    serial.set_receive_callback(
        lambda b: print("[SERIAL ECHO]", b.decode("utf-8"))
    )
    serial.start()
    serial.send_telemetry(
        primary.id,
        "face",
        primary.confidence,
        nc,
        [primary.velocity[0], primary.velocity[1]],
        t1,
    )
    time.sleep(0.05)
    serial.stop()

# print outputs
print(json.dumps({"timestamp": t1, "detections": dets}, indent=2))
print("saved image:", str(img_path))
