import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.visualization import draw_detection
import cv2
import numpy as np
import time
from typing import cast

cfg = load_config(str(ROOT / "configs" / "debug" / "sim_debug.yaml"))

# ensure simulation
cfg.debug.simulate_camera = True
cfg.debug.inject_fake_face = True
# force face every frame for visible output
cfg.inference.face_model_interval = 0.0

dm = DetectorManager(cfg)
# create synthetic frame
w = getattr(cfg.camera, "width", 640)
h = getattr(cfg.camera, "height", 360)
frame = 255 * np.ones((h, w, 3), dtype="uint8")
# run prediction
res = dm.predict(frame)
print("Detections:", res)
# draw detections
for d in res:
    bbox = d.get("bbox")
    if bbox is None:
        continue
    bbox = cast(tuple[int, int, int, int], bbox)

    label = d.get("class_name") or d.get("label") or "obj"
    conf = d.get("confidence", 0.0)
    draw_detection(
        frame,
        bbox,
        label=str(d.get("class_name") or d.get("label") or "obj"),
        confidence=float(d.get("confidence") or 0.0),
    )

# save
out_dir = ROOT / "files"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"annotated_{int(time.time())}.png"
cv2.imwrite(str(out_path), frame)
print("Saved annotated frame to", out_path)
