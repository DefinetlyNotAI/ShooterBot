import json
import sys
import time
from pathlib import Path

import numpy as np

# noinspection DuplicatedCode
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager

cfg = load_config(str(ROOT / "configs" / "default.yaml"))

# make sure simulation enabled
cfg.debug.simulate_camera = True
cfg.debug.inject_fake_face = True

# create detector manager
dm = DetectorManager(cfg)

# synthetic frame: gray image
h, w = cfg.camera.height, cfg.camera.width
frame = np.full((h, w, 3), 120, dtype=np.uint8)

# run predict
dets = dm.predict(frame)

print(json.dumps({"timestamp": time.time(), "detections": dets}, indent=2))
