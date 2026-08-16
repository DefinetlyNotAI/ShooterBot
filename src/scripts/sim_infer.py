from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# noinspection DuplicatedCode
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager


def json_default(value: Any) -> str:
    if value is None:
        return "None"

    return str(value)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    cfg = load_config(str(config_path))

    cfg.debug.simulate_camera = True
    cfg.debug.inject_fake_face = True

    detector_manager = DetectorManager(cfg)

    frame_height = int(cfg.camera.height)
    frame_width = int(cfg.camera.width)

    frame = np.full(
        (frame_height, frame_width, 3),
        120,
        dtype=np.uint8,
    )

    detections = detector_manager.predict(frame)

    output = {
        "timestamp": time.time(),
        "detections": detections,
    }

    print(
        json.dumps(
            output,
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
