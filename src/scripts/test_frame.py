from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# noinspection DuplicatedCode
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    cfg = load_config(str(config_path))

    cfg.debug.simulate_camera = True
    cfg.debug.inject_fake_face = True

    print("Config loaded (simulate_camera=True)")

    detector_manager = DetectorManager(cfg)

    detector_keys = [
        str(key) if key is not None else "unknown"
        for key in detector_manager.detectors
    ]

    print("Detectors:", detector_keys)

    cap = cv2.VideoCapture(0)

    frame: np.ndarray | None = None

    try:
        if cap.isOpened():
            success, captured_frame = cap.read()

            if success and captured_frame is not None:
                frame = captured_frame
    finally:
        cap.release()

    if frame is None:
        width_value = getattr(cfg.camera, "width", 1280)
        height_value = getattr(cfg.camera, "height", 720)

        width = (
            int(width_value)
            if isinstance(width_value, (int, float))
            else 1280
        )
        height = (
            int(height_value)
            if isinstance(height_value, (int, float))
            else 720
        )

        synthetic_frame: np.ndarray = np.full(
            (height, width, 3),
            255,
            dtype=np.uint8,
        )

        print(
            "Using synthetic frame for simulation",
            synthetic_frame.shape,
        )

        frame = synthetic_frame
    else:
        print(
            "Got frame",
            frame.shape,
        )

    detections = detector_manager.predict(frame)

    print(
        "Detections:",
        detections,
    )


if __name__ == "__main__":
    main()
