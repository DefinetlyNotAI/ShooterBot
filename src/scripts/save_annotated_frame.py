from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.scripts.paths import files_dir
from src.scripts.runtime import configure_script_output
from src.visualization import draw_detection

print = configure_script_output(__name__)


def safe_str(value: Any, default: str = "obj") -> str:
    if value is None:
        return default

    return str(value)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            "Configuration is missing. Run the installer first: "
            "python -m src.cli.installer"
        )

    cfg = load_config(str(config_path))

    cfg.debug.simulate_camera = True
    cfg.debug.inject_fake_face = True

    # Force face detection every frame for visible output.
    cfg.inference.face_model_interval = 0.0

    detector_manager = DetectorManager(cfg)

    frame_width = int(getattr(cfg.camera, "width", 640))
    frame_height = int(getattr(cfg.camera, "height", 360))

    frame = np.full(
        (frame_height, frame_width, 3),
        255,
        dtype=np.uint8,
    )

    detections = detector_manager.predict(frame)

    print("Detections:", detections)

    for detection in detections:
        bbox_value = detection.get("bbox")

        if (
                not isinstance(bbox_value, (list, tuple))
                or len(bbox_value) != 4
        ):
            continue

        # noinspection DuplicatedCode
        bbox: tuple[int, int, int, int] = (
            int(bbox_value[0]),
            int(bbox_value[1]),
            int(bbox_value[2]),
            int(bbox_value[3]),
        )

        class_name = detection.get("class_name")
        fallback_label = detection.get("label")

        label = safe_str(
            class_name
            if class_name is not None
            else fallback_label,
        )

        confidence_value = detection.get("confidence")

        if isinstance(confidence_value, (int, float)):
            confidence = float(confidence_value)
        else:
            confidence = 0.0

        draw_detection(
            frame,
            bbox,
            label=label,
            confidence=confidence,
        )

    out_dir = files_dir()

    out_path = out_dir / f"annotated_{int(time.time())}.png"

    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(
            f"Failed to save annotated frame to {out_path}"
        )

    print(
        "Saved annotated frame to",
        str(out_path),
    )


if __name__ == "__main__":
    main()
