from __future__ import annotations

import json
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
from src.serial_comm import SerialInterface
from src.scripts.paths import files_dir
from src.tracker import Tracker
from src.visualization import (
    draw_center_ui,
    draw_detection,
    draw_top_left_panel,
    draw_track,
    show_info,
)


def safe_str(value: Any) -> str:
    if value is None:
        return "None"

    return str(value)


def get_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"cuda ({safe_str(device_name)})"

    except Exception:
        pass

    return "cpu"


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    cfg = load_config(str(config_path))

    cfg.debug.simulate_camera = True
    cfg.debug.inject_fake_face = True

    out_dir = files_dir()

    img_path = out_dir / f"sim_output_{int(time.time())}.png"

    detector_manager = DetectorManager(cfg)

    tracker = Tracker(
        max_lost=cfg.tracking.max_lost,
        iou_threshold=cfg.tracking.iou_threshold,
        track_only=getattr(cfg.tracking, "track_only", None),
        use_kalman=getattr(cfg.tracking, "use_kalman", True),
    )

    serial = SerialInterface(simulation=True)

    frame_width = int(cfg.camera.width)
    frame_height = int(cfg.camera.height)

    frame = np.full(
        (frame_height, frame_width, 3),
        120,
        dtype=np.uint8,
    )

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

    start_time = time.time()

    detections = detector_manager.predict(frame)

    end_time = time.time()

    tracks = tracker.update(
        detections,
        end_time,
    )

    primary = (
        max(
            tracks,
            key=lambda track: float(track.confidence),
        )
        if tracks
        else None
    )

    for detection in detections:
        bbox_value = detection.get("bbox")

        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            continue

        # noinspection DuplicatedCode
        bbox: tuple[int, int, int, int] = (
            int(bbox_value[0]),
            int(bbox_value[1]),
            int(bbox_value[2]),
            int(bbox_value[3]),
        )

        class_name = detection.get("class_name")
        label_value = detection.get("label")

        label = (
            safe_str(class_name)
            if class_name is not None
            else safe_str(label_value)
            if label_value is not None
            else ""
        )

        confidence_value = detection.get("confidence", 0.0)

        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = 0.0

        draw_detection(
            frame,
            bbox,
            label=label,
            confidence=confidence,
            color=(200, 200, 200),
        )

    tracked_info: dict[str, Any] | None = None

    if primary is not None:
        draw_track(
            frame,
            primary,
            color=(0, 180, 80),
        )

        tracked_info = {
            "class_name": "face",
            "confidence": float(primary.confidence),
            "bbox": list(primary.bbox),
        }

    track_only = getattr(
        cfg.tracking,
        "track_only",
        None,
    )

    draw_top_left_panel(
        frame,
        tracked_info,
        looking_for=track_only,
    )

    device = get_device_name()

    inference_time_ms = float(
        (end_time - start_time) * 1000.0
    )

    show_info(
        frame,
        fps=30.0,
        inference_time_ms=inference_time_ms,
        device=device,
    )

    draw_center_ui(
        frame,
        serial_center=None,
    )

    cv2.imwrite(
        str(img_path),
        frame,
    )

    if primary is not None:
        image_height, image_width = frame.shape[:2]

        x1, y1, x2, y2 = primary.bbox

        center_x = (float(x1) + float(x2)) / 2.0
        center_y = (float(y1) + float(y2)) / 2.0

        normalized_center = [
            center_x / float(image_width),
            center_y / float(image_height),
        ]

        velocity = [
            float(primary.velocity[0]),
            float(primary.velocity[1]),
        ]

        def on_receive(data: bytes) -> None:
            text = data.decode(
                "utf-8",
                errors="replace",
            )

            print(
                "[SERIAL ECHO]",
                text,
            )

        serial.set_receive_callback(on_receive)

        serial.start()

        try:
            serial.send_telemetry(
                primary.id,
                "face",
                float(primary.confidence),
                normalized_center,
                velocity,
                end_time,
            )

            time.sleep(0.05)

        finally:
            serial.stop()

    output = {
        "timestamp": end_time,
        "detections": detections,
    }

    print(
        json.dumps(
            output,
            indent=2,
            default=safe_str,
        )
    )

    print(
        "saved image:",
        str(img_path),
    )


if __name__ == "__main__":
    main()
