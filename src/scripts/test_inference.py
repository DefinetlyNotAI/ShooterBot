from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics.engine.results import Results

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from src.scripts.paths import files_dir, model_path
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def safe_str(value: Any) -> str:
    if value is None:
        return "None"

    return str(value)


def main() -> None:
    model_paths = [
        model_path("yolov8n.pt"),
        model_path("yolo26n-face.pt"),
    ]

    loaded: dict[str, YOLO] = {}

    print("Python", sys.version)

    try:
        import ultralytics

        version = getattr(
            ultralytics,
            "__version__",
            None,
        )

        print(
            "ultralytics",
            safe_str(version),
        )

    except Exception as exc:
        print(
            "ultralytics import failed:",
            safe_str(exc),
        )

    for model_file in model_paths:
        if not model_file.exists():
            print(
                "Model",
                str(model_file),
                "not found",
            )
            continue

        try:
            print(
                "Loading model",
                str(model_file),
            )

            model = YOLO(str(model_file))

            loaded[model_file.name] = model

            print(
                "Loaded",
                str(model_file),
            )

        except Exception as exc:
            print(
                "Failed to load",
                str(model_file),
                safe_str(exc),
            )

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
        print("Camera unavailable; using a synthetic 1280x720 frame")
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    frame_height, frame_width = frame.shape[:2]

    print(
        "Captured frame",
        frame_width,
        "x",
        frame_height,
    )

    out_img = frame.copy()

    for name, model in loaded.items():
        print(
            "Running inference with",
            name,
        )

        try:
            start_time = time.perf_counter()

            results = model.predict(frame)

            end_time = time.perf_counter()

            inference_ms = float(
                (end_time - start_time) * 1000.0
            )

            print(
                f"Inference time: {inference_ms:.1f} ms"
            )

            for result in results:
                if not isinstance(result, Results):
                    continue

                boxes = result.boxes

                if boxes is None:
                    continue

                for box in boxes:
                    try:
                        xyxy_tensor = box.xyxy

                        if xyxy_tensor is None or len(xyxy_tensor) == 0:
                            continue

                        coordinates = xyxy_tensor[0]

                        if hasattr(coordinates, "cpu"):
                            coordinates = coordinates.cpu()

                        if hasattr(coordinates, "tolist"):
                            coordinate_values = coordinates.tolist()
                        else:
                            coordinate_values = list(coordinates)

                        if len(coordinate_values) != 4:
                            continue

                        x1 = int(coordinate_values[0])
                        y1 = int(coordinate_values[1])
                        x2 = int(coordinate_values[2])
                        y2 = int(coordinate_values[3])

                        confidence = 0.0

                        if box.conf is not None and len(box.conf) > 0:
                            confidence_value = box.conf[0]

                            if hasattr(confidence_value, "item"):
                                confidence = float(
                                    confidence_value.item()
                                )
                            elif isinstance(
                                    confidence_value,
                                    (int, float),
                            ):
                                confidence = float(
                                    confidence_value
                                )

                        class_id = -1

                        if box.cls is not None and len(box.cls) > 0:
                            class_value = box.cls[0]

                            if hasattr(class_value, "item"):
                                class_id = int(
                                    class_value.item()
                                )
                            elif isinstance(
                                    class_value,
                                    (int, float),
                            ):
                                class_id = int(
                                    class_value
                                )

                        bbox = [
                            x1,
                            y1,
                            x2,
                            y2,
                        ]

                        print(
                            f" {name}: "
                            f"cls={class_id} "
                            f"conf={confidence:.2f} "
                            f"bbox={bbox}"
                        )

                        cv2.rectangle(
                            out_img,
                            (x1, y1),
                            (x2, y2),
                            (10, 200, 10),
                            2,
                        )

                        cv2.putText(
                            out_img,
                            f"{name} {confidence:.2f}",
                            (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                    except Exception as exc:
                        print(
                            "box parse error:",
                            safe_str(exc),
                        )

        except Exception as exc:
            print(
                "Inference failed for",
                name,
                safe_str(exc),
            )

    out_dir = files_dir()

    out_path = (
            out_dir
            / f"inference_{int(time.time())}.jpg"
    )

    if not cv2.imwrite(
            str(out_path),
            out_img,
    ):
        raise RuntimeError(
            f"Failed to write {out_path}"
        )

    print(
        "Wrote",
        str(out_path),
    )

    print("Done")


if __name__ == "__main__":
    main()
