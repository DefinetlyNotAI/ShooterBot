"""Entry point to run NIRT ShooterBot (debug mode included)."""

from __future__ import annotations

# ensure package imports resolve when running script directly
import sys
from pathlib import Path

from src.noise_control import configure_library_noise

configure_library_noise()

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.utils import setup_logging, logger
from src.ui import print_banner

# Install the project's formatter before importing libraries that may emit
# diagnostics during import (OpenCV, NumPy, TensorFlow, or MediaPipe). The
# config-specific logger is applied later by load_application_config().
setup_logging()

from src.config import load_config, resolve_config_path
from src.camera import CameraManager, CameraStream
from src.inference import DetectorManager, InferenceWorker
from src.tracker import Tracker
from src.target_queue import TargetQueue
from src.visualization import (
    draw_detection,
    draw_track,
    show_info,
    draw_center_ui,
    draw_top_left_panel,
    draw_tracking_queue,
)
from src.serial_comm import SerialInterface
from src.optional_features import OptionalFeatures, face_appearance
from src.setup_check import check_runtime_setup

import argparse
import json
import logging
import math
import threading
import time

from typing import Any, Dict, List

import cv2


class NIRTShooterBotApplication:
    WINDOW_NAME = "nirt-shooterbot"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.cam_mgr = CameraManager()
        self.detector = DetectorManager(cfg)
        self.inf_worker = InferenceWorker(self.detector)

        self.tracker = self._create_tracker()
        self.optional_features = OptionalFeatures(cfg)
        self.target_queue = TargetQueue(
            cycle_remember=getattr(
                cfg.tracking,
                "cycle_remember",
                True,
            )
        )

        self.serial = self._create_serial()

        self.serial_center: tuple[float, float] | None = None

        self._shot_requested = False
        self._shot_target_id: int | None = None
        self._shot_lock = threading.Lock()

        self.latest_click_tracks: list[
            tuple[int, tuple[float, float, float, float]]
        ] = []

        self.last_results: dict[
            int,
            List[Dict[str, Any]],
        ] = {}

        self.last_time = time.time()
        self.infer_time_ms = 0.0
        self.smooth_pos: list[float] | None = None

        self.track_only = getattr(
            cfg.tracking,
            "track_only",
            None,
        )

        self.loop_logger = logging.getLogger("nirt_shooterbot.loop")
        self.inference_logger = logging.getLogger(
            "nirt_shooterbot.inference"
        )
        self.display_logger = logging.getLogger(
            "nirt_shooterbot.display"
        )
        self.serial_logger = logging.getLogger(
            "nirt_shooterbot.serial"
        )
        logger.info(
            "Runtime components prepared: cameras=%s serial=%s tracking=%s",
            len(cfg.camera.sources),
            "simulation" if self.serial.simulation else "hardware",
            self.track_only or "all classes",
        )

    def _create_tracker(self):
        cfg = self.cfg

        max_age_value = getattr(cfg.tracking, "max_age", None)
        if isinstance(max_age_value, int):
            max_age = max_age_value
        else:
            max_age = int(cfg.tracking.max_lost)

        logger.debug(
            "Creating tracker max_lost=%s max_age=%s min_hits=%s iou=%s",
            cfg.tracking.max_lost,
            max_age,
            getattr(cfg.tracking, "min_hits", 1),
            cfg.tracking.iou_threshold,
        )
        return Tracker(
            max_lost=int(cfg.tracking.max_lost),
            iou_threshold=float(cfg.tracking.iou_threshold),
            track_only=getattr(cfg.tracking, "track_only", None),
            min_hits=int(getattr(cfg.tracking, "min_hits", 1)),
            max_age=max_age,
            class_priority=getattr(cfg.tracking, "class_priority", None),
            remember_faces=bool(
                getattr(cfg.tracking, "remember_faces", False)
            ),
            face_memory_threshold=float(
                getattr(
                    cfg.tracking,
                    "face_memory_threshold",
                    0.78,
                )
            ),
            face_memory_max_age=int(
                getattr(
                    cfg.tracking,
                    "face_memory_max_age",
                    300,
                )
            ),
        )

    def _create_serial(self) -> SerialInterface:
        cfg = self.cfg

        simulation = (
            cfg.serial.simulation
            if getattr(cfg.serial, "enabled", False)
            else True
        )

        logger.info(
            "Preparing serial interface port=%s baudrate=%s mode=%s",
            cfg.serial.port,
            cfg.serial.baudrate,
            "simulation" if simulation else "hardware",
        )
        return SerialInterface(
            port=cfg.serial.port,
            baudrate=cfg.serial.baudrate,
            crc=cfg.serial.crc,
            simulation=simulation,
        )

    def _setup_cameras(self) -> None:
        cfg = self.cfg

        for source in cfg.camera.sources:
            simulate = self._should_simulate_camera(source)
            logger.info(
                "Preparing camera source=%s mode=%s resolution=%sx%s fps=%s",
                source,
                "simulation" if simulate else "hardware",
                cfg.camera.width,
                cfg.camera.height,
                cfg.camera.fps,
            )

            camera = CameraStream(
                source=source,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fps=cfg.camera.fps,
                simulate=simulate,
                sim_video=cfg.debug.simulation_video,
                inject_fake_face=getattr(
                    cfg.debug,
                    "inject_fake_face",
                    False,
                ),
                backend_preference=getattr(
                    cfg.camera,
                    "backend_preference",
                    None,
                ),
            )

            self.cam_mgr.add(camera)

        logger.info("Configured %s camera stream(s)", len(cfg.camera.sources))

    def _should_simulate_camera(self, source) -> bool:
        if self.cfg.debug.simulate_camera:
            return True

        if not isinstance(source, int):
            return False

        try:
            test_cap = cv2.VideoCapture(source)

            try:
                opened = test_cap.isOpened()
            finally:
                test_cap.release()

            if opened:
                logger.debug("Camera %s passed startup availability check", source)
                return False

            logger.warning(
                "Camera %s not available, falling back to simulation.",
                source,
            )

            return True

        except Exception:
            logger.debug(
                "Failed to test camera %s",
                source,
                exc_info=True,
            )
            return True

    def _log_detectors(self) -> None:
        try:
            for name, detector_info in self.detector.detectors.items():
                obj = detector_info.get("obj")

                model_name = (
                    getattr(obj, "model_name", None)
                    if obj is not None
                    else None
                )

                logger.info(
                    "Loaded detector %s: type=%s model=%s priority=%s",
                    name,
                    detector_info.get("type"),
                    model_name,
                    detector_info.get("priority"),
                )

        except Exception:
            logger.debug(
                "Could not list detectors",
                exc_info=True,
            )

    def _setup_serial(self) -> None:
        self.serial_logger.info("Starting serial interface")
        self.serial.set_receive_callback(self._on_receive)
        self.serial.start()

    def _setup_window(self) -> None:
        self.display_logger.info("Creating display window '%s'", self.WINDOW_NAME)
        cv2.namedWindow(
            self.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            self.WINDOW_NAME,
            self._on_mouse,
        )

    def _start(self) -> None:
        logger.info("Starting camera, detector, serial, and display components")
        self._setup_cameras()
        self._log_detectors()

        self.inf_worker.start()
        self.inference_logger.info("Inference worker started")

        self._setup_serial()
        self._setup_window()

    def _ensure_inference_worker(self) -> None:
        try:
            if self.inf_worker.thread_is_alive():
                return

            self.inference_logger.warning(
                "Inference worker thread not alive; restarting"
            )

            self.inf_worker.stop()
            self.inf_worker.start()

        except Exception:
            self.inference_logger.exception(
                "Failed to restart inference worker"
            )

    def _request_shot(
            self,
            target_id: int | None = None,
    ) -> None:
        with self._shot_lock:
            self._shot_requested = True
            self._shot_target_id = target_id
        self.serial_logger.debug("Shot request received for target=%s", target_id)

    def _consume_shot(
            self,
    ) -> tuple[int | None, bool]:
        with self._shot_lock:
            if not self._shot_requested:
                return None, False

            target_id = self._shot_target_id

            self._shot_requested = False
            self._shot_target_id = None

            return target_id, True

    def _on_receive(self, data: bytes) -> None:
        try:
            text = data.decode(
                "utf-8",
                errors="ignore",
            ).strip()
            self.serial_logger.debug("Received serial payload: %s", text)

            if text.upper() in {
                "SHOT",
                "HIT",
                "TRIGGER",
            }:
                self.serial_logger.info("Received serial shot event '%s'", text)
                self._request_shot()
                return

            payload = json.loads(text)

            if (
                    payload.get("shot") is True
                    or payload.get("hit") is True
                    or str(payload.get("event", "")).lower()
                    in {"shot", "hit"}
            ):
                self.serial_logger.info("Received serial shot event payload")
                requested_id = payload.get("id")

                self._request_shot(
                    int(requested_id)
                    if requested_id is not None
                    else None
                )
                return

            self._update_serial_center(payload)

        except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                TypeError,
                ValueError,
        ):
            self.serial_logger.debug(
                "Ignored malformed serial packet",
                exc_info=True,
            )

    def _update_serial_center(
            self,
            payload: dict[str, Any],
    ) -> None:
        nc_payload = payload.get("nc")

        if (
                isinstance(nc_payload, list)
                and len(nc_payload) == 2
        ):
            try:
                self.serial_center = (
                    float(nc_payload[0]),
                    float(nc_payload[1]),
                )
            except (TypeError, ValueError):
                pass

            return

        if "x" not in payload or "y" not in payload:
            return

        try:
            self.serial_center = (
                float(payload["x"]),
                float(payload["y"]),
            )
        except (TypeError, ValueError):
            pass

    def _on_mouse(
            self,
            event: int,
            x: int,
            y: int,
            _flags: int,
            _param: Any,
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.cfg.debug.enabled:
            return

        for track_id, bbox in reversed(
                self.latest_click_tracks
        ):
            x1, y1, x2, y2 = map(int, bbox)

            if x1 <= x <= x2 and y1 <= y <= y2:
                self._request_shot(track_id)
                return

    def _get_detections(
            self,
            camera_index: int,
            frame,
            frame_timestamp: float,
    ) -> tuple[
        List[Dict[str, Any]],
        float,
        bool,
        Any,
    ]:
        fresh_result = False
        result = None

        try:
            self.loop_logger.debug(
                "Main loop frame idx=%s ts=%s",
                camera_index,
                frame_timestamp,
            )

            try:
                self.inf_worker.submit(
                    camera_index,
                    frame,
                    frame_timestamp,
                )

            except Exception:
                self.loop_logger.exception(
                    "Failed to submit frame to inference worker"
                )

            result = self.inf_worker.get(camera_index)

            if result:
                detections = result["detections"]

                self.infer_time_ms = result[
                    "inference_time_ms"
                ]

                self.last_results[
                    camera_index
                ] = detections

                timestamp = result["timestamp"]
                fresh_result = True

            else:
                detections = self.last_results.get(
                    camera_index,
                    [],
                )
                timestamp = time.time()

        except Exception:
            self.loop_logger.exception(
                "Frame inference handling failed"
            )

            detections = []
            timestamp = time.time()

        return (
            detections,
            timestamp,
            fresh_result,
            result,
        )

    def _process_fresh_detections(
            self,
            frame,
            detections: List[Dict[str, Any]],
    ) -> None:
        for detection in detections:
            class_name = detection.get("class_name")

            if not isinstance(class_name, str):
                class_name = detection.get("label")

            if (
                    isinstance(class_name, str)
                    and class_name.lower() == "face"
            ):
                detection["appearance"] = face_appearance(
                    frame,
                    detection.get("bbox"),
                )

        self.optional_features.annotate_emotions(
            frame,
            detections,
        )

    def _publish_detection_dump(
            self,
            camera_index: int,
            detections: List[Dict[str, Any]],
            timestamp: float,
            result: Any,
    ) -> None:
        if not (
                self.cfg.serial.advanced_datapackets
                and result
                and detections
        ):
            return

        events = []

        for detection in detections:
            events.append(
                {
                    "bbox": detection.get("bbox"),
                    "bbox_normalized": detection.get(
                        "bbox_normalized"
                    ),
                    "center": detection.get("center"),
                    "normalized_center": detection.get(
                        "normalized_center"
                    ),
                    "class_id": detection.get("class_id"),
                    "class_name": detection.get(
                        "class_name"
                    ),
                    "confidence": detection.get(
                        "confidence"
                    ),
                    "timestamp": timestamp,
                }
            )

        if not events:
            return

        print(
            json.dumps(
                {
                    "camera_index": camera_index,
                    "timestamp": timestamp,
                    "detections": events,
                }
            )
        )

    def _update_tracking(
            self,
            detections: List[Dict[str, Any]],
            timestamp: float,
    ):
        tracks = self.tracker.update(
            detections,
            timestamp,
        )

        self.target_queue.sync(tracks)

        shot_id, requested = self._consume_shot()

        if requested:
            self.target_queue.mark_shot(
                shot_id
                if shot_id is not None
                else self.target_queue.current_id
            )

        target_id = self.target_queue.select_next()

        tracks_by_id = {
            track.id: track
            for track in tracks
        }

        primary = (
            tracks_by_id.get(target_id)
            if target_id is not None
            else None
        )

        if primary is not None and primary.lost != 0:
            primary = None

        return tracks, primary

    @staticmethod
    def _is_detection_match(
            detection: dict[str, Any],
            track: Any,
            threshold_px: float = 40,
    ) -> bool:
        try:
            bbox = detection["bbox"]

            dcx = (bbox[0] + bbox[2]) / 2.0
            dcy = (bbox[1] + bbox[3]) / 2.0

            tcx = (
                          track.bbox[0] + track.bbox[2]
                  ) / 2.0

            tcy = (
                          track.bbox[1] + track.bbox[3]
                  ) / 2.0

            distance = math.hypot(
                dcx - tcx,
                dcy - tcy,
            )

            return distance <= threshold_px

        except (
                KeyError,
                TypeError,
                IndexError,
        ):
            return False

    def _draw_detections(
            self,
            frame,
            detections: List[Dict[str, Any]],
            primary: Any,
    ) -> None:
        height, width = frame.shape[:2]

        frame_cx = width / 2.0
        frame_cy = height / 2.0

        threshold = getattr(
            self.cfg.visualization,
            "center_threshold_px",
            40,
        )

        real_mode = not self.serial.simulation

        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection.get("class_name")

            if isinstance(class_name, str):
                label = class_name
            else:
                class_id = detection.get("class_id")

                label = (
                    str(class_id)
                    if isinstance(
                        class_id,
                        (int, float),
                    )
                    else ""
                )

            emotion = detection.get("emotion")

            if isinstance(emotion, str) and emotion:
                label = f"{label} | {emotion}"

            if real_mode:
                color = self._get_detection_color(
                    bbox,
                    frame_cx,
                    frame_cy,
                    threshold,
                )
            else:
                color = (200, 200, 200)

                if (
                        primary is not None
                        and self._is_detection_match(detection, primary, threshold_px=50)
                ):
                    color = (20, 150, 255)

            draw_detection(
                frame,
                bbox,
                label=label,
                confidence=detection.get(
                    "confidence",
                    0.0,
                ),
                color=color,
            )

    @staticmethod
    def _get_detection_color(
            bbox,
            frame_cx: float,
            frame_cy: float,
            threshold: float,
    ):
        try:
            dcx = (bbox[0] + bbox[2]) / 2.0
            dcy = (bbox[1] + bbox[3]) / 2.0

            distance = math.hypot(
                dcx - frame_cx,
                dcy - frame_cy,
            )

            if distance <= threshold:
                return 0, 200, 0

        except (
                TypeError,
                IndexError,
        ):
            pass

        return 0, 200, 200

    @staticmethod
    def _draw_hands(
            frame,
            hand_results,
    ) -> None:
        for hand in hand_results:
            points = hand.get("points", [])

            for point in points:
                cv2.circle(
                    frame,
                    point,
                    3,
                    (255, 180, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )

            if not points:
                continue

            hx = min(point[0] for point in points)

            hy = max(
                15,
                min(point[1] for point in points) - 8,
            )

            text = (
                f"{hand.get('handedness', 'hand')}: "
                f"{hand.get('gesture', 'unknown')}"
            )

            cv2.putText(
                frame,
                text,
                (hx, hy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 80),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_secondary_tracks(
            frame,
            tracks,
            primary,
    ) -> None:
        for track in tracks:
            if (
                    primary is None
                    or track.id != primary.id
            ):
                draw_track(
                    frame,
                    track,
                    color=(120, 120, 120),
                )

    def _draw_primary_track(
            self,
            frame,
            primary,
    ) -> dict[str, Any] | None:
        if primary is None:
            return None

        height, width = frame.shape[:2]

        frame_cx = width / 2.0
        frame_cy = height / 2.0

        threshold = getattr(
            self.cfg.visualization,
            "center_threshold_px",
            40,
        )

        center_x = (
                           primary.bbox[0] + primary.bbox[2]
                   ) / 2.0

        center_y = (
                           primary.bbox[1] + primary.bbox[3]
                   ) / 2.0

        center_distance = math.hypot(
            center_x - frame_cx,
            center_y - frame_cy,
        )

        if (
                not self.serial.simulation
                and center_distance <= threshold
        ):
            track_color = (0, 200, 0)
        else:
            track_color = (0, 180, 80)

        draw_track(
            frame,
            primary,
            color=track_color,
        )

        self._draw_prediction(
            frame,
            primary,
        )

        class_name = self._class_name(
            primary.class_id
        )

        normalized_center = (
            center_x / width,
            center_y / height,
        )

        self._update_smooth_position(
            normalized_center
        )

        return {
            "class_name": class_name,
            "confidence": primary.confidence,
            "bbox": list(primary.bbox),
        }

    def _draw_prediction(
            self,
            frame,
            primary,
    ) -> None:
        try:
            height, width = frame.shape[:2]

            lead = getattr(
                self.cfg.tracking,
                "extrapolate_secs",
                0.2,
            )

            cx = (
                         primary.bbox[0] + primary.bbox[2]
                 ) / 2.0

            cy = (
                         primary.bbox[1] + primary.bbox[3]
                 ) / 2.0

            px = cx + primary.velocity[0] * lead
            py = cy + primary.velocity[1] * lead

            px = max(
                0,
                min(width - 1, px),
            )

            py = max(
                0,
                min(height - 1, py),
            )

            cv2.line(
                frame,
                (int(cx), int(cy)),
                (int(px), int(py)),
                (0, 200, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                (int(px), int(py)),
                5,
                (0, 200, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

        except Exception:
            self.display_logger.debug(
                "Could not draw target prediction",
                exc_info=True,
            )

    def _update_smooth_position(
            self,
            normalized_center: tuple[float, float],
    ) -> None:
        if self.smooth_pos is None:
            self.smooth_pos = list(
                normalized_center
            )
            return

        self.smooth_pos[0] = (
                self.smooth_pos[0] * 0.8
                + normalized_center[0] * 0.2
        )

        self.smooth_pos[1] = (
                self.smooth_pos[1] * 0.8
                + normalized_center[1] * 0.2
        )

    @staticmethod
    def _class_name(class_id: int) -> str:
        try:
            from src.coco import COCO_CLASSES

            if 0 <= class_id < len(COCO_CLASSES):
                name = COCO_CLASSES[class_id]

                return (
                    name
                    if isinstance(name, str)
                    else str(class_id)
                )

        except (
                IndexError,
                TypeError,
        ):
            pass

        return str(class_id)

    def _draw_center_ui(
            self,
            frame,
    ) -> None:
        center = None

        if (
                self.cfg.debug.enabled
                and self.serial.simulation
        ):
            center = self.serial_center

            if center is None:
                current_time = time.time()

                center = (
                    0.5
                    + 0.35
                    * math.sin(current_time * 0.8),
                    0.5
                    + 0.25
                    * math.cos(current_time * 0.6),
                )

        draw_center_ui(
            frame,
            center,
        )

    def _draw_panels(
            self,
            frame,
            tracked_info,
    ) -> None:
        draw_top_left_panel(
            frame,
            tracked_info,
            looking_for=self.track_only,
        )

        if getattr(
                self.cfg.visualization,
                "show_tracking_queue",
                True,
        ):
            draw_tracking_queue(
                frame,
                self.target_queue.target_ids(),
                self.target_queue.current_id,
            )

    def _device_name(self) -> str:
        device = getattr(
            self.detector,
            "device",
            None,
        )

        if isinstance(device, str) and device:
            return device

        try:
            first_detector = next(
                iter(
                    self.detector.detectors.values()
                )
            )

            return str(
                getattr(
                    first_detector.get("obj"),
                    "device",
                    None,
                )
                or "cpu"
            )

        except Exception:
            return "cpu"

    def _display_frame(
            self,
            frame,
            detections,
            tracks,
    ) -> None:
        now = time.time()

        elapsed = now - self.last_time

        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        self.last_time = now

        try:
            show_info(
                frame,
                fps,
                self.infer_time_ms,
                device=self._device_name(),
                detections=len(detections),
                tracks=len(tracks),
                font_scale_override=0.42,
            )

            cv2.imshow(
                self.WINDOW_NAME,
                frame,
            )

            self.latest_click_tracks = [
                (track.id, track.bbox)
                for track in tracks
                if track.lost == 0 and not self.target_queue.is_hit(track.id)
            ]

        except Exception:
            self.display_logger.exception(
                "Failed to render frame"
            )

    def _send_telemetry(
            self,
            frame,
            primary,
            timestamp: float,
    ) -> None:
        if primary is None:
            return

        try:
            height, width = frame.shape[:2]

            cx = (
                         primary.bbox[0] + primary.bbox[2]
                 ) / 2.0

            cy = (
                         primary.bbox[1] + primary.bbox[3]
                 ) / 2.0

            normalized_center = [
                cx / width,
                cy / height,
            ]

            lead = getattr(
                self.cfg.tracking,
                "extrapolate_secs",
                0.2,
            )

            predicted_x = (
                    cx + primary.velocity[0] * lead
            )

            predicted_y = (
                    cy + primary.velocity[1] * lead
            )

            predicted_center = [
                predicted_x / width,
                predicted_y / height,
            ]

            self.serial.send_telemetry(
                primary.id,
                self._class_name(
                    primary.class_id
                ),
                primary.confidence,
                normalized_center,
                [
                    primary.velocity[0],
                    primary.velocity[1],
                ],
                timestamp,
                predicted_center=predicted_center,
                require_ack=False,
                advanced=(
                    self.cfg.serial
                    .advanced_datapackets
                ),
            )

        except Exception:
            self.serial_logger.exception(
                "Failed to send telemetry"
            )

    def _process_frame(
            self,
            camera_index: int,
            frame_timestamp: float,
            frame,
    ) -> None:
        (
            detections,
            timestamp,
            fresh_result,
            result,
        ) = self._get_detections(
            camera_index,
            frame,
            frame_timestamp,
        )

        if fresh_result:
            self._process_fresh_detections(
                frame,
                detections,
            )

        hand_results = (
            self.optional_features.detect_hands(
                frame
            )
        )

        self._publish_detection_dump(
            camera_index,
            detections,
            timestamp,
            result,
        )

        tracks, primary = self._update_tracking(
            detections,
            timestamp,
        )

        self._draw_detections(
            frame,
            detections,
            primary,
        )

        self._draw_hands(
            frame,
            hand_results,
        )

        self._draw_secondary_tracks(
            frame,
            tracks,
            primary,
        )

        tracked_info = self._draw_primary_track(
            frame,
            primary,
        )

        self._draw_center_ui(frame)

        self._draw_panels(
            frame,
            tracked_info,
        )

        self._display_frame(
            frame,
            detections,
            tracks,
        )

        self._send_telemetry(
            frame,
            primary,
            timestamp,
        )

    @staticmethod
    def _quit_requested() -> bool:
        try:
            return (
                    cv2.waitKey(1) & 0xFF
            ) == ord("q")

        except Exception:
            return False

    def run(self) -> None:
        self._start()
        logger.info("Runtime event loop started")

        try:
            while True:
                self._ensure_inference_worker()

                frames = self.cam_mgr.read_all()

                if not frames:
                    time.sleep(0.005)
                    continue

                for (
                        camera_index,
                        timestamp,
                        frame,
                ) in frames:
                    self._process_frame(
                        camera_index,
                        timestamp,
                        frame,
                    )

                    if self._quit_requested():
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            logger.info("Shutting down")

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Stopping runtime components")
        try:
            self.inf_worker.stop()
        except Exception:
            logger.debug(
                "Failed to stop inference worker",
                exc_info=True,
            )

        try:
            self.cam_mgr.stop_all()
        except Exception:
            logger.debug(
                "Failed to stop cameras",
                exc_info=True,
            )

        try:
            self.serial.stop()
        except Exception:
            logger.debug(
                "Failed to stop serial interface",
                exc_info=True,
            )

        cv2.destroyAllWindows()
        logger.info("Runtime shutdown complete")

        print()
        print_banner("ShooterBot - By Shahm Najeeb")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config",
    )

    return parser.parse_args()


def load_application_config(config_argument: str):
    logger.info("Application config is loading.")

    try:
        config_path = resolve_config_path(config_argument)
    except ValueError as exc:
        logger.critical("Invalid configuration path: %s", exc)
        raise SystemExit(2) from exc
    logger.debug("Resolved configuration path to %s", config_path)

    if not config_path.is_file():
        logger.critical(
            f"Configuration was not found: {config_path}\n"
            "Run the installer first: python -m src.cli.installer"
        )
        raise SystemExit(2)

    cfg = load_config(config_path)

    setup_logging(
        cfg.logging.level,
        cfg.logging.file,
        color=cfg.logging.color,
        verbose=cfg.logging.verbose,
    )
    logger.info(
        "Logging reconfigured from configuration level=%s verbose=%s",
        cfg.logging.level,
        cfg.logging.verbose,
    )

    try:
        check_runtime_setup(cfg)

    except RuntimeError as exc:
        logger.critical(
            "Setup check failed: %s",
            exc,
        )
        raise SystemExit(2) from exc

    logger.info("Runtime setup check passed")

    return cfg


def main() -> int:
    print()
    print_banner("ShooterBot - NIRT")
    logger.info("Starting ShooterBot setup phase")

    args = parse_args()
    cfg = load_application_config(args.config)

    app = NIRTShooterBotApplication(cfg)
    app.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Startup interrupted by user; exiting cleanly.")
        raise SystemExit(130)
