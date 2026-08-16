"""Entry point to run the realtime CV framework (debug mode included)."""

from __future__ import annotations

# ensure package imports resolve when running script directly
import os
import sys
from typing import Dict, List, Any

from src.noise_control import configure_library_noise

configure_library_noise()

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
import time
import json
import cv2
from src.config import load_config
from src.utils import setup_logging, logger
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(
            "Configuration was not found: " + config_path + "\n"
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
    try:
        check_runtime_setup(cfg)
    except RuntimeError as exc:
        logger.critical("Setup check failed: %s", exc)
        raise SystemExit(2) from exc

    cam_mgr = CameraManager()
    import logging

    for src in cfg.camera.sources:
        # prefer real camera unless user explicitly requested simulation
        requested_sim = cfg.debug.simulate_camera
        sim_video = cfg.debug.simulation_video
        sim = requested_sim
        # if simulation not requested, test whether camera opens; if not, fall back to simulation
        if not requested_sim and isinstance(src, int):
            try:
                test_cap = cv2.VideoCapture(int(src))
                ok = test_cap.isOpened()
                test_cap.release()
                if not ok:
                    logging.getLogger("realtime_cv").warning(
                        "Camera %s not available, falling back to simulation.",
                        src,
                    )
                    sim = True
                else:
                    sim = False
            except Exception:
                sim = True
        cam = CameraStream(
            source=src,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            simulate=sim,
            sim_video=sim_video,
            inject_fake_face=getattr(cfg.debug, "inject_fake_face", False),
            backend_preference=getattr(cfg.camera, "backend_preference", None),
        )
        cam_mgr.add(cam)

    detector = DetectorManager(cfg)
    # log loaded detectors for diagnostics
    try:
        for k, v in detector.detectors.items():
            obj = v.get("obj")
            mname = (
                getattr(obj, "model_name", None) if obj is not None else None
            )
            logger.info(
                "Loaded detector %s: type=%s model=%s priority=%s",
                k,
                v.get("type"),
                mname,
                v.get("priority"),
            )
    except Exception:
        logger.debug("Could not list detectors", exc_info=True)
    # create inference worker to reduce UI lag
    inf_worker = InferenceWorker(detector)
    inf_worker.start()

    track_only = getattr(cfg.tracking, "track_only", None)
    tracker = Tracker(
        max_lost=cfg.tracking.max_lost,
        iou_threshold=cfg.tracking.iou_threshold,
        track_only=track_only,
        min_hits=getattr(cfg.tracking, "min_hits", 1),
        max_age=getattr(cfg.tracking, "max_age", cfg.tracking.max_lost),
        class_priority=getattr(cfg.tracking, "class_priority", None),
        remember_faces=getattr(cfg.tracking, "remember_faces", False),
        face_memory_threshold=getattr(
            cfg.tracking, "face_memory_threshold", 0.78
        ),
        face_memory_max_age=getattr(cfg.tracking, "face_memory_max_age", 300),
    )
    optional_features = OptionalFeatures(cfg)
    target_queue = TargetQueue(
        cycle_remember=getattr(cfg.tracking, "cycle_remember", True)
    )
    # Decide serial simulation mode:
    # - If serial is not enabled in config, always simulate (safe default)
    # - If serial.enabled is True, use the configured simulation setting
    if not getattr(cfg.serial, "enabled", False):
        sim_mode = True
    else:
        sim_mode = cfg.serial.simulation
    serial = SerialInterface(
        port=cfg.serial.port,
        baudrate=cfg.serial.baudrate,
        crc=cfg.serial.crc,
        simulation=sim_mode,
    )

    # ensure inference worker remains alive; restart if thread died
    def _ensure_inf_worker():
        try:
            if not inf_worker._thread.is_alive():
                inference_logger = __import__("logging").getLogger(
                    "realtime_cv.inference"
                )
                inference_logger.warning(
                    "Inference worker thread not alive; restarting"
                )
                inf_worker.stop()
                inf_worker.start()
        except Exception:
            pass

    # Shared event state: serial callbacks run on the serial reader thread.
    serial_center = {"nc": None}
    shot_state = {"requested": False, "target_id": None}
    import threading

    shot_lock = threading.Lock()

    def _request_shot(target_id=None):
        with shot_lock:
            shot_state["requested"] = True
            shot_state["target_id"] = target_id

    def _consume_shot():
        with shot_lock:
            if not shot_state["requested"]:
                return None, False
            target_id = shot_state["target_id"]
            shot_state["requested"] = False
            shot_state["target_id"] = None
            return target_id, True

    def _on_receive(data: bytes):
        try:
            import json

            text = data.decode("utf-8", errors="ignore").strip()
            # Accept a compact line signal as well as JSON from the Arduino.
            if text.upper() in {"SHOT", "HIT", "TRIGGER"}:
                _request_shot()
                return
            payload = json.loads(text)
            if (
                    payload.get("shot") is True
                    or payload.get("hit") is True
                    or str(payload.get("event", "")).lower() in {"shot", "hit"}
            ):
                requested_id = payload.get("id")
                _request_shot(
                    int(requested_id) if requested_id is not None else None
                )
                return
            # support both legacy 'nc' and simplified 'x','y' telemetry
            if (
                    "nc" in payload
                    and isinstance(payload.get("nc"), list)
                    and len(payload.get("nc")) == 2
            ):
                nc = payload.get("nc")
                serial_center["nc"] = (float(nc[0]), float(nc[1]))
            elif "x" in payload and "y" in payload:
                try:
                    serial_center["nc"] = (
                        float(payload.get("x")),
                        float(payload.get("y")),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    serial.set_receive_callback(_on_receive)
    serial.start()

    cv2.namedWindow("realtime-cv", cv2.WINDOW_NORMAL)
    latest_click_tracks = []

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # In debug mode a click inside any displayed target is the shot signal.
        if not cfg.debug.enabled:
            return
        for track_id, bbox in reversed(latest_click_tracks):
            x1, y1, x2, y2 = map(int, bbox)
            if x1 <= x <= x2 and y1 <= y <= y2:
                _request_shot(track_id)
                return

    cv2.setMouseCallback("realtime-cv", _on_mouse)
    last_time = time.time()
    infer_time_ms = 0.0
    fps = 0.0
    last_results: dict[int, List[Dict[str, Any]]] = {}
    smooth_pos = None
    primary_id = None
    try:
        while True:
            # ensure inference worker running
            try:
                _ensure_inf_worker()
            except Exception:
                pass
            frames = cam_mgr.read_all()
            if not frames:
                time.sleep(0.005)
                continue
            for idx, ts, frame in frames:
                fresh_result = False
                try:
                    loop_logger = __import__("logging").getLogger(
                        "realtime_cv.loop"
                    )
                    loop_logger.debug("Main loop frame idx=%s ts=%s", idx, ts)
                    # submit frame for background inference (non-blocking)
                    try:
                        inf_worker.submit(idx, frame, ts)
                    except Exception:
                        loop_logger = __import__("logging").getLogger(
                            "realtime_cv.loop"
                        )
                        loop_logger.exception(
                            "Failed to submit frame to inference worker"
                        )
                    # try to get latest result for this camera
                    res = inf_worker.get(idx)
                    if res:
                        dets = res["detections"]
                        infer_time_ms = res["inference_time_ms"]
                        last_results[idx] = dets
                        timestamp = res["timestamp"]
                        fresh_result = True
                    else:
                        dets = last_results.get(idx, [])
                        timestamp = time.time()
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    dets = []
                    timestamp = time.time()

                if fresh_result:
                    for detection in dets:
                        if (
                                str(
                                    detection.get(
                                        "class_name", detection.get("label", "")
                                    )
                                ).lower()
                                == "face"
                        ):
                            detection["appearance"] = face_appearance(
                                frame, detection.get("bbox")
                            )
                    optional_features.annotate_emotions(frame, dets)
                hand_results = optional_features.detect_hands(frame)

                # Detailed detection dumps are opt-in; normal operation stays quiet.
                if cfg.serial.advanced_datapackets and res and dets:
                    det_events = []
                    for d in dets:
                        det_events.append(
                            {
                                "bbox": d.get("bbox"),
                                "bbox_normalized": d.get("bbox_normalized"),
                                "center": d.get("center"),
                                "normalized_center": d.get(
                                    "normalized_center"
                                ),
                                "class_id": d.get("class_id"),
                                "class_name": d.get("class_name", None),
                                "confidence": d.get("confidence"),
                                "timestamp": timestamp,
                            }
                        )
                    if det_events:
                        print(
                            json.dumps(
                                {
                                    "camera_index": idx,
                                    "timestamp": timestamp,
                                    "detections": det_events,
                                }
                            )
                        )

                # update tracker only with detections intended for tracking
                tracks = tracker.update(dets, timestamp)
                target_queue.sync(tracks)
                # A hardware shot has the same effect as a debug click.
                shot_id, shot_requested = _consume_shot()
                if shot_requested:
                    target_queue.mark_shot(
                        shot_id
                        if shot_id is not None
                        else target_queue.current_id
                    )
                target_id = target_queue.select_next()
                tracks_by_id = {track.id: track for track in tracks}
                primary = (
                    tracks_by_id.get(target_id)
                    if target_id is not None
                    else None
                )
                if primary is not None and primary.lost != 0:
                    primary = None
                primary_id = primary.id if primary else None

                # visualization: draw ALL detections, but only draw tracked overlay for primary
                # helper to match a detection to primary by center distance
                def _is_match(det, tr, thresh_px=40):
                    try:
                        cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                        cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                        tcx = (tr.bbox[0] + tr.bbox[2]) / 2.0
                        tcy = (tr.bbox[1] + tr.bbox[3]) / 2.0
                        return (
                                (cx - tcx) ** 2 + (cy - tcy) ** 2
                        ) ** 0.5 <= thresh_px
                    except Exception:
                        return False

                # determine modes
                real_mode = not serial.simulation
                h, w = frame.shape[:2]
                frame_cx, frame_cy = w / 2.0, h / 2.0
                center_thresh_px = getattr(
                    cfg.visualization, "center_threshold_px", 40
                )

                for d in dets:
                    bbox = d["bbox"]
                    label = str(d.get("class_name", d.get("class_id", "")))
                    if d.get("emotion"):
                        label = f"{label} | {d['emotion']}"
                    # color selection depends on mode
                    if real_mode:
                        # compute det center
                        try:
                            dcx = (bbox[0] + bbox[2]) / 2.0
                            dcy = (bbox[1] + bbox[3]) / 2.0
                            dist = (
                                           (dcx - frame_cx) ** 2 + (dcy - frame_cy) ** 2
                                   ) ** 0.5
                            if dist <= center_thresh_px:
                                color = (0, 200, 0)  # green when centered
                            else:
                                color = (
                                    0,
                                    200,
                                    200,
                                )  # yellow/turquoise when not centered
                        except Exception:
                            color = (0, 200, 200)
                    else:
                        # debug/simulated mode: muted boxes, highlight primary in blue
                        color = (200, 200, 200)
                        if primary and _is_match(d, primary, thresh_px=50):
                            color = (20, 150, 255)
                    draw_detection(
                        frame,
                        bbox,
                        label=label,
                        confidence=d.get("confidence", 0.0),
                        color=color,
                    )

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
                    if points:
                        hx = min(p[0] for p in points)
                        hy = max(15, min(p[1] for p in points) - 8)
                        cv2.putText(
                            frame,
                            f"{hand.get('handedness', 'hand')}: {hand.get('gesture', 'unknown')}",
                            (hx, hy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 220, 80),
                            1,
                            cv2.LINE_AA,
                        )

                # Keep all non-locked targets visible so the operator can click them
                # in debug mode and so multi-target state is easy to inspect.
                for track in tracks:
                    if not primary or track.id != primary.id:
                        draw_track(frame, track, color=(120, 120, 120))

                # draw primary track with stronger UI and update smooth_pos
                tracked_info = None
                if primary:
                    # draw tracked overlay near original bbox
                    # color green if centered (real mode) else cyan
                    try:
                        pcx = (primary.bbox[0] + primary.bbox[2]) / 2.0
                        pcy = (primary.bbox[1] + primary.bbox[3]) / 2.0
                        distc = (
                                        (pcx - frame_cx) ** 2 + (pcy - frame_cy) ** 2
                                ) ** 0.5
                        track_color = (
                            (0, 200, 0)
                            if (
                                    not serial.simulation
                                    and distc <= center_thresh_px
                            )
                            else (0, 180, 80)
                        )
                    except Exception:
                        track_color = (0, 180, 80)
                    draw_track(frame, primary, color=track_color)
                    # draw extrapolated future point and line
                    try:
                        lead = getattr(cfg.tracking, "extrapolate_secs", 0.2)
                        cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
                        cy = (primary.bbox[1] + primary.bbox[3]) / 2.0
                        px = cx + primary.velocity[0] * lead
                        py = cy + primary.velocity[1] * lead
                        # clamp
                        px = max(0, min(w - 1, px))
                        py = max(0, min(h - 1, py))
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
                        pass
                    # map class id to name if possible
                    try:
                        from src.coco import COCO_CLASSES

                        class_name = (
                            COCO_CLASSES[primary.class_id]
                            if 0 <= primary.class_id < len(COCO_CLASSES)
                            else str(primary.class_id)
                        )
                    except Exception:
                        class_name = str(primary.class_id)
                    tracked_info = {
                        "class_name": class_name,
                        "confidence": primary.confidence,
                        "bbox": list(primary.bbox),
                    }
                    # center smoothing
                    cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
                    cy = (primary.bbox[1] + primary.bbox[3]) / 2.0
                    w, h = frame.shape[1], frame.shape[0]
                    nc = (cx / w, cy / h)
                    if smooth_pos is None:
                        smooth_pos = list(nc)
                    else:
                        smooth_pos[0] = smooth_pos[0] * 0.8 + nc[0] * 0.2
                        smooth_pos[1] = smooth_pos[1] * 0.8 + nc[1] * 0.2
                else:
                    tracked_info = None

                # draw center UI: debug mode should move crosshair via serial simulation; real mode stays fixed
                sc = None
                if cfg.debug.enabled:
                    if serial.simulation:
                        sc = serial_center.get("nc")
                        # if no serial updates available, synthesize a smooth moving dot
                        if sc is None:
                            t = time.time()
                            sx = 0.5 + 0.35 * (__import__("math").sin(t * 0.8))
                            sy = 0.5 + 0.25 * (__import__("math").cos(t * 0.6))
                            sc = (sx, sy)
                    else:
                        sc = None
                draw_center_ui(frame, sc)

                # top-right panel for primary tracked (prominent placement); show what we're looking for
                draw_top_left_panel(
                    frame, tracked_info, looking_for=track_only
                )
                if getattr(cfg.visualization, "show_tracking_queue", True):
                    draw_tracking_queue(
                        frame,
                        target_queue.target_ids(),
                        target_queue.current_id,
                    )

                now = time.time()
                fps = 1.0 / (now - last_time) if now != last_time else 0.0
                last_time = now
                device_name = getattr(detector, "device", None)
                if not device_name:
                    try:
                        first_detector = next(
                            iter(detector.detectors.values())
                        )
                        device_name = (
                                getattr(first_detector.get("obj"), "device", None)
                                or "cpu"
                        )
                    except Exception:
                        device_name = "cpu"
                try:
                    show_info(
                        frame,
                        fps,
                        infer_time_ms,
                        device=device_name,
                        detections=len(dets),
                        tracks=len(tracks),
                        font_scale_override=0.42,
                    )
                    cv2.imshow("realtime-cv", frame)
                    latest_click_tracks = [
                        (track.id, track.bbox)
                        for track in tracks
                        if track.lost == 0
                        if not target_queue.is_hit(track.id)
                    ]
                except Exception:
                    display_logger = __import__("logging").getLogger(
                        "realtime_cv.display"
                    )
                    display_logger.exception("Failed to render frame")
                # publish simplified telemetry for primary only
                if primary:
                    try:
                        # noinspection DuplicatedCode
                        h, w = frame.shape[:2]
                        cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
                        cy = (primary.bbox[1] + primary.bbox[3]) / 2.0
                        nc = [cx / w, cy / h]
                        # extrapolate predicted position (lead seconds)
                        lead = getattr(cfg.tracking, "extrapolate_secs", 0.2)
                        px = cx + primary.velocity[0] * lead
                        py = cy + primary.velocity[1] * lead
                        pnc = [px / w, py / h]
                        # send class_name (friendly) instead of id-only string; include predicted center
                        serial.send_telemetry(
                            primary.id,
                            class_name,
                            primary.confidence,
                            nc,
                            [primary.velocity[0], primary.velocity[1]],
                            timestamp,
                            predicted_center=pnc,
                            require_ack=False,
                            advanced=cfg.serial.advanced_datapackets,
                        )
                    except Exception:
                        serial_logger = __import__("logging").getLogger(
                            "realtime_cv.serial"
                        )
                        serial_logger.exception("Failed to send telemetry")
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt
                except Exception:
                    # sometimes waitKey can fail in some OpenCV builds; keep looping
                    pass
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        cam_mgr.stop_all()
        serial.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
