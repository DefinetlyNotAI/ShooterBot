# Settings Reference

This project loads YAML into typed dataclasses in `src/config.py`. Unknown keys in the YAML file are ignored, so a setting only works if the corresponding dataclass field exists or the code reads it with `getattr(...)`.

## Camera

- `camera.sources` - camera indexes or video paths.
- `camera.width` - capture width.
- `camera.height` - capture height.
- `camera.fps` - requested frame rate.
- `camera.backend_preference` - preferred OpenCV backends such as `DSHOW`, `MSMF`, or `ANY`.

## Inference

- `inference.detector` - detector family name, currently `yolov8`.
- `inference.model` - one model or a fallback list of models.
- `inference.device` - `auto`, `cpu`, or `cuda`.
- `inference.confidence_threshold` - minimum confidence for normal detections (default: `0.55`).
- `inference.classes` - allowed class IDs or names.
- `inference.ignored_classes` - class IDs or names to skip.
- `inference.min_area` - minimum detection area as a fraction of the frame.
- `inference.disabled_detectors` - detector names to disable, such as `face`.
- `inference.face_model` - face-specific `.pt` model used alongside the general YOLO model.
- `inference.face_confidence` - threshold for the face detector (default: `0.35`).
- `inference.max_detector_time` - intended per-detector time cap.
- `inference.face_model_interval` - throttle interval for the face detector.
- `inference.heavy_model_interval` - throttle interval for heavier models.
- `inference.prefer_high_quality` - prefer higher quality YOLO variants when `model: auto`.
- `inference.auto_load_models_from_cwd` - auto-discover additional local `.pt` models in the working tree.
- `inference.max_models_to_load` - cap the number of auto-discovered models.
- `inference.run_mode` - `cascade`, `all`, or `face_only`.
- `inference.nms_iou_threshold` - IoU threshold for duplicate detection pruning.
- `inference.text_search.enabled` - controls class-name search helpers.
- `inference.text_search.method` - `fuzzy` or `semantic`.

## Tracking

- `tracking.max_lost` - how many frames a track can disappear before removal.
- `tracking.iou_threshold` - minimum IoU for association.
- `tracking.min_hits` - number of matches required before a track is confirmed.
- `tracking.max_age` - maximum age in update cycles before a track is removed.
- `tracking.track_only` - classes or IDs to keep as active tracks. The shipped defaults track `face`, `sports ball`, and `cell phone` together.
- `tracking.class_priority` - preferred classes when detections overlap.
- `tracking.extrapolate_secs` - look-ahead window for the predicted point.
- `tracking.cycle_remember` - re-add a shot target at the end of the live queue when true; permanently exclude it when false.
- `tracking.remember_faces` - preserve face IDs in memory when a face leaves the view; memory is discarded when the process ends.
- `tracking.face_memory_threshold` - appearance similarity threshold used when reviving a remembered face.
- `tracking.face_memory_max_age` - maximum age in seconds for an in-memory face identity.

## Optional Features

- `features.emotion_tracking` - annotate detected faces with an emotion label. Requires the optional `fer` package.
- `features.emotion_backend` - currently `fer`.
- `features.hand_tracking` - detect hands, count extended fingers, and show a basic gesture label. Requires the optional `mediapipe` package.
- `features.hand_backend` - currently `mediapipe`; the installer pins the compatible legacy Solutions API release.
- `features.hand_gesture_map` - maps the detected number of extended fingers (`0`-`5`) to an operator label.

## Visualization

- `visualization.show_fps` - toggles FPS display.
- `visualization.show_inference_time` - toggles inference timing display.
- `visualization.colors` - optional class color overrides.
- `visualization.font_scale` - text scaling for overlays.
- `visualization.center_threshold_px` - pixel radius used to mark a target as centered.
- `visualization.show_tracking_queue` - show the tracking queue submenu in the camera view.

## Serial

- `serial.enabled` - enables hardware serial use.
- `serial.port` - port name such as `COM3`.
- `serial.baudrate` - serial baud rate.
- `serial.crc` - appends CRC32 bytes to each outgoing packet.
- `serial.simulation` - keeps the serial layer in simulation mode.
- `serial.advanced_datapackets` - enables verbose console detection packets and optional serial metadata. Leave false for minimal `x/y` packets.

## Logging

- `logging.level` controls the console threshold and, by default, the log-file
  threshold.
- `logging.verbose: true` keeps the console at `logging.level`, but records
  DEBUG entries and source locations in the log file.

- `logging.level` - log level such as `INFO` or `DEBUG`.
- `logging.file` - runtime log file path; defaults to `logs/nirt_shooterbot.log` and is always kept under the project log folder unless an explicit path is supplied.
- `logging.color` - enables colored console levels when output goes to a terminal; file logs remain plain text.
- `logging.verbose` - enables DEBUG-level output and adds function/line context to console messages.

## Debug

- `debug.enabled` - general debug toggle.
- `debug.simulate_camera` - forces camera simulation.
- `debug.simulation_video` - optional video file used as simulated input.
- `debug.inject_fake_face` - injects a fake face when no detections are found.

## Configuration behavior

- Unknown YAML keys are ignored; a setting must exist in `src/config.py` or be explicitly read by runtime code.
- Relative model paths are resolved under `models/`; absolute paths are supported.
- When `inference.class_priority` is not provided, the tracking priority list is used for inference overlap pruning.
- Face memory is process-local and is discarded when the application exits.
- Optional emotion and hand features are disabled with a warning when their backends are unavailable. FER is loaded from `fer.fer.FER`; MediaPipe hand classification follows its mirrored-image handedness convention.
