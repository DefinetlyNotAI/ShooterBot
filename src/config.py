"""Configuration loader and typed config objects."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, cast

import yaml

T = TypeVar("T")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class CameraConfig:
    sources: List[Any] = field(default_factory=lambda: [0])
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend_preference: Optional[List[str]] = (
        None  # e.g. ['DSHOW','MSMF','ANY']
    )


@dataclass
class InferenceConfig:
    detector: str = "yolov8"

    # Model list is required to support quality fallback chains:
    # Example: ["yolov8m.pt", "yolov8n.pt"]
    # First model = preferred, following models = fallback
    # "auto" is supported as a special value.
    model: List[str] = field(default_factory=lambda: ["auto"])

    device: str = "auto"
    confidence_threshold: float = 0.55
    classes: List[int] = field(default_factory=list)
    ignored_classes: List[int] = field(default_factory=list)
    disabled_detectors: List[str] = field(default_factory=list)
    min_area: float = 0.0
    nms_iou_threshold: float = 0.45
    class_priority: List[str] = field(
        default_factory=lambda: [
            "face",
            "person",
            "cell phone",
            "phone",
            "bottle",
            "cup",
        ]
    )
    auto_load_models_from_cwd: bool = True
    max_models_to_load: int = 5

    text_search: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "method": "fuzzy"}
    )

    prefer_high_quality: bool = True
    prefer_high_quality_model: Optional[bool] = None

    # Single optional face model
    face_model: Optional[str] = None
    face_confidence: float = 0.35

    max_detector_time: float = 0.5

    run_mode: str = "cascade"
    face_model_interval: float = 1.0
    heavy_model_interval: float = 0.2


@dataclass
class TrackingConfig:
    max_lost: int = 30
    iou_threshold: float = 0.3
    min_hits: int = 1
    max_age: int = 30
    track_only: List[Any] = field(
        default_factory=lambda: ["face", "sports ball", "cell phone"]
    )
    class_priority: List[str] = field(
        default_factory=lambda: [
            "face",
            "sports ball",
            "cell phone",
            "phone",
            "person",
            "bottle",
            "cup",
        ]
    )
    extrapolate_secs: float = 0.2
    cycle_remember: bool = True
    remember_faces: bool = False
    face_memory_threshold: float = 0.78
    face_memory_max_age: int = 300


@dataclass
class FeaturesConfig:
    emotion_tracking: bool = False
    emotion_backend: str = "fer"
    hand_tracking: bool = False
    hand_backend: str = "mediapipe"
    hand_gesture_map: Dict[str, str] = field(
        default_factory=lambda: {
            "0": "fist",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "open",
        }
    )


@dataclass
class VisualizationConfig:
    show_fps: bool = True
    show_inference_time: bool = True
    colors: Dict[str, List[int]] = field(default_factory=dict)
    font_scale: float = 0.6
    center_threshold_px: int = 40
    show_tracking_queue: bool = True


@dataclass
class SerialConfig:
    enabled: bool = False
    port: str = "COM3"
    baudrate: int = 115200
    crc: bool = True
    simulation: bool = True
    advanced_datapackets: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/nirt_shooterbot.log"
    color: bool = True
    verbose: bool = False


@dataclass
class DebugConfig:
    enabled: bool = True
    simulate_camera: bool = False
    simulation_video: str = ""
    inject_fake_face: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )
    serial: SerialConfig = field(default_factory=SerialConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


MODEL_DIR = ROOT / "models"


def normalize_model_path(
        models: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
    """
    Ensures all inference models are inside models/.

    Supports:
    - "yolov8n.pt"
    - ["yolov8m.pt", "yolov8n.pt"]
    - ["models/yolov8m.pt", "models/yolov8n.pt"]
    - "auto"
    """

    if models is None:
        return None

    if isinstance(models, str):
        if not models:
            return None
        models = [models]

    if not models:
        return []

    normalized = []

    for model in models:
        if model == "auto":
            normalized.append(model)
            continue

        path = Path(model)

        if path.is_absolute():
            normalized.append(str(path))
        elif path.parts and path.parts[0] == MODEL_DIR.name:
            normalized.append(str(ROOT / path))
        else:
            normalized.append(str(MODEL_DIR / path))

    return normalized


def normalize_single_model_path(model: Optional[str]) -> Optional[str]:
    """
    Normalizes single model paths like face_model.
    """

    if not model:
        return model

    path = Path(model)

    if path.is_absolute():
        return str(path)

    if path.parts and path.parts[0] == MODEL_DIR.name:
        return str(ROOT / path)

    return str(MODEL_DIR / path)


def load_config(path: str) -> AppConfig:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def merge(
            cls: type[T],
            data: Optional[Mapping[str, Any]],
    ) -> T:
        if data is None:
            return cls()

        valid_fields = {
            item.name
            for item in fields(cast(Any, cls))
        }

        filtered = {
            key: value
            for key, value in data.items()
            if key in valid_fields
        }

        return cls(**filtered)

    cfg = AppConfig(
        camera=merge(CameraConfig, raw.get("camera", {})),
        inference=merge(InferenceConfig, raw.get("inference", {})),
        tracking=merge(TrackingConfig, raw.get("tracking", {})),
        features=merge(FeaturesConfig, raw.get("features", {})),
        visualization=merge(VisualizationConfig, raw.get("visualization", {})),
        serial=merge(SerialConfig, raw.get("serial", {})),
        logging=merge(LoggingConfig, raw.get("logging", {})),
        debug=merge(DebugConfig, raw.get("debug", {})),
    )

    raw_inference = raw.get("inference", {}) or {}
    raw_tracking = raw.get("tracking", {}) or {}

    if (
            "class_priority" not in raw_inference
            and "class_priority" in raw_tracking
    ):
        cfg.inference.class_priority = list(cfg.tracking.class_priority)
    elif (
            "class_priority" not in raw_tracking
            and "class_priority" in raw_inference
    ):
        cfg.tracking.class_priority = list(cfg.inference.class_priority)

    if getattr(cfg.inference, "prefer_high_quality_model", None) is not None:
        cfg.inference.prefer_high_quality = bool(
            cfg.inference.prefer_high_quality_model
        )

    interference_model = normalize_model_path(cfg.inference.model)
    if interference_model is None:
        raise RuntimeError("Interference model is not initialized")
    cfg.inference.model = interference_model
    cfg.inference.face_model = normalize_single_model_path(
        cfg.inference.face_model
    )

    return cfg
