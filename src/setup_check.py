"""Startup checks for Python dependencies, enabled features, and model data."""

from __future__ import annotations

import importlib
import logging
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger("realtime_cv.setup")

REQUIRED_PACKAGES = {
    "torch": "torch>=2.5.0",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "psutil": "psutil",
    "serial": "pyserial",
}
REQUIRED_MINIMUMS = {
    "torch": ("torch", (2, 5)),
    "ultralytics": ("ultralytics", (8, 1, 28)),
    "cv2": ("opencv-python", (4, 7)),
    "yaml": ("PyYAML", (6, 0)),
    "numpy": ("numpy", (1, 26)),
    "psutil": ("psutil", (5, 9)),
    "serial": ("pyserial", (3, 5)),
}


def _missing_packages(packages: dict[str, str]) -> list[str]:
    missing = []
    for module, install_name in packages.items():
        distribution, minimum = REQUIRED_MINIMUMS.get(
            module, (install_name, (0,))
        )
        try:
            version = metadata.version(distribution).split("+")[0]
            actual = tuple(
                int(part) for part in version.split(".") if part.isdigit()
            )[:3]
            if actual < minimum:
                missing.append(
                    f"{install_name} (installed version {version} is too old)"
                )
        except (metadata.PackageNotFoundError, ValueError, TypeError):
            missing.append(install_name)
    return missing


def _feature_import_error(feature: str) -> str | None:
    try:
        if feature == "emotion":
            module = importlib.import_module("fer.fer")
            if not hasattr(module, "FER"):
                return "fer.fer.FER is unavailable"
        elif feature == "hands":
            module = importlib.import_module("mediapipe")
            if not hasattr(module, "solutions"):
                return "mediapipe.solutions is unavailable; install mediapipe==0.10.21"
    except Exception as exc:
        return str(exc)
    return None


def _configured_model_paths(config: Any) -> list[Path]:
    paths: list[Path] = []

    models = getattr(config.inference, "model", []) or []

    if isinstance(models, str):
        models = [models]

    for model in models:
        if isinstance(model, str) and model.lower() != "auto":
            paths.append(Path(model))

    face_model = getattr(config.inference, "face_model", None)

    if isinstance(face_model, str) and face_model:
        paths.append(Path(face_model))

    return paths


def check_runtime_setup(config: Any) -> None:
    """Fail fast for mandatory packages and warn about recoverable setup issues."""
    logger.info("Runtime setup check starting.")

    missing = _missing_packages(REQUIRED_PACKAGES)
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Missing required Python packages: "
            + packages
            + ". Run: python -m src.cli.installer from an activated virtual environment."
        )
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    if not model_dir.exists():
        logger.warning("Models directory is missing: %s", model_dir)
    for model_path in _configured_model_paths(config):
        resolved = (
            model_path if model_path.is_absolute() else root / model_path
        )
        if not resolved.exists():
            logger.warning(
                "Configured model is not present: %s. Rerun python -m src.cli.installer or place the file in models/.",
                resolved,
            )

    features = getattr(config, "features", None)
    optional = {}
    if features and getattr(features, "emotion_tracking", False):
        optional["fer"] = "fer"
    if features and getattr(features, "hand_tracking", False):
        optional["mediapipe"] = "mediapipe"
    missing_optional = _missing_packages(optional)
    for package in missing_optional:
        logger.warning(
            "Configured optional feature is unavailable because '%s' is not installed. "
            "Run: python -m src.cli.installer and choose the optional feature, or install its package manually.",
            package,
        )
    if (
            features
            and getattr(features, "emotion_tracking", False)
            and "fer" not in missing_optional
    ):
        error = _feature_import_error("emotion")
        if error:
            logger.warning(
                "Emotion tracking backend is unavailable: %s", error
            )
    if (
            features
            and getattr(features, "hand_tracking", False)
            and "mediapipe" not in missing_optional
    ):
        error = _feature_import_error("hands")
        if error:
            logger.warning("Hand tracking backend is unavailable: %s", error)

    logger.info("Runtime setup check passed: required packages available")
