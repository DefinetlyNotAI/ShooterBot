from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.scripts._runtime import configure_script_output

print = configure_script_output(__name__)


def safe_str(value: Any) -> str:
    """Convert a value to a display-safe string."""
    if value is None:
        return "None"

    return str(value)


def safe_repr(value: Any, max_length: int = 200) -> str:
    """Return a bounded representation of a value."""
    if value is None:
        return "None"

    try:
        result = repr(value)
    except Exception as exc:
        return f"<repr failed: {type(exc).__name__}>"

    return result[:max_length]


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    cfg = load_config(str(config_path))
    detector_manager = DetectorManager(cfg)

    detector_keys = [
        safe_str(key)
        for key in detector_manager.detectors
    ]

    print("detector keys:", detector_keys)

    for key, metadata in detector_manager.detectors.items():
        detector_type = metadata.get("type")
        priority = metadata.get("priority")
        detector_obj = metadata.get("obj")

        print(
            safe_str(key),
            "type=",
            safe_str(detector_type),
            "priority=",
            safe_str(priority),
        )

        print(
            "  obj repr:",
            safe_repr(detector_obj),
        )

    print("device:", safe_str(detector_manager.device))


if __name__ == "__main__":
    main()
