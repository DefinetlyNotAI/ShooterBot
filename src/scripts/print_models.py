from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def safe_str(value: Any) -> str:
    if value is None:
        return "None"

    return str(value)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"

    cfg = load_config(str(config_path))
    detector_manager = DetectorManager(cfg)

    for key, metadata in detector_manager.detectors.items():
        detector_type = metadata.get("type")
        detector_obj = metadata.get("obj")

        key_str = safe_str(key)
        type_str = safe_str(detector_type)

        if (
                detector_type == "yolo"
                and detector_obj is not None
                and hasattr(detector_obj, "model_name")
        ):
            model_name = getattr(detector_obj, "model_name", None)

            print(
                key_str,
                "model_name=",
                safe_str(model_name),
            )
        else:
            print(
                key_str,
                "type=",
                type_str,
            )


if __name__ == "__main__":
    main()
