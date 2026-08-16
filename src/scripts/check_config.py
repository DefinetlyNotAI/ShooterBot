"""Validate the active configuration without loading models or cameras."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Configuration is missing. Run python -m src.cli.installer first."
        )
    config = load_config(str(config_path))
    print("Configuration loaded successfully:", config_path)
    print("Camera sources:", config.camera.sources)
    print("Inference device:", config.inference.device)
    print("Configured models:", config.inference.model)
    print("Face model:", config.inference.face_model or "not configured")
    print("Serial mode:", "enabled" if config.serial.enabled else "simulation")
    print("Configuration validation passed")


if __name__ == "__main__":
    main()
