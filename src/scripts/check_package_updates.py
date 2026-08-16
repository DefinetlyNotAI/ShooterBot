"""Report available pip updates without changing the environment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)

PROJECT_PACKAGES = {
    "ultralytics",
    "opencv-python",
    "pyyaml",
    "numpy",
    "psutil",
    "pyserial",
    "torch",
    "torchvision",
    "torchaudio",
    "fer",
    "mediapipe",
    "scikit-learn",
    "sentence-transformers",
    "scipy",
}


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            "pip update check failed: " + (result.stderr.strip() or "unknown error")
        )
    try:
        available = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("pip returned invalid update data") from exc
    updates = [
        item for item in available
        if str(item.get("name", "")).lower() in PROJECT_PACKAGES
    ]
    if not updates:
        print("No project package upgrades are available")
        return
    print("Available project package upgrades:")
    for item in sorted(updates, key=lambda value: str(value["name"]).lower()):
        print(
            f"  {item['name']}: {item['version']} -> {item['latest_version']}"
        )
    print("No packages were changed")


if __name__ == "__main__":
    main()
