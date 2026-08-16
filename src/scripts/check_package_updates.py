"""Report available pip updates without changing the environment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts._runtime import configure_script_output

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
CUDA_PACKAGE_NAMES = {"torch", "torchvision", "torchaudio"}


def is_accelerated_build(package_name: str, installed_version: str) -> bool:
    """Return whether a PyTorch package is tied to a non-default wheel index."""
    return package_name.lower() in CUDA_PACKAGE_NAMES and "+" in installed_version


def split_updates(available: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Avoid treating PyPI's CPU wheel as an update for an accelerated build."""
    updates: list[dict[str, object]] = []
    accelerated: list[dict[str, object]] = []
    for item in available:
        package_name = str(item.get("name", ""))
        if package_name.lower() not in PROJECT_PACKAGES:
            continue
        if is_accelerated_build(package_name, str(item.get("version", ""))):
            accelerated.append(item)
        else:
            updates.append(item)
    return updates, accelerated


def main() -> None:
    print("Starting outdated check with pip.. May take a while")
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
    updates, accelerated = split_updates(available)
    if not updates and not accelerated:
        print("No project package upgrades are available")
        return
    if updates:
        print("Available project package upgrades:")
        for item in sorted(updates, key=lambda value: str(value["name"]).lower()):
            print(
                f"  {item['name']}: {item['version']} -> {item['latest_version']}"
            )
    for item in sorted(accelerated, key=lambda value: str(value["name"]).lower()):
        print.info(
            f"  {item['name']}: {item['version']} is an accelerated library; "
            "the script cannot check its version.\n  Please manually check "
            "the matching PyTorch CUDA wheel index and compatibility matrix."
        )
    print("No packages were changed")


if __name__ == "__main__":
    main()
