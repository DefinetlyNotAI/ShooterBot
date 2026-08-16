"""Report local runtime readiness without loading detector models."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.paths import models_dir
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)

PACKAGES = ("ultralytics", "opencv-python", "PyYAML", "numpy", "pyserial")


def distribution_version(name: str) -> str:
    """Return an installed package version without failing the report."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    installed_models_dir = models_dir()
    print("NIRT ShooterRobot runtime report")
    print("Project root:", ROOT)
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Configuration:", "ready" if config_path.is_file() else "missing")
    print(
        "Models directory:",
        "ready" if installed_models_dir.is_dir() else "missing",
    )
    if installed_models_dir.is_dir():
        models = sorted(path.name for path in installed_models_dir.glob("*.pt"))
        print("Models:", ", ".join(models) if models else "none found")
    print("Packages:")
    for package in PACKAGES:
        print(f"  {package}: {distribution_version(package)}")


if __name__ == "__main__":
    main()
