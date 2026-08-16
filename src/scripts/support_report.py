"""Create a diagnostic report that a user can attach to a GitHub issue."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts._paths import generated_directory
from src.scripts._runtime import configure_script_output

print = configure_script_output(__name__)

PACKAGES = ("torch", "torchvision", "ultralytics", "opencv-python", "numpy", "PyYAML", "pyserial")
SENSITIVE_KEYS = {"password", "secret", "token", "key", "credential"}


def package_versions() -> dict[str, str]:
    """Return installed versions without importing heavyweight packages."""
    result: dict[str, str] = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def pip_check() -> str:
    """Capture dependency consistency without mutating the environment."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unable to run pip check: {exc}"
    return (result.stdout or result.stderr or "no issues reported").strip()


def command_output(command: list[str], timeout: int = 15) -> str:
    """Capture a diagnostic command without raising or changing project state."""
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr or f"exit code {result.returncode}").strip()


def redact_config() -> str:
    """Include configuration context without exposing obvious secret values."""
    config_path = ROOT / "configs" / "default.yaml"
    if not config_path.is_file():
        return "configuration file is missing"
    lines = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key = line.partition(":")[0].strip().lower()
        lines.append(
            f"{line.partition(':')[0]}: [REDACTED]"
            if any(marker in key for marker in SENSITIVE_KEYS)
            else line
        )
    return "\n".join(lines)


def log_tail() -> str:
    """Return recent application diagnostics, which may help issue triage."""
    log_path = generated_directory("logs") / "nirt_shooterbot.log"
    if not log_path.is_file():
        return "application log is missing"
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:])


def directory_state() -> dict[str, object]:
    """Report useful root-level artefact state without reading model bodies."""
    usage = shutil.disk_usage(ROOT)
    return {
        "free_disk_bytes": usage.free,
        "total_disk_bytes": usage.total,
        "files": sorted(path.name for path in (ROOT / "files").glob("*") if path.is_file()),
        "logs": sorted(path.name for path in generated_directory("logs").glob("*") if path.is_file()),
    }


def main() -> None:
    """Write a timestamped, plain-text diagnostic report under root logs/."""
    print.warning(
        "The support report may include sensitive device and user information. "
        "Review it before uploading it to GitHub."
    )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "computer": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "virtual_environment": sys.prefix,
        "command_line": sys.argv,
        "python_path": sys.path,
        "selected_environment": {
            name: os.environ.get(name, "")
            for name in ("OS", "PROCESSOR_ARCHITECTURE", "CUDA_PATH", "VIRTUAL_ENV")
        },
        "packages": package_versions(),
        "pip_check": pip_check(),
        "pip_version": command_output([sys.executable, "-m", "pip", "--version"]),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "config_present": (ROOT / "configs" / "default.yaml").is_file(),
        "redacted_config": redact_config(),
        "models": sorted(path.name for path in (ROOT / "models").glob("*.pt")),
        "recent_application_log": log_tail(),
        "project_directories": directory_state(),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = generated_directory("logs", create=True) / f"nirt_support_{timestamp}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print.info("Support report written to", destination)


if __name__ == "__main__":
    main()
