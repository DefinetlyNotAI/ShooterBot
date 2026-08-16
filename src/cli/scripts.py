"""Interactive launcher for the repository's diagnostic and utility scripts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.noise_control import configure_library_noise

configure_library_noise()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.installer import Stop, UI, integer_value
from src.utils import logger, setup_logging

setup_logging()

CONFIG = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ScriptSpec:
    """A runnable script and the local capabilities it requires."""

    key: str
    label: str
    module: str
    description: str
    packages: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    config_required: bool = True
    cuda_required: bool = False


SCRIPTS = (
    ScriptSpec(
        "runtime_report",
        "Runtime Report",
        "src.scripts.runtime_report",
        "Report configuration, model, Python, and package readiness.",
        config_required=False,
    ),
    ScriptSpec(
        "detectors",
        "Inspect Detectors",
        "src.scripts.check_detectors",
        "Load configured detectors and print their metadata.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "models",
        "Print Loaded Models",
        "src.scripts.print_models",
        "List configured detector models after loading them.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "simulation",
        "Run Simulation Inference",
        "src.scripts.sim_infer",
        "Run a synthetic frame through the configured detector stack.",
        packages=("numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "simulation_image",
        "Save Simulation Output",
        "src.scripts.run_sim_save",
        "Create an annotated synthetic image under files/.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "annotated_frame",
        "Save Annotated Frame",
        "src.scripts.save_annotated_frame",
        "Create an annotated simulation image using its debug configuration.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "frame",
        "Test One Frame",
        "src.scripts.test_frame",
        "Use one webcam frame when available, otherwise a synthetic frame.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "inference",
        "Camera Inference Smoke Test",
        "src.scripts.test_inference",
        "Run each installed model against one live webcam frame.",
        packages=("cv2", "ultralytics"),
        models=("yolov8n.pt",),
        config_required=False,
    ),
    ScriptSpec(
        "benchmark",
        "Benchmark Configured Inference",
        "src.scripts.benchmark",
        "Benchmark the configured detector stack on a live webcam.",
        packages=("cv2", "numpy", "yaml", "ultralytics"),
    ),
    ScriptSpec(
        "cuda_benchmark",
        "CUDA YOLO Benchmark",
        "src.scripts.test_yolo",
        "Benchmark YOLO on a live webcam using CUDA.",
        packages=("cv2", "torch", "ultralytics"),
        models=("yolov8n.pt",),
        config_required=False,
        cuda_required=True,
    ),
)


def available_packages(packages: tuple[str, ...]) -> list[str]:
    """Return package import names that are currently unavailable."""
    return [name for name in packages if importlib.util.find_spec(name) is None]


def cuda_available() -> bool:
    """Avoid importing PyTorch unless a CUDA-only script is selected."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def script_issues(spec: ScriptSpec) -> list[str]:
    """Return actionable reasons that prevent a script from running."""
    issues: list[str] = []
    missing_packages = available_packages(spec.packages)
    if missing_packages:
        issues.append("missing packages: " + ", ".join(missing_packages))
    if spec.config_required and not CONFIG.is_file():
        issues.append("missing configs/default.yaml")
    missing_files = [
        path for path in spec.required_files if not (ROOT / path).is_file()
    ]
    if missing_files:
        issues.append("missing files: " + ", ".join(missing_files))
    missing_models = [
        name for name in spec.models if not (ROOT / "models" / name).is_file()
    ]
    if missing_models:
        issues.append("missing models: " + ", ".join(missing_models))
    if spec.cuda_required and not cuda_available():
        issues.append("CUDA is unavailable")
    return issues


def health_check(ui: UI) -> bool:
    """Display script availability without running a model or camera."""
    ui.header("Scripts Health Check")
    healthy = True
    for spec in SCRIPTS:
        issues = script_issues(spec)
        if issues:
            healthy = False
            ui.out(f"  {spec.label}: unavailable ({'; '.join(issues)})", "yellow")
        else:
            ui.out(f"  {spec.label}: ready", "green")
    return healthy


def choose_script(ui: UI) -> ScriptSpec | None:
    """Show a numbered, capability-aware menu of scripts."""
    ui.header("Project Scripts")
    ui.out("    1. Health Check", "white")
    enabled: dict[int, ScriptSpec] = {}
    for number, spec in enumerate(SCRIPTS, 2):
        issues = script_issues(spec)
        if issues:
            ui.out(
                f"    {number}. {spec.label} (disabled: {'; '.join(issues)})",
                "dim",
            )
        else:
            enabled[number] = spec
            ui.out(f"    {number}. {spec.label}", "white")
        ui.out(f"       {spec.description}", "dim")

    def validate(value: str) -> int:
        choice = integer_value(value, 1, len(SCRIPTS) + 1)
        if choice != 1 and choice not in enabled:
            raise ValueError
        return choice

    choice = ui.ask(
        "Choose an action",
        "1",
        validate,
        "Choose Health Check or an enabled script number.",
    )
    if choice == 1:
        return None
    return enabled[choice]


def run_script(ui: UI, spec: ScriptSpec) -> int:
    """Run one script as a module with the project root as its cwd."""
    ui.header(f"Running: {spec.label}")
    ui.out(f"  {spec.description}", "dim")
    logger.info("Launching utility script %s", spec.module)
    try:
        result = subprocess.run(
            [sys.executable, "-m", spec.module],
            cwd=ROOT,
            check=False,
        )
    except OSError as exc:
        raise Stop(f"Could not start {spec.label}: {exc}") from exc
    if result.returncode:
        ui.failure(
            "SCRIPT FAILED",
            f"{spec.label} exited with code {result.returncode}. "
            "Review the output above and logs/realtime_cv.log for details.",
        )
    else:
        ui.out(f"  {spec.label} completed successfully.", "green")
    return result.returncode


def main() -> int:
    args = SimpleNamespace(plain=False, yes=False, verbose=False)
    ui = UI(args)
    ui.clear_screen()
    try:
        spec = choose_script(ui)
        if spec is None:
            return 0 if health_check(ui) else 2
        return run_script(ui, spec)
    except Stop as exc:
        ui.failure("SCRIPTS STOPPED", exc)
        return 2
    except (KeyboardInterrupt, EOFError):
        ui.out("\n  Scripts console cancelled.", "yellow")
        return 130
    except Exception as exc:
        logger.exception("Scripts console failed")
        ui.failure(
            "SCRIPTS FAILED SAFELY",
            f"{type(exc).__name__}: {exc}. See logs/realtime_cv.log for details.",
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
