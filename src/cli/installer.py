"""Standard-library-only interactive installer for NIRT ShooterRobot."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from io import TextIOWrapper
from pathlib import Path
from types import SimpleNamespace
from typing import cast, BinaryIO

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "default.yaml"
CREATED_MODELS: list[Path] = []
CONFIG_BACKUP: bytes | None = None
CONFIG_EXISTED = False
MODELS_DIR_EXISTED = False
INSTALL_LOG: TextIOWrapper | None = None
REQUIRED = [
    ("ultralytics", "ultralytics>=8.1.28"),
    ("cv2", "opencv-python>=4.7.0"),
    ("yaml", "PyYAML>=6.0"),
    ("numpy", "numpy>=1.26"),
    ("psutil", "psutil>=5.9"),
    ("serial", "pyserial>=3.5"),
]
TORCH_SPEC = "torch>=2.5.0"
MODULE_DISTRIBUTIONS: dict[str, str] = {
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "psutil": "psutil",
    "serial": "pyserial",
    "torch": "torch",
}
MINIMUM_VERSIONS = {
    "ultralytics": ("ultralytics", "8.1.28"),
    "cv2": ("opencv-python", "4.7.0"),
    "yaml": ("PyYAML", "6.0"),
    "numpy": ("numpy", "1.26"),
    "psutil": ("psutil", "5.9"),
    "serial": ("pyserial", "3.5"),
}
OPTIONAL = {
    # FER imports pkg_resources; setuptools is therefore an intentional part
    # of this optional feature's repair set.
    "emotion": (["fer>=25.10.3", "setuptools>=65,<81"], "emotion tracking"),
    # The project uses the legacy solutions API. 0.10.21 is the last known
    # compatible release for this code path; newer releases omit it.
    "hands": (["mediapipe==0.10.21"], "hand gestures and finger mapping"),
    "semantic": (
        ["scikit-learn>=1.2.2", "sentence-transformers>=2.2.2"],
        "semantic search",
    ),
    "scipy": (["scipy>=1.10"], "faster Hungarian tracking"),
}
MANAGED_DISTRIBUTIONS = [
    "ultralytics",
    "opencv-python",
    "PyYAML",
    "numpy",
    "psutil",
    "pyserial",
    "torch",
    "torchvision",
    "torchaudio",
    "fer",
    "setuptools",
    "mediapipe",
    "scikit-learn",
    "sentence-transformers",
    "scipy",
]
MODELS = {
    "yolov8n.pt": (
        "Lightweight general detector; recommended for CPU",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        True,
    ),
    "yolov8m.pt": (
        "Higher-quality general detector; GPU recommended",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        False,
    ),
    "yolo26n-face.pt": (
        "Face detector used by the default configuration",
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt",
        True,
    ),
    "yolo26n.pt": (
        "Optional lightweight general detector already supported by the project",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        False,
    ),
    "yolov11m-face.pt": (
        "Higher-quality face detector",
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt",
        False,
    ),
    "yolov11l-face.pt": (
        "Large high-quality face detector; GPU recommended",
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt",
        False,
    ),
    "yolov10n-face.pt": (
        "Older lightweight face detector",
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-face.pt",
        False,
    ),
}


class Stop(Exception):
    pass


def log_event(text):
    """Write raw installer activity to the optional diagnostic log."""
    if INSTALL_LOG is None:
        return
    try:
        INSTALL_LOG.write(str(text).rstrip() + "\n")
        INSTALL_LOG.flush()
    except OSError:
        pass


class UI:
    C = {
        "cyan": "96",
        "blue": "94",
        "green": "92",
        "yellow": "93",
        "red": "91",
        "dim": "90",
        "white": "97",
    }

    def __init__(self, args):
        self.plain = (
                args.plain
                or bool(os.environ.get("NO_COLOR"))
                or not getattr(sys.stdout, "isatty", lambda: False)()
        )
        self.yes = args.yes
        self.verbose = args.verbose

    @staticmethod
    def clear_screen() -> None:
        """Clear the current terminal without relying on third-party tools."""
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "cls"],
                check=False,
            )
        else:
            print("\033[2J\033[H", end="", flush=True)

    def failure(self, title, reason):
        self.out(f"\n  {title}", "red")
        for line in str(reason).splitlines() or [""]:
            self.out(f"    > {line}", "red")

    def color(self, text, name=""):
        if self.plain or not name:
            return text
        return f"\033[{self.C.get(name, '0')}m{text}\033[0m"

    def out(self, text="", name=""):
        log_event(text)
        print(self.color(text, name))

    def progress(self, label, current, total=None, done=False, detail=""):
        """Render installer-owned progress; never exposes pip/download output."""
        if total:
            width = 28
            filled = min(width, int(width * current / max(1, total)))
            bar = "#" * filled + "." * (width - filled)
            text = f"  {label:<28} | [{bar}] | {current:02d}/{total:02d}"
        else:
            width = 28
            offset = int(time.monotonic() * 8) % width
            bar = ["."] * width
            for index in range(offset, min(offset + 8, width)):
                bar[index] = "#"
            text = f"  {label:<28} | [{''.join(bar)}] | --/--"
        if detail:
            text += f" | {self.fit(detail, 54)}"
        end = "\n" if done else "\r"
        rendered = self.fit(text)
        if done:
            log_event(rendered)
        print(self.color(rendered, "cyan"), end=end, flush=True)

    def progress_bytes(
            self, label, current_bytes, total_bytes=None, detail="", done=False
    ):
        if total_bytes:
            width = 28
            filled = min(
                width, int(width * current_bytes / max(1, total_bytes))
            )
            bar = "#" * filled + "." * (width - filled)
            text = (
                f"  {label:<28} | [{bar}] | "
                f"{self.bytes_text(current_bytes):>10} / {self.bytes_text(total_bytes):<10}"
            )
        else:
            width = 28
            offset = int(time.monotonic() * 8) % width
            bar = ["."] * width
            for index in range(offset, min(offset + 8, width)):
                bar[index] = "#"
            text = f"  {label:<28} | [{''.join(bar)}] | written: {self.bytes_text(current_bytes):>10}"
        if detail:
            text += f" | {self.fit(detail, 54)}"
        rendered = self.fit(text)
        if done:
            log_event(rendered)
        print(self.color(rendered), end="\n" if done else "\r", flush=True)

    @staticmethod
    def fit(text, width=None):
        width = width or shutil.get_terminal_size((120, 24)).columns
        width = max(1, width)
        if len(text) <= width:
            return text.ljust(width)
        if width <= 3:
            return text[:width]
        return text[: width - 3].rstrip() + "..."

    @staticmethod
    def bytes_text(value: int | float) -> str:
        units = ("B", "KB", "MB", "GB")
        amount = float(value)

        for unit in units:
            if amount < 1024 or unit == units[-1]:
                return f"{amount:.1f} {unit}"
            amount /= 1024

        raise RuntimeError("Unreachable")

    def header(self, text):
        self.out()
        self.out("=" * 72, "blue")
        self.out("  " + text, "cyan")
        self.out("=" * 72, "blue")

    def ask(
            self,
            text,
            default="",
            validator=None,
            invalid_message="Invalid input.",
    ):
        if self.yes:
            return default
        for attempt in range(3):
            value = self._read(text, default)
            if value == "":
                value = str(default)
            if validator is None:
                return value
            try:
                result = validator(value)
                if result is not None and result is not False:
                    return result
            except (TypeError, ValueError):
                pass
            remaining = 2 - attempt
            if remaining:
                self.out(
                    f"    > {invalid_message} ({remaining} attempt{'s' if remaining != 1 else ''} remaining.)",
                    "yellow",
                )
            else:
                raise Stop(f"Too many invalid attempts for '{text}'.")

    def _read(self, text, default=""):
        try:
            log_event(f"PROMPT: {text} [default={default}]")
            if (
                    os.name == "nt"
                    and getattr(sys.stdin, "isatty", lambda: False)()
            ):
                return self._ask_windows(text, default)
            if self.plain:
                prompt = f"  {text} [{default}]: "
            else:
                prompt = (
                    f"\033[{self.C['cyan']}m  {text} "
                    f"\033[{self.C['dim']}m[{default}]\033[0m"
                    f"\033[{self.C['cyan']}m: \033[97m"
                )
            value = input(prompt).strip()
            log_event(f"ANSWER: {value if value else default}")
            if not value:
                if not self.plain:
                    print(f"\033[{self.C['dim']}m{default}\033[0m")
                else:
                    print(default)
            elif not self.plain:
                print("\033[0m", end="")
            return value
        except (EOFError, KeyboardInterrupt):
            if not self.plain:
                print("\033[0m", end="")
            raise Stop("Input cancelled.")

    def _ask_windows(self, text, default):
        """Small dependency-free prompt editor with a gray ghost default."""
        import msvcrt

        if self.plain:
            sys.stdout.write(f"  {text} [{default}]: ")
        else:
            sys.stdout.write(
                f"\033[{self.C['cyan']}m  {text} [{default}]: \033[97m"
            )
        sys.stdout.flush()
        value = []
        while True:
            key = msvcrt.getwch()
            if key in ("\r", "\n"):
                accepted = "".join(value) or str(default)
                if not value:
                    sys.stdout.write(
                        f"\033[{self.C['dim']}m{accepted}\033[0m"
                        if not self.plain
                        else accepted
                    )
                else:
                    sys.stdout.write("\033[0m" if not self.plain else "")
                sys.stdout.write("\n")
                return "".join(value) or default
            if key == "\003":
                raise Stop("Input cancelled.")
            if key in ("\b", "\x7f"):
                if value:
                    value.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
                continue
            if key in ("\x00", "\xe0"):
                msvcrt.getwch()  # consume special-key suffix
                continue
            if not key.isprintable():
                continue
            value.append(key)
            sys.stdout.write(key)
            sys.stdout.flush()

    def yn(self, text, default=False):
        suffix = "(Y/n)" if default else "(y/N)"
        for attempt in range(3):
            log_event(f"PROMPT: {text} {suffix}")
            if self.yes:
                answer = "y" if default else "n"
            else:
                try:
                    prompt = f"  {text} {suffix}: "
                    if self.plain:
                        answer = input(prompt).strip().lower()
                        log_event(
                            f"ANSWER: {answer if answer else '<default>'}"
                        )
                    else:
                        answer = (
                            input(f"\033[{self.C['cyan']}m{prompt}\033[97m")
                            .strip()
                            .lower()
                        )
                        print("\033[0m", end="")
                except (EOFError, KeyboardInterrupt):
                    if not self.plain:
                        print("\033[0m", end="")
                    raise Stop("Input cancelled.")
            if answer == "":
                answer = "y" if default else "n"
            if answer in {"yes", "y", "1"}:
                self._rewrite_yn_result(text, suffix, "Yes", "green")
                return True
            if answer in {"no", "n", "0"}:
                self._rewrite_yn_result(text, suffix, "No", "red")
                return False
            remaining = 2 - attempt
            if remaining:
                self.out(
                    f"    > Please answer yes/y/1 or no/n/0. ({remaining} attempts remaining.)",
                    "yellow",
                )
            else:
                raise Stop(f"Too many invalid attempts for '{text}'.")

    def _rewrite_yn_result(self, text, suffix, result, color):
        """Replace the just-completed prompt row with its normalized answer."""
        if not self.plain and getattr(sys.stdout, "isatty", lambda: False)():
            # input() has already advanced to the next row. Move back, clear
            # the original row, and print the normalized result in its place.
            sys.stdout.write("\033[1A\r\033[2K")
            sys.stdout.write(
                f"\033[{self.C['cyan']}m  {text} {suffix}: "
                f"\033[{self.C[color]}m{result}\033[0m\n"
            )
            sys.stdout.flush()
        else:
            self.out(f"    > {result}", color)

    def choose(self, text, choices, default=1):
        self.out("  " + text, "cyan")
        for n, label in enumerate(choices, 1):
            display = (
                label[1]
                if isinstance(label, tuple) and len(label) > 1
                else label
            )
            self.out(f"    {'*' if n == default else ' '} {n}. {display}")
        if self.yes:
            return choices[default - 1][0]

        def valid_choice(value):
            number = integer_value(value, 1, len(choices))
            return choices[number - 1][0]

        return self.ask(
            "Choose a number",
            str(default),
            valid_choice,
            f"Choose a whole number from 1 to {len(choices)}.",
        )


def available(module: str) -> bool:
    try:
        distribution = (
            MODULE_DISTRIBUTIONS[module]
            if module in MODULE_DISTRIBUTIONS
            else module
        )
        importlib.metadata.version(distribution)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def required_ready(module):
    if not available(module):
        return False
    distribution, minimum = MINIMUM_VERSIONS[module]
    return version_at_least(distribution, minimum)


def required_runtime_ready(module):
    """Deep check used by Health Check; installation checks stay metadata-only."""
    if not required_ready(module):
        return False
    statements = {
        "ultralytics": "import ultralytics",
        "cv2": "import cv2",
        "yaml": "import yaml",
        "numpy": "import numpy",
        "psutil": "import psutil",
        "serial": "import serial",
    }
    return probe_import(statements[module])


def version_at_least(distribution, minimum):
    try:
        version = importlib.metadata.version(distribution).split("+")[0]
        current = tuple(int(part) for part in re.findall(r"\d+", version)[:3])
        target = tuple(int(part) for part in re.findall(r"\d+", minimum)[:3])
        return current >= target
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        return False


def torch_installed():
    return version_at_least("torch", "2.5.0")


def integer_value(value, minimum=0, maximum=None):
    cleaned = str(value).strip()
    if not re.fullmatch(r"\d+", cleaned):
        raise ValueError
    number = int(cleaned)
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError
    return number


def preflight(ui):
    ui.header("NIRT ShooterRobot installer")
    ui.out(f"  Python: {platform.python_version()}")
    if sys.version_info < (3, 11):
        raise Stop("Python 3.11 or newer is required.")
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or hasattr(
        sys, "real_prefix"
    )
    ui.out(
        f"  Virtual environment: {'active' if active else 'NOT ACTIVE'}",
        "green" if active else "red",
    )
    if not active:
        raise Stop(
            "Refusing to install into system Python."
            "\nRun: python -m venv .venv, activate it, then rerun python -m src.cli.installer."
        )
    if shutil.which("pip") is None:
        raise Stop(
            "pip is unavailable in this venv. Run: python -m ensurepip --upgrade"
        )
    try:
        probe = Path(sys.prefix) / ".nirt_installer_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        raise Stop(f"The active venv is not writable: {exc}")
    ui.out("  Permissions: writable", "green")
    ui.out("  Package manager: pip available", "green")


def pip(
        ui,
        packages,
        description,
        download_options=None,
        install_options=None,
        cleanup_distributions=None,
):
    if not packages:
        return
    stage = Path(tempfile.mkdtemp(prefix="nirt-install-"))
    process = None
    try:
        # Download wheels/sdists and dependencies quietly so progress is based
        # on real staged bytes rather than an arbitrary spinner.
        download_command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--progress-bar",
            "on",
            "--dest",
            str(stage),
            *(download_options or []),
            *packages,
        ]
        if ui.verbose:
            ui.out("  $ " + " ".join(download_command), "dim")
        process, output_queue = start_hidden_process(download_command)
        pip_status = {
            "text": "resolving dependencies",
            "percent": None,
            "bytes": None,
            "total_bytes": 0,
        }
        while process.poll() is None:
            pip_status = drain_pip_status(output_queue, pip_status)
            staged_count = len(package_artifacts(stage))
            staged_bytes = directory_bytes(stage)
            pip_detail = pip_status["text"]
            if pip_status["bytes"]:
                pip_detail = f"{pip_detail} ({pip_status['bytes']})"
            ui.progress_bytes(
                "Downloading packages",
                staged_bytes,
                None,
                detail=f"{staged_count:02d} artifacts | {pip_detail} | {description}",
            )
            time.sleep(0.12)
        if process.returncode:
            raise Stop(
                f"Package download failed. No installer files were changed."
            )
        total_bytes = sum(
            path.stat().st_size for path in stage.rglob("*") if path.is_file()
        )
        artifacts = package_artifacts(stage)
        artifact_total = len(artifacts)
        if artifact_total == 0:
            raise Stop("Package download produced no installable artifacts.")
        ui.progress_bytes(
            "Downloading packages",
            total_bytes,
            total_bytes,
            done=True,
            detail=f"{artifact_total:02d} artifacts | {ui.bytes_text(total_bytes)} total | {description}",
        )

        if cleanup_distributions:
            remove_incompatible_packages(ui, cleanup_distributions)

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "on",
            *(install_options or []),
            "--no-index",
            "--find-links",
            str(stage),
            *packages,
        ]
        if ui.verbose:
            ui.out("  $ " + " ".join(command), "dim")
        process, output_queue = start_hidden_process(command)
        pip_status = {
            "text": "installing staged artifacts",
            "percent": None,
            "bytes": None,
            "total_bytes": total_bytes,
        }
        while process.poll() is None:
            pip_status = drain_pip_status(output_queue, pip_status)
            ui.progress(
                "Installing packages",
                int(time.monotonic() * 8),
                None,
                detail=f"--/{artifact_total:02d} artifacts | {pip_status['text']} | {description}",
            )
            time.sleep(0.12)
        if process.returncode:
            raise Stop(
                f"Package installation failed. No installer files were changed."
            )
        ui.progress(
            "Installing packages",
            artifact_total,
            artifact_total,
            done=True,
            detail=f"{artifact_total:02d}/{artifact_total:02d} artifacts | complete | {description}",
        )
    except KeyboardInterrupt:
        if process and hasattr(process, "poll") and process.poll() is None:
            process.terminate()
        raise Stop(
            "Package installation cancelled. No installer files were changed."
        )
    except OSError as exc:
        raise Stop(f"Could not run package installation: {exc}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def remove_incompatible_packages(ui, distributions):
    """Remove only explicitly identified broken distributions before repair."""
    names = [name for name in dict.fromkeys(distributions) if available(name)]
    if not names:
        return
    ui.out(
        "  Removing incompatible package installs: " + ", ".join(names),
        "yellow",
    )
    command = [sys.executable, "-m", "pip", "uninstall", "--yes", *names]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    log_event("$ " + " ".join(command))
    log_event(result.stderr or "pip uninstall completed")
    if result.returncode:
        detail = " ".join(result.stderr.split())[:240]
        raise Stop(
            f"Could not remove incompatible packages ({', '.join(names)}): {detail}"
        )
    ui.out(
        "  Incompatible package installs removed; continuing with clean repair.",
        "green",
    )


def start_hidden_process(command):
    """Run a command silently while exposing output to the installer parser."""
    log_event("$ " + " ".join(str(item) for item in command))

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess stdout.")

    stdout = cast(BinaryIO, process.stdout)
    messages: queue.Queue[str | None] = queue.Queue()

    def reader():
        try:
            while True:
                chunk = stdout.read(256)
                if not chunk:
                    break

                decoded = chunk.decode("utf-8", errors="replace")
                log_event(decoded.rstrip())
                messages.put(decoded)
        finally:
            messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    return process, messages


def drain_pip_status(messages, current):
    while True:
        try:
            item = messages.get_nowait()
        except queue.Empty:
            return current
        if item is None:
            return current
        parts = [
            part.strip()
            for part in item.replace("\r", "\n").splitlines()
            if part.strip()
        ]
        if parts:
            text = " ".join(parts[-1].split())
            percent_match = re.search(r"(\d{1,3})%", text)
            bytes_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)\s*/\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)",
                text,
                re.IGNORECASE,
            )
            current["text"] = compact_pip_text(text)
            size_matches = re.findall(
                r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)", text, re.IGNORECASE
            )
            if size_matches:
                current["total_bytes"] = max(
                    current.get("total_bytes", 0),
                    max(
                        size_to_bytes(amount, unit)
                        for amount, unit in size_matches
                    ),
                )
            if percent_match:
                current["percent"] = max(
                    0, min(100, int(percent_match.group(1)))
                )
            if bytes_match:
                current["bytes"] = bytes_match.group(0)
        return current


def size_to_bytes(amount, unit):
    multipliers = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    return int(float(amount) * multipliers[unit.upper()])


def compact_pip_text(text):
    """Keep pip's useful current item while removing noisy long URLs."""
    if re.search(r"successfully installed", text, re.IGNORECASE):
        return "installation complete"
    collecting = re.search(r"Collecting\s+([^\s(]+)", text, re.IGNORECASE)
    if collecting:
        return f"collecting {collecting.group(1)}"
    downloading = re.search(r"Downloading\s+([^\s(]+)", text, re.IGNORECASE)
    if downloading:
        return f"downloading {Path(downloading.group(1)).name}"
    if "Installing collected packages" in text:
        return "installing dependencies"
    text = re.sub(r"https?://\S+", "download", text)
    text = re.sub(r"\s+", " ", text).strip()
    for prefix in (
            "Downloading ",
            "Collecting ",
            "Installing collected packages: ",
    ):
        if text.startswith(prefix):
            text = prefix.rstrip(": ") + ": " + text[len(prefix):]
    return text[:42]


def package_artifacts(directory):
    """Return actual distributable files staged by pip, excluding metadata."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.suffix.lower() in {".whl", ".zip", ".gz", ".bz2", ".xz"})
    )


def directory_bytes(directory):
    """Count files currently written, including pip's temporary partial files."""
    total = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def make_config(ui, force=False, assume_modify=False):
    global CONFIG_BACKUP, CONFIG_EXISTED
    CONFIG_EXISTED = CONFIG.exists()
    CONFIG_BACKUP = CONFIG.read_bytes() if CONFIG_EXISTED else None
    if CONFIG.exists() and not force and not assume_modify:
        if ui.yes:
            ui.out("  Configuration: existing file kept", "green")
            return
        modify = ui.yn("Modify the existing configuration", False)
        if not modify:
            ui.out("  Configuration: existing values kept", "green")
            return
    ui.header("Configuration")
    ui.out(
        "  Press Enter to accept defaults. Default mode is safe for click-through installs."
    )
    camera = ui.ask(
        "Camera index",
        config_value(r"^\s*sources:\s*\[?([^,\]\s]+)", "0"),
        lambda value: integer_value(value, 0),
        "Camera index must be a non-negative whole number.",
    )
    width = ui.ask(
        "Camera width",
        config_value(r"^\s*width:\s*(\d+)", "1280"),
        lambda value: integer_value(value, 1, 10000),
        "Camera width must be a whole number from 1 to 10000.",
    )
    height = ui.ask(
        "Camera height",
        config_value(r"^\s*height:\s*(\d+)", "720"),
        lambda value: integer_value(value, 1, 10000),
        "Camera height must be a whole number from 1 to 10000.",
    )
    fps = ui.ask(
        "Camera FPS",
        config_value(r"^\s*fps:\s*(\d+)", "30"),
        lambda value: integer_value(value, 1, 240),
        "Camera FPS must be a whole number from 1 to 240.",
    )
    device = ui.choose(
        "Inference device",
        [("auto", "Auto"), ("cpu", "CPU"), ("cuda", "CUDA")],
        {"auto": 1, "cpu": 2, "cuda": 3}.get(
            config_value(r"^\s*device:\s*(\w+)", "auto"), 1
        ),
    )
    remember = ui.yn(
        "Remember face IDs during this run",
        config_value(r"^\s*remember_faces:\s*(true|false)", "true").lower()
        == "true",
    )
    emotion = ui.yn(
        "Enable emotion tracking",
        config_value(r"^\s*emotion_tracking:\s*(true|false)", "true").lower()
        == "true",
    )
    hands = ui.yn(
        "Enable hand tracking",
        config_value(r"^\s*hand_tracking:\s*(true|false)", "true").lower()
        == "true",
    )
    simulation = ui.yn(
        "Use simulation fallback if camera fails",
        config_value(r"^\s*simulate_camera:\s*(true|false)", "false").lower()
        == "true",
    )
    cam = str(camera)
    if CONFIG_EXISTED and not force:
        update_existing_config(
            camera,
            width,
            height,
            fps,
            device,
            remember,
            emotion,
            hands,
            simulation,
        )
        ui.out(f"  Configuration: updated {CONFIG.relative_to(ROOT)}", "green")
        return
    text = f"""# Generated by installer.py
camera:
  sources: [{cam}]  # Camera index or source path.
  width: {width}  # Requested capture width in pixels.
  height: {height}  # Requested capture height in pixels.
  fps: {fps}  # Requested camera frame rate.
inference:
  detector: yolov8  # Primary detector family.
  model: [yolov8n.pt, yolov8m.pt]  # General model priority order.
  device: {device}  # auto, cpu, or cuda.
  confidence_threshold: 0.55  # Minimum confidence for normal detections.
  face_confidence: 0.35  # Minimum confidence for faces; see docs/Settings.md.
  ignored_classes: ['person']  # Class IDs to ignore; see docs/Detection ID.md.
  face_model: yolo26n-face.pt  # Dedicated face detector model.
  run_mode: cascade  # Run lightweight and fallback detectors in sequence.
tracking:
  track_only: ['face', 'sports ball', 'cell phone']  # Classes sent to the tracker; IDs in docs/Detection ID.md.
  max_lost: 30  # Frames a track may disappear before removal.
  iou_threshold: 0.3  # Minimum overlap used to associate detections.
  class_priority: ['face', 'sports ball', 'cell phone', 'phone', 'person']  # Queue/tracker priority; IDs in docs/Detection ID.md.
  remember_faces: {str(remember).lower()}  # Re-identify faces during this run only.
  face_memory_threshold: 0.78  # Appearance similarity required to revive a face ID.
  face_memory_max_age: 3000  # Frames to retain a remembered face.
features:
  emotion_tracking: {str(emotion).lower()}  # Add emotion labels above detected faces.
  emotion_backend: fer  # Optional emotion backend.
  hand_tracking: {str(hands).lower()}  # Detect hands, fingers, and basic gestures.
  hand_backend: mediapipe  # Optional hand-landmark backend.
  hand_gesture_map: {{'0': fist, '1': one, '2': two, '3': three, '4': four, '5': open}}  # Finger-count labels.
visualization:
  show_fps: true  # Display measured camera FPS.
  show_inference_time: true  # Display detector timing.
  center_threshold_px: 40  # Radius used to consider a target centered.
  show_tracking_queue: true  # Display the target queue in the top-right.
serial:
  enabled: false  # Enable physical serial output.
  simulation: true  # Use simulated serial output when enabled.
  crc: true  # Append CRC to telemetry packets.
logging:
  level: INFO  # Console/file logging level.
  color: true  # Use color in interactive console output.
  verbose: false  # Include source function and line details.
  file: logs/realtime_cv.log  # Runtime log file; kept under logs/.
debug:
  enabled: true  # Enable debug-only behavior.
  simulate_camera: {str(simulation).lower()}  # Use synthetic frames if requested.
  inject_fake_face: true  # Add a synthetic face in simulation mode.
"""
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(text, encoding="utf-8")
    ui.out(f"  Configuration: created {CONFIG.relative_to(ROOT)}", "green")


def update_existing_config(
        camera, width, height, fps, device, remember, emotion, hands, simulation
):
    """Update only known installer settings and preserve custom YAML content."""
    text = CONFIG.read_text(encoding="utf-8")
    fields = [
        ("camera", "sources", f"[{camera}]"),
        ("camera", "width", width),
        ("camera", "height", height),
        ("camera", "fps", fps),
        ("inference", "device", device),
        ("tracking", "remember_faces", str(remember).lower()),
        ("features", "emotion_tracking", str(emotion).lower()),
        ("features", "hand_tracking", str(hands).lower()),
        ("debug", "simulate_camera", str(simulation).lower()),
    ]
    for section, key, value in fields:
        text = set_yaml_field(text, section, key, value)
    CONFIG.write_text(text, encoding="utf-8")


def set_yaml_field(text, section, key, value):
    field_pattern = rf"^\s{{2}}{re.escape(key)}:\s*.*$"
    text, count = re.subn(
        field_pattern, f"  {key}: {value}", text, count=1, flags=re.MULTILINE
    )
    if count:
        return text
    section_pattern = rf"^{re.escape(section)}:\s*$"
    if re.search(section_pattern, text, flags=re.MULTILINE):
        return re.sub(
            section_pattern,
            f"{section}:\n  {key}: {value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    return text.rstrip() + f"\n\n{section}:\n  {key}: {value}\n"


def install_required(ui):
    missing = [spec for module, spec in REQUIRED if not required_ready(module)]
    if not missing:
        ui.out("  Required packages: already installed", "green")
        return
    cleanup = [
        MODULE_DISTRIBUTIONS[module]
        for module, _ in REQUIRED
        if not required_ready(module)
    ]
    pip(
        ui,
        missing,
        "required packages",
        install_options=["--upgrade"],
        cleanup_distributions=cleanup,
    )
    missing = [spec for module, spec in REQUIRED if not required_ready(module)]
    if missing:
        raise Stop("Required imports are still missing: " + ", ".join(missing))
    ui.out("  Required packages: ready", "green")


def install_gpu(ui, skip=False):
    if not shutil.which("nvidia-smi"):
        ui.out(
            "  GPU acceleration: unsupported or no NVIDIA GPU detected", "dim"
        )
        if not torch_installed():
            pip(
                ui,
                [TORCH_SPEC],
                "CPU PyTorch",
                install_options=["--upgrade"],
                cleanup_distributions=["torch"],
            )
        else:
            ui.out("  PyTorch: already installed and compatible", "green")
        return
    if not torch_installed():
        existing_torch = installed_version("torch")
        if existing_torch:
            ui.out(
                f"  PyTorch: installed version {existing_torch} is incompatible; CUDA setup is required.",
                "yellow",
            )
        else:
            ui.out(
                "  PyTorch 2.5+: not installed; CUDA setup is required.",
                "yellow",
            )
    else:
        ui.out("  PyTorch package: already installed and compatible", "green")
    torch_state = check_torch_cuda()
    if torch_state == "cuda" and torch_installed():
        ui.out("  CUDA PyTorch: already installed and verified", "green")
        return
    if torch_state == "cpu":
        ui.out("  PyTorch: installed, but CUDA is unavailable", "yellow")
    ui.out("  GPU acceleration: NVIDIA GPU detected", "green")
    if skip or not ui.yn("Install CUDA-enabled PyTorch", True):
        if not torch_installed():
            ui.out(
                "  Installing compatible CPU PyTorch because CUDA was skipped.",
                "yellow",
            )
            pip(
                ui,
                [TORCH_SPEC],
                "CPU PyTorch",
                install_options=["--upgrade"],
                cleanup_distributions=["torch"],
            )
            if not torch_installed():
                raise Stop(
                    "Compatible PyTorch could not be installed after CUDA was skipped."
                )
        ui.out(
            "  CUDA libraries: skipped; compatible CPU mode remains available",
            "yellow",
        )
        return
    ui.out("  PyTorch provides CUDA acceleration.", "yellow")
    pip(
        ui,
        [TORCH_SPEC, "torchvision"],
        "CUDA PyTorch",
        download_options=[
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        install_options=[
            "--upgrade",
            "--force-reinstall",
            "--ignore-installed",
        ],
        cleanup_distributions=["torch", "torchvision", "torchaudio"],
    )
    if not torch_installed():
        raise Stop(
            f"CUDA PyTorch installation did not replace the incompatible torch installation; "
            f"found {installed_version('torch') or 'missing'}."
        )
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    log_event("torch CUDA probe stdout: " + check.stdout.strip())
    log_event("torch CUDA probe stderr: " + check.stderr.strip())
    if check.returncode != 0 or check.stdout.strip() != "cuda":
        ui.out(
            "  CUDA verification: PyTorch is installed but CUDA is unavailable; CPU mode will be used.",
            "yellow",
        )
    else:
        ui.out("  CUDA verification: available", "green")


def install_optional(ui, skip=False):
    if skip:
        ui.out("  Optional packages: skipped", "yellow")
        return
    ui.out("  Optional package choices:", "cyan")
    selected = []
    cleanup = []
    selected_keys = []
    for key, (packages, LABEL) in OPTIONAL.items():
        if optional_ready(key):
            ui.out(f"    {LABEL}: already installed and verified", "green")
            continue
        if optional_distribution_present(key):
            ui.out(
                f"    {LABEL}: installed but incompatible; repair may be needed",
                "yellow",
            )
        if ui.yn(f"Install or repair {LABEL}", True):
            selected.extend(packages)
            selected_keys.append(key)
            cleanup.extend(
                {
                    "emotion": ["fer"],
                    "hands": ["mediapipe"],
                    "semantic": ["scikit-learn", "sentence-transformers"],
                    "scipy": ["scipy"],
                }[key]
            )
    if selected:
        pip(
            ui,
            selected,
            "optional packages",
            install_options=["--upgrade"],
            cleanup_distributions=cleanup,
        )
        failed = [key for key in selected_keys if not optional_ready(key)]
        if failed:
            labels = ", ".join(OPTIONAL[key][1] for key in failed)
            raise Stop(
                f"Optional package repair completed but feature verification failed: {labels}."
            )
        ui.out("  Optional packages: ready", "green")
    else:
        ui.out("  Optional packages: none selected", "dim")


def configured_feature_enabled(feature):
    patterns = {
        "emotion": r"^\s*emotion_tracking:\s*(true|false)",
        "hands": r"^\s*hand_tracking:\s*(true|false)",
    }
    return config_value(patterns[feature], "false").lower() == "true"


def health_check(ui):
    """Perform a detailed, non-mutating validation of the finished setup."""
    ui.header("Health Check")
    issues = []

    def check(LABEL, passed, detail, repair=None):
        if passed:
            ui.out(f"  [OK] {LABEL}: {detail}", "green")
        else:
            ui.out(f"  [FAIL] {LABEL}: {detail}", "red")
            issues.append({"label": LABEL, "detail": detail, "repair": repair})

    check(
        "Configuration",
        CONFIG.exists(),
        (
            f"found {CONFIG.relative_to(ROOT)}"
            if CONFIG.exists()
            else "default configuration is missing"
        ),
    )
    required_failures = [
        module for module, _ in REQUIRED if not required_runtime_ready(module)
    ]
    check(
        "Required packages",
        not required_failures,
        "installed, version-compatible, and importable",
        [spec for module, spec in REQUIRED if module in required_failures],
    )
    check(
        "PyTorch",
        torch_installed(),
        installed_version("torch") or "missing or older than 2.5",
        [TORCH_SPEC] if not torch_installed() else None,
    )

    cuda_state = check_torch_cuda() if torch_installed() else "missing"
    requested_device = config_value(r"^\s*device:\s*(\w+)", "auto").lower()
    torch_runtime_ok = torch_installed() and cuda_state != "missing"
    torch_version = installed_version("torch") or "unknown"

    check(
        "PyTorch runtime",
        torch_runtime_ok,
        (
            f"CUDA available ({torch_version})"
            if cuda_state == "cuda"
            else (
                "CPU available"
                if torch_runtime_ok
                else "PyTorch cannot be loaded safely"
            )
        ),
        [TORCH_SPEC] if cuda_state == "missing" else None,
    )
    if requested_device == "cuda":
        check(
            "Requested CUDA device",
            cuda_state == "cuda",
            (
                "CUDA is available"
                if cuda_state == "cuda"
                else "configuration requests CUDA, but CUDA is unavailable"
            ),
        )

    configured = configured_models()
    face_model = Path(config_value(r"^\s*face_model:\s*([^\s#]+)", "")).name
    model_names = list(
        dict.fromkeys(configured + ([face_model] if face_model else []))
    )
    for name in model_names:
        path = ROOT / "models" / name
        check(
            f"Model {name}",
            path.is_file() and path.stat().st_size >= 1024,
            (
                f"present ({ui.bytes_text(path.stat().st_size)})"
                if path.is_file()
                else "missing or invalid"
            ),
            None,
        )
    if not model_names:
        check(
            "Model configuration", False, "no configured detector models found"
        )

    for key, label in (
            ("emotion", "Emotion tracking"),
            ("hands", "Hand tracking"),
    ):
        if configured_feature_enabled(key):
            feature_ok = optional_ready(key)
            check(
                label,
                feature_ok,
                (
                    "backend installed and API verified"
                    if feature_ok
                    else "backend missing or API incompatible"
                ),
                OPTIONAL[key][0],
            )
        else:
            ui.out(f"  [SKIP] {label}: disabled in configuration", "dim")

    try:
        import yaml

        with CONFIG.open("r", encoding="utf-8") as stream:
            yaml.safe_load(stream)
        check("Configuration syntax", True, "valid YAML")
    except Exception as exc:
        check("Configuration syntax", False, f"could not parse YAML: {exc}")

    if not issues:
        ui.out("  Health Check: all checks passed", "green")
    else:
        ui.out(f"  Health Check: {len(issues)} issue(s) found", "yellow")
    return issues


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def repair_health(ui, issues):
    """Repair only known package issues; never overwrite user files or models."""
    ui.header("Repair")
    package_specs = []
    model_repairs = []
    repair_errors = []
    for issue in issues:
        repair = issue.get("repair")
        if isinstance(repair, list):
            package_specs.extend(repair)
        if issue["label"].startswith("Model "):
            model_repairs.append(issue["label"][6:])
    package_specs = list(dict.fromkeys(package_specs))
    for name in model_repairs:
        model = MODELS.get(name)
        destination = ROOT / "models" / name
        if model and model[1] and not destination.exists():
            try:
                download_model_asset(ui, name, model[1], destination)
            except (Stop, OSError) as exc:
                repair_errors.append(f"{name}: {exc}")
                ui.out(f"  Repair failed for {name}: {exc}", "red")
        elif model and not model[1]:
            ui.out(
                f"  {name}: no automatic repair URL; place it in models/ manually",
                "yellow",
            )
    attempted = bool(package_specs or model_repairs)
    if package_specs:
        ui.out("  Repairing missing or incompatible packages.", "yellow")
        try:
            if TORCH_SPEC in package_specs and shutil.which("nvidia-smi"):
                package_specs = [
                    item for item in package_specs if item != TORCH_SPEC
                ]
                pip(
                    ui,
                    [TORCH_SPEC, "torchvision"],
                    "CUDA PyTorch repair",
                    download_options=[
                        "--index-url",
                        "https://download.pytorch.org/whl/cu124",
                    ],
                    install_options=[
                        "--upgrade",
                        "--force-reinstall",
                        "--ignore-installed",
                    ],
                    cleanup_distributions=[
                        "torch",
                        "torchvision",
                        "torchaudio",
                    ],
                )
            if package_specs:
                cleanup = []
                for issue in issues:
                    label = issue["label"]
                    if label == "Emotion tracking":
                        cleanup.append("fer")
                    elif label == "Hand tracking":
                        cleanup.append("mediapipe")
                    elif label == "Semantic search":
                        cleanup.extend(
                            ["scikit-learn", "sentence-transformers"]
                        )
                    elif label == "Faster Hungarian tracking":
                        cleanup.append("scipy")
                pip(
                    ui,
                    package_specs,
                    "health-check repairs",
                    install_options=["--upgrade"],
                    cleanup_distributions=cleanup,
                )
        except (Stop, OSError) as exc:
            repair_errors.append(str(exc))
            ui.out(f"  Package repair failed: {exc}", "red")
    if repair_errors:
        ui.out(
            "  Some repairs failed; the repeated Health Check will show the remaining issues.",
            "yellow",
        )
    if not attempted:
        ui.out(
            "  No automatic package repair is available for the reported issues.",
            "yellow",
        )
    ui.out("  Repair pass complete; repeating Health Check.", "cyan")


def download_to_temp(
        ui,
        name: str,
        url: str,
        directory: Path,
        label: str,
) -> tuple[Path, int, int]:
    """Download a URL to a temporary file and report progress."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NIRT-ShooterRobot-installer"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)

        with tempfile.NamedTemporaryFile(
                prefix=f".{name}.",
                suffix=".part",
                dir=directory,
                delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            received = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break

                temp.write(chunk)
                received += len(chunk)

                ui.progress_bytes(
                    label,
                    received,
                    total or None,
                )

    return temp_path, received, total


def download_model_asset(ui, name, url, destination):
    """Download one model atomically for the Health Check repair pass."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None

    try:
        temp_path, received, total = download_to_temp(
            ui,
            name,
            url,
            destination.parent,
            f"Repairing {name}",
        )

        if received < 1024 or (total and received != total):
            raise Stop(f"Incomplete model repair for {name}.")

        temp_path.replace(destination)
        CREATED_MODELS.append(destination)

        ui.progress_bytes(
            f"Repairing {name}",
            received,
            received,
            done=True,
            detail=f"{ui.bytes_text(received)} repaired",
        )
        ui.out(f"  {name}: repaired", "green")

    except (urllib.error.URLError, OSError, Stop) as exc:
        if temp_path:
            temp_path.unlink(missing_ok=True)

        if isinstance(exc, Stop):
            raise

        raise Stop(f"Could not repair {name}: {exc}")


def run_health_check(ui):
    issues = health_check(ui)
    if not issues:
        return
    repair_health(ui, issues)
    remaining = health_check(ui)
    if remaining:
        details = "; ".join(
            f"{item['label']}: {item['detail']}" for item in remaining
        )
        raise Stop(
            "Health Check still has unresolved issues. "
            + details
            + " Try activating the venv, checking model files in models/, or reviewing logs/installer.log."
        )


def check_torch_cuda():
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "TF_CPP_MIN_LOG_LEVEL": "3",
                "GLOG_minloglevel": "3",
                "ABSL_MIN_LOG_LEVEL": "3",
                "PYTHONWARNINGS": "ignore",
            },
        )
    except subprocess.TimeoutExpired:
        return "missing"
    log_event("torch probe stdout: " + result.stdout.strip())
    log_event("torch probe stderr: " + result.stderr.strip())
    if result.returncode != 0:
        return "missing"
    return (
        result.stdout.strip()
        if result.stdout.strip() in {"cuda", "cpu"}
        else "missing"
    )


def optional_ready(key):
    try:
        if key == "emotion" and not version_at_least("fer", "25.10.3"):
            return False
        if key == "hands" and installed_version("mediapipe") != "0.10.21":
            return False
        if key == "semantic" and not (
                version_at_least("scikit-learn", "1.2.2")
                and version_at_least("sentence-transformers", "2.2.2")
        ):
            return False
        if key == "scipy" and not version_at_least("scipy", "1.10"):
            return False
        if key == "emotion":
            return probe_import("from fer.fer import FER")
        if key == "hands":
            return probe_import(
                "import mediapipe as mp; assert hasattr(mp, 'solutions')"
            )
        if key == "semantic":
            return probe_import("import sklearn, sentence_transformers")
        if key == "scipy":
            return probe_import("import scipy")
    except Exception:
        return False
    return False


def optional_distribution_present(key):
    """Report package presence separately from feature/API compatibility."""
    distributions = {
        "emotion": ("fer",),
        "hands": ("mediapipe",),
        "semantic": ("scikit-learn", "sentence-transformers"),
        "scipy": ("scipy",),
    }
    return all(available(name) for name in distributions[key])


def probe_import(statement):
    env = os.environ.copy()
    env.update(
        {
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "GLOG_minloglevel": "3",
            "ABSL_MIN_LOG_LEVEL": "3",
            "TRANSFORMERS_VERBOSITY": "error",
            "PYTHONWARNINGS": "ignore",
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", statement],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    log_event(f"probe: {statement} (exit={result.returncode})")
    if result.stdout:
        log_event("probe stdout: " + result.stdout.strip())
    if result.stderr:
        log_event("probe stderr: " + result.stderr.strip())
    return result.returncode == 0


def install_models(ui):
    """Offer model downloads and atomically install each completed file."""
    global MODELS_DIR_EXISTED
    ui.header("Models")
    catalog = model_catalog()
    configured = configured_models()
    configured_face = Path(
        config_value(r"^\s*face_model:\s*([^\s#]+)", "")
    ).name
    has_model_settings = bool(configured or configured_face)
    edit_model_settings = True
    if has_model_settings:
        ui.out("  Existing model settings detected:", "cyan")
        if configured:
            ui.out("    General order: " + " -> ".join(configured), "green")
        if configured_face:
            ui.out("    Face model: " + configured_face, "green")
        edit_model_settings = ui.yn("Edit existing model settings", False)
        if not edit_model_settings:
            ui.out("  Existing model settings will be preserved.", "green")
    ui.out(
        "  Select model numbers to install, or press Enter for defaults.",
        "cyan",
    )
    general = [
        (n, item)
        for n, item in enumerate(catalog, 1)
        if "face" not in item[0].lower()
    ]
    faces = [
        (n, item)
        for n, item in enumerate(catalog, 1)
        if "face" in item[0].lower()
    ]
    print_model_group(ui, "General detection models", general)
    print_model_group(ui, "Face detection models", faces)
    defaults = ",".join(
        str(n) for n, (_, (_, _, default)) in enumerate(catalog, 1) if default
    )

    def valid_models(value):
        tokens = [
            token.strip() for token in str(value).split(",") if token.strip()
        ]
        if not tokens:
            raise ValueError
        numbers = {integer_value(token, 1, len(catalog)) for token in tokens}
        return ",".join(str(number) for number in sorted(numbers))

    raw = ui.ask(
        "Models to install (comma-separated numbers)",
        defaults,
        valid_models,
        f"Use comma-separated whole numbers from 1 to {len(catalog)}.",
    )
    selected_numbers = {int(value) for value in raw.split(",")}
    model_dir = ROOT / "models"
    MODELS_DIR_EXISTED = model_dir.exists()
    model_dir.mkdir(parents=True, exist_ok=True)
    selected = catalog
    for index, (name, (info, url, _)) in enumerate(selected, 1):
        if index not in selected_numbers:
            continue
        destination = model_dir / name
        if destination.exists():
            ui.out(f"  {name}: already present", "green")
            continue
        if not url:
            ui.out(
                f"  {name}: no public download URL configured; place it in models/ manually",
                "yellow",
            )
            continue
        temp_path = None
        try:
            temp_path, received, total = download_to_temp(
                ui,
                name,
                url,
                model_dir,
                f"Downloading {name}",
            )
            if total and received != total:
                raise Stop(f"Incomplete download for {name}.")
            if temp_path.stat().st_size < 1024:
                raise Stop(f"Downloaded model {name} is unexpectedly small.")
            temp_path.replace(destination)
            CREATED_MODELS.append(destination)
            ui.progress_bytes(
                f"Downloading {name}",
                received,
                received,
                done=True,
                detail=f"{ui.bytes_text(received)} downloaded",
            )
            ui.out(f"  {name}: installed", "green")
        except (urllib.error.URLError, OSError, Stop) as exc:
            if temp_path:
                temp_path.unlink(missing_ok=True)
            if isinstance(exc, Stop):
                raise
            raise Stop(
                f"Could not download {name}: {exc}. Partial files were removed."
            )

    configure_model_choices(ui, model_dir, allow_edit=edit_model_settings)

    required_models = ("yolov8n.pt", "yolo26n-face.pt")
    missing_required = [
        name for name in required_models if not (model_dir / name).exists()
    ]
    if missing_required:
        raise Stop(
            "Required model files are missing: "
            + ", ".join(missing_required)
            + ". Select them in the Models section or place them in models/ manually."
        )


def model_catalog():
    """Return models in stable general-first, face-second order."""
    general = [
        (name, data)
        for name, data in MODELS.items()
        if "face" not in name.lower()
    ]
    faces = [
        (name, data) for name, data in MODELS.items() if "face" in name.lower()
    ]
    return general + faces


def print_model_group(ui, title, entries):
    ui.out(f"  {title}", "blue")
    for number, (name, (info, _url, default)) in entries:
        installed = (ROOT / "models" / name).is_file()
        status = "installed" if installed else "not installed"
        status_color = "green" if installed else "yellow"
        recommendation = "recommended" if default else "optional"
        prefix = "*" if default else "-"
        ui.out(
            f"    {prefix} {number}. {name:<20} - {info} ({recommendation})",
            "cyan",
        )
        ui.out(f"        status: {status}", status_color)


def config_value(pattern, fallback=""):
    if not CONFIG.exists():
        return fallback
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return fallback
    value = match.group(1).split("#", 1)[0].strip().strip("'\"")
    return value or fallback


def configured_models():
    raw = config_value(r"^\s*model:\s*\[([^]]*)\]")
    return [
        Path(item.strip(" '\"")).name
        for item in raw.split(",")
        if item.strip()
    ]


def update_config_models(general, face):
    if not CONFIG.exists():
        return
    text = CONFIG.read_text(encoding="utf-8")
    text = set_yaml_field(
        text, "inference", "model", "[" + ", ".join(general) + "]"
    )
    text = set_yaml_field(text, "inference", "face_model", face)
    CONFIG.write_text(text, encoding="utf-8")


def configure_model_choices(ui, model_dir, allow_edit=True):
    installed = [name for name in MODELS if (model_dir / name).exists()]
    general = [name for name in installed if "face" not in name.lower()]
    faces = [name for name in installed if "face" in name.lower()]
    configured_general = [
        name for name in configured_models() if name in general
    ]
    configured_face = Path(
        config_value(r"^\s*face_model:\s*([^\s#]+)", "")
    ).name
    if len(faces) > 1:
        if configured_face in faces:
            ui.out(
                f"  Face model selected by config: {configured_face}", "green"
            )
            if not allow_edit:
                pass
        else:
            if not allow_edit:
                ui.out(
                    "  Face model settings preserved; no configured face model was changed.",
                    "yellow",
                )
                configured_face = ""
            if ui.yes:
                raise Stop(
                    "Multiple face models are installed but the configuration does not select one."
                )
            ui.out(
                "  Multiple face models are installed; choose exactly one.",
                "yellow",
            )
            choices = [(str(i), name) for i, name in enumerate(faces, 1)]
            selected_face = ui.choose("Face model", choices, 1)
            selected_face = faces[int(selected_face) - 1]
            if ui.yn("Save this face-model choice to the configuration", True):
                update_config_models(
                    configured_general or general[:1], selected_face
                )
    if len(general) > 1:
        if (
                configured_general
                and len(configured_general) == len(general)
                and set(configured_general) == set(general)
        ):
            ui.out(
                "  General model order from config: "
                + " -> ".join(configured_general),
                "green",
            )
        elif not allow_edit:
            ui.out("  General model order settings preserved.", "green")
        elif ui.yes:
            raise Stop(
                "Multiple general models are installed but the configuration does not define their order."
            )
        else:
            ui.out(
                "  Multiple general models are installed; specify their priority order.",
                "yellow",
            )
            raw = ui.ask(
                "General model order (comma-separated names)",
                ",".join(general),
                lambda value: validate_model_order(value, general),
                "Use every installed general-model filename exactly once.",
            )
            ordered = raw.split(",")
            if ui.yn(
                    "Save this general-model order to the configuration", True
            ):
                update_config_models(
                    ordered,
                    (
                        configured_face or faces[0]
                        if faces
                        else "yolo26n-face.pt"
                    ),
                )


def validate_model_order(value, available_names):
    names = [
        Path(item.strip(" '\"")).name
        for item in str(value).split(",")
        if item.strip()
    ]
    if len(names) != len(available_names) or set(names) != set(
            available_names
    ):
        raise ValueError
    return ",".join(names)


def rollback():
    """Remove only artifacts created by this installer invocation."""
    for model in CREATED_MODELS:
        try:
            model.unlink(missing_ok=True)
        except OSError:
            pass


def action_menu(ui):
    """Choose the installer operation after state-aware preflight checks."""
    initialized = CONFIG.exists()
    ui.header("What would you like to do?")
    actions = [
        (
            "health",
            "Health Check",
            initialized,
            "Program hasn't initialised yet",
        ),
        ("install", "Install", True, "always available"),
        (
            "clean",
            "Wipe and Install",
            True,
            "destructive: removes generated project data",
        ),
        (
            "basic",
            "Modify Config (Basic)",
            initialized,
            "Program hasn't initialised yet",
        ),
        (
            "advanced",
            "Modify Config (Advanced)",
            initialized,
            "Program hasn't initialised yet",
        ),
    ]
    for index, (_key, label, enabled, reason) in enumerate(actions, 1):
        if enabled:
            ui.out(f"    {index}. {label}", "white")
        else:
            ui.out(f"    {index}. {label} (disabled: {reason})", "dim")

    def validate(value):
        number = integer_value(value, 1, len(actions))
        action = actions[number - 1]
        if not action[2]:
            raise ValueError
        return action[0]

    return ui.ask(
        "Choose an action",
        "2",
        validate,
        "Choose an enabled action number from the list.",
    )


def wipe_generated_project_data(ui):
    """Remove only known installer/runtime-generated project paths."""
    targets = [
        ROOT / "configs",
        ROOT / "models",
        ROOT / "logs",
        ROOT / ".cache",
        ROOT / "files",
        ROOT / "src" / "scripts" / "models",
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        ui.out(
            "  The following generated project paths will be removed:",
            "yellow",
        )
        for path in existing:
            ui.out(f"    - {path.relative_to(ROOT)}", "yellow")
    if not ui.yn("Delete these generated files and folders", False):
        raise Stop(
            "Clean Install cancelled before any project files were removed."
        )
    for path in existing:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise Stop(
                f"Could not remove generated path {path.relative_to(ROOT)}: {exc}"
            )
    ui.out("  Generated project data removed.", "green")


def remove_managed_packages(ui):
    installed = [name for name in MANAGED_DISTRIBUTIONS if available(name)]
    if not installed:
        ui.out("  No managed pip distributions are installed.", "dim")
        return
    ui.out("  Managed distributions selected for removal:", "yellow")
    ui.out("    " + ", ".join(installed), "yellow")
    if not ui.yn("Remove these project-managed pip distributions", False):
        ui.out("  Managed pip distributions kept.", "green")
        return
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", *installed],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    log_event(result.stdout)
    if result.returncode:
        raise Stop(
            "Managed pip distribution removal failed. See logs/installer.log for details."
        )
    ui.out("  Managed pip distributions removed.", "green")


ADVANCED_DEFAULTS = {
    ("camera", "sources"): "[0]",
    ("camera", "backend_preference"): "[]",
    ("inference", "detector"): "yolov8",
    ("inference", "model"): "[auto]",
    ("inference", "device"): "auto",
    ("inference", "confidence_threshold"): "0.55",
    ("inference", "face_confidence"): "0.35",
    ("inference", "classes"): "[]",
    ("inference", "ignored_classes"): "[]",
    ("inference", "min_area"): "0.0",
    ("inference", "nms_iou_threshold"): "0.45",
    ("inference", "max_models_to_load"): "5",
    ("inference", "face_model"): "yolo26n-face.pt",
    ("inference", "max_detector_time"): "0.5",
    ("inference", "run_mode"): "cascade",
    ("tracking", "max_lost"): "30",
    ("tracking", "iou_threshold"): "0.3",
    ("tracking", "min_hits"): "1",
    ("tracking", "max_age"): "30",
    ("tracking", "track_only"): "[face, sports ball, cell phone]",
    ("tracking", "cycle_remember"): "true",
    ("tracking", "remember_faces"): "false",
    ("tracking", "face_memory_threshold"): "0.78",
    ("tracking", "face_memory_max_age"): "300",
    ("features", "emotion_tracking"): "false",
    ("features", "hand_tracking"): "false",
    ("visualization", "font_scale"): "0.6",
    ("visualization", "center_threshold_px"): "40",
    ("visualization", "show_tracking_queue"): "true",
    ("serial", "enabled"): "false",
    ("serial", "port"): "COM3",
    ("serial", "baudrate"): "115200",
    ("logging", "level"): "INFO",
    ("logging", "verbose"): "false",
    ("logging", "file"): "logs/realtime_cv.log",
    ("debug", "simulate_camera"): "false",
    ("debug", "simulation_video"): "",
}


def advanced_default(section, key):
    return ADVANCED_DEFAULTS[(section, key)]


def advanced_config(ui):
    """Edit all scalar configuration values without requiring a YAML library."""
    global CONFIG_BACKUP, CONFIG_EXISTED
    if not CONFIG.exists():
        raise Stop(
            "Configuration file is missing. Run Install or Modify Config (Basic) "
            "first to create configs/default.yaml."
        )
    # Advanced editing is transactional: keep the original bytes untouched
    # while prompts and validation are still in progress.
    CONFIG_EXISTED = True
    backup = CONFIG.read_bytes()
    CONFIG_BACKUP = backup
    staged_text = backup.decode("utf-8")
    ui.header("Advanced Configuration")
    ui.out(
        "  Press Enter to keep each current value. Advanced changes preserve the existing file structure."
    )
    fields = [
        (
            "camera",
            "sources",
            r"^\s*sources:\s*(.+)$",
            "Camera source list",
            None,
        ),
        (
            "camera",
            "backend_preference",
            r"^\s*backend_preference:\s*(.+)$",
            "Camera backend preference list",
            None,
        ),
        (
            "inference",
            "detector",
            r"^\s*detector:\s*(.+)$",
            "Detector family",
            None,
        ),
        (
            "inference",
            "model",
            r"^\s*model:\s*(.+)$",
            "General model priority list",
            None,
        ),
        (
            "inference",
            "device",
            r"^\s*device:\s*(.+)$",
            "Inference device",
            None,
        ),
        (
            "inference",
            "confidence_threshold",
            r"^\s*confidence_threshold:\s*(.+)$",
            "Normal confidence threshold",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "inference",
            "face_confidence",
            r"^\s*face_confidence:\s*(.+)$",
            "Face confidence threshold",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "inference",
            "classes",
            r"^\s*classes:\s*(.+)$",
            "Allowed class IDs; see docs/Detection ID.md",
            None,
        ),
        (
            "inference",
            "ignored_classes",
            r"^\s*ignored_classes:\s*(.+)$",
            "Ignored class IDs; see docs/Detection ID.md",
            None,
        ),
        (
            "inference",
            "min_area",
            r"^\s*min_area:\s*(.+)$",
            "Minimum detection area",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "inference",
            "nms_iou_threshold",
            r"^\s*nms_iou_threshold:\s*(.+)$",
            "Detection overlap suppression threshold",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "inference",
            "max_models_to_load",
            r"^\s*max_models_to_load:\s*(.+)$",
            "Maximum loaded models",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        (
            "inference",
            "face_model",
            r"^\s*face_model:\s*(.+)$",
            "Face model filename",
            None,
        ),
        (
            "inference",
            "max_detector_time",
            r"^\s*max_detector_time:\s*(.+)$",
            "Maximum detector time in seconds",
            lambda lambda_value: float_value(lambda_value, 0.0),
        ),
        (
            "inference",
            "run_mode",
            r"^\s*run_mode:\s*(.+)$",
            "Detector run mode",
            None,
        ),
        (
            "tracking",
            "max_lost",
            r"^\s*max_lost:\s*(.+)$",
            "Frames before a lost track is removed",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        (
            "tracking",
            "iou_threshold",
            r"^\s*iou_threshold:\s*(.+)$",
            "Track association overlap threshold",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "tracking",
            "min_hits",
            r"^\s*min_hits:\s*(.+)$",
            "Hits required to confirm a track",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        (
            "tracking",
            "max_age",
            r"^\s*max_age:\s*(.+)$",
            "Maximum track age",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        (
            "tracking",
            "track_only",
            r"^\s*track_only:\s*(.+)$",
            "Tracked classes; see docs/Detection ID.md",
            None,
        ),
        (
            "tracking",
            "cycle_remember",
            r"^\s*cycle_remember:\s*(.+)$",
            "Requeue targets after a shot",
            bool_value,
        ),
        (
            "tracking",
            "remember_faces",
            r"^\s*remember_faces:\s*(.+)$",
            "Remember face IDs during this run",
            bool_value,
        ),
        (
            "tracking",
            "face_memory_threshold",
            r"^\s*face_memory_threshold:\s*(.+)$",
            "Face appearance match threshold",
            lambda lambda_value: float_value(lambda_value, 0.0, 1.0),
        ),
        (
            "tracking",
            "face_memory_max_age",
            r"^\s*face_memory_max_age:\s*(.+)$",
            "Face memory lifetime in frames",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        (
            "features",
            "emotion_tracking",
            r"^\s*emotion_tracking:\s*(.+)$",
            "Enable face emotion labels",
            bool_value,
        ),
        (
            "features",
            "hand_tracking",
            r"^\s*hand_tracking:\s*(.+)$",
            "Enable hand landmarks and gestures",
            bool_value,
        ),
        (
            "visualization",
            "font_scale",
            r"^\s*font_scale:\s*(.+)$",
            "Overlay font scale",
            lambda lambda_value: float_value(lambda_value, 0.0),
        ),
        (
            "visualization",
            "center_threshold_px",
            r"^\s*center_threshold_px:\s*(.+)$",
            "Centered-target radius in pixels",
            integer_value,
        ),
        (
            "visualization",
            "show_tracking_queue",
            r"^\s*show_tracking_queue:\s*(.+)$",
            "Show the target queue",
            bool_value,
        ),
        (
            "serial",
            "enabled",
            r"^\s*enabled:\s*(.+)$",
            "Enable serial output",
            bool_value,
        ),
        ("serial", "port", r"^\s*port:\s*(.+)$", "Serial port", None),
        (
            "serial",
            "baudrate",
            r"^\s*baudrate:\s*(.+)$",
            "Serial baud rate",
            lambda lambda_value: integer_value(lambda_value, 1),
        ),
        ("logging", "level", r"^\s*level:\s*(.+)$", "Logging level", None),
        (
            "logging",
            "verbose",
            r"^\s*verbose:\s*(.+)$",
            "Verbose logging",
            bool_value,
        ),
        ("logging", "file", r"^\s*file:\s*(.+)$", "Runtime log path", None),
        (
            "debug",
            "simulate_camera",
            r"^\s*simulate_camera:\s*(.+)$",
            "Use camera simulation",
            bool_value,
        ),
        (
            "debug",
            "simulation_video",
            r"^\s*simulation_video:\s*(.+)$",
            "Simulation video path",
            None,
        ),
    ]
    for section, key, pattern, label, validator in fields:
        current = config_value(
            pattern, advanced_default(section, key)
        )
        if validator is None:
            value = ui.ask(label, current)
        else:
            value = ui.ask(
                label, current, validator, f"Invalid value for {label}."
            )
        staged_text = set_yaml_field(staged_text, section, key, value)
    write_config_atomically(staged_text)
    ui.out(
        f"  Configuration: advanced settings updated {CONFIG.relative_to(ROOT)}",
        "green",
    )


def float_value(value, minimum=0.0, maximum=None):
    number = float(str(value).strip())
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError
    return number


def bool_value(value):
    cleaned = str(value).strip().lower()
    if cleaned in {"true", "yes", "y", "1"}:
        return "true"
    if cleaned in {"false", "no", "n", "0"}:
        return "false"
    raise ValueError


def write_config_atomically(text):
    """Commit a completed configuration without exposing a partial write."""
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=CONFIG.parent,
                prefix=f".{CONFIG.name}.",
                suffix=".tmp",
                delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CONFIG)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise Stop(
            "Could not save the configuration safely. "
            "The existing configs/default.yaml was not changed. "
            f"Check that it is writable and try again ({exc})."
        ) from exc


def main():
    global INSTALL_LOG
    args = SimpleNamespace(plain=False, yes=False, verbose=False)
    log_path = ROOT / "logs" / "installer.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        INSTALL_LOG = log_path.open("a", encoding="utf-8", buffering=1)
        log_event(
            f"\n--- installer started {time.strftime('%Y-%m-%d %H:%M:%S')} ---"
        )
    except OSError as exc:
        print(f"Could not open installer log {log_path}: {exc}")
        return 3
    ui = UI(args)
    ui.clear_screen()
    try:
        preflight(ui)
        action = action_menu(ui)
        if action == "health":
            issues = health_check(ui)
            if issues:
                details = "; ".join(
                    f"{item['label']}: {item['detail']}" for item in issues
                )
                ui.failure("HEALTH CHECK FAILED", details)
                return 2
            ui.out(
                "  Health Check completed successfully; no changes were made.",
                "green",
            )
            return 0
        if action == "basic":
            make_config(ui, assume_modify=True)
            return 0
        if action == "advanced":
            advanced_config(ui)
            return 0
        if action == "clean":
            ui.header("Clean Install")
            ui.out(
                "  This removes generated configs, models, logs, caches, and files.",
                "yellow",
            )
            if not ui.yn("Continue with Clean Install", False):
                raise Stop("Clean Install cancelled.")
            if INSTALL_LOG is not None:
                INSTALL_LOG.close()
                INSTALL_LOG = None
            wipe_generated_project_data(ui)
            remove_managed_packages(ui)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            INSTALL_LOG = log_path.open("a", encoding="utf-8", buffering=1)
            log_event(
                f"\n--- clean install resumed {time.strftime('%Y-%m-%d %H:%M:%S')} ---"
            )
        make_config(ui)
        ui.header("Installation")
        install_required(ui)
        install_gpu(ui)
        install_models(ui)
        install_optional(ui)
        run_health_check(ui)
        ui.header("Installation complete")
        ui.out(
            "  Start: python -m src.cli.run --config configs/default.yaml",
            "green",
        )
        return 0
    except Stop as exc:
        rollback()
        ui.failure("INSTALLATION STOPPED", exc)
        return 2
    except (KeyboardInterrupt, EOFError):
        rollback()
        ui.out("\n  Installation cancelled.", "yellow")
        return 130
    except Exception as exc:
        rollback()
        log_event("Unexpected installer error:\n" + traceback.format_exc())
        ui.failure(
            "INSTALLATION FAILED SAFELY",
            f"{type(exc).__name__}: {exc}. See logs/installer.log for details.",
        )
        return 3
    finally:
        # Keep the final prompt/output visually separated from installer output.
        print()
        log_event("--- installer finished ---")
        if INSTALL_LOG is not None:
            INSTALL_LOG.close()
            INSTALL_LOG = None


if __name__ == "__main__":
    raise SystemExit(main())
