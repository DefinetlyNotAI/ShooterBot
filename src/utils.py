"""Utility helpers and logging setup."""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional as _Opt
from typing import Tuple


class _PrettyFormatter(logging.Formatter):
    """Readable colored console formatter; file output deliberately stays plain."""

    _COLORS = {
        logging.DEBUG: "\033[38;5;245m",  # soft gray
        logging.INFO: "\033[38;5;75m",  # blue
        logging.WARNING: "\033[38;5;220m",  # amber
        logging.ERROR: "\033[38;5;203m",  # coral
        logging.CRITICAL: "\033[1;38;5;196m",
    }
    _RESET = "\033[0m"
    _DIM = "\033[2m"

    def __init__(
            self,
            color: bool = True,
            verbose: bool = False,
            max_width: int | None = None,
    ):
        self.color = color
        self.verbose = verbose
        self.max_width = max_width
        self.project_root = Path(__file__).resolve().parents[1]
        super().__init__()

    @staticmethod
    def _source_name(record: logging.LogRecord) -> str:
        name = record.name.removeprefix("realtime_cv.")
        return "app" if name == "realtime_cv" else name

    def _relative_paths(self, text: str) -> str:
        root = str(self.project_root).replace("\\", "/").rstrip("/")
        # Normalize absolute paths belonging to this project, including Windows paths.
        pattern = re.compile(re.escape(root) + r"[\\/]?", re.IGNORECASE)
        return pattern.sub("./", text.replace("\\", "/"))

    def _wrap_message(self, message: str, prefix: str) -> list[str]:
        """Wrap without truncating or splitting words.

        The first line receives the structured log prefix. Every continuation
        line is indented to the beginning of the message column.
        """
        width = self.max_width or shutil.get_terminal_size((120, 24)).columns
        width = max(1, int(width))
        message_width = max(1, width - len(prefix))
        continuation = " " * len(prefix)
        result: list[str] = []
        for paragraph in message.splitlines() or [""]:
            wrapped = textwrap.wrap(
                paragraph,
                width=message_width,
                initial_indent="",
                subsequent_indent="",
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=True,
            ) or [""]
            for index, part in enumerate(wrapped):
                result.append((prefix if not result else continuation) + part)
        return result

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        timestamp = f"{timestamp}.{record.msecs:03.0f}"
        level = record.levelname[:8]
        name = self._source_name(record)
        location = (
            f" @{record.funcName}:{record.lineno}" if self.verbose else ""
        )
        message = self._relative_paths(record.getMessage())
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        prefix = f"{timestamp:>12} | {level:<8} | {name:<14}{location} | "
        lines = self._wrap_message(message, prefix)
        line = "\n".join(lines)
        if not self.color:
            return line
        colored_prefix = (
            f"{self._DIM}{timestamp:>12}{self._RESET} | "
            f"{self._COLORS.get(record.levelno, self._RESET)}{level:<8}{self._RESET} | "
            f"{name:<14}{location} | "
        )
        plain_prefix_len = len(prefix)
        colored_lines = []
        for index, line in enumerate(lines):
            content = (
                line[plain_prefix_len:]
                if index == 0
                else line[plain_prefix_len:]
            )
            colored_lines.append(
                (colored_prefix if index == 0 else " " * plain_prefix_len)
                + content
            )
        return "\n".join(colored_lines)


def setup_logging(
        level: str = "INFO",
        logfile: _Opt[str] = None,
        color: bool = True,
        verbose: bool = False,
) -> None:
    """Configure colorful console logs and plain, searchable file logs."""
    try:
        from .noise_control import configure_library_noise

        configure_library_noise()
    except Exception:
        pass
    configured_level = getattr(logging, level.upper(), logging.INFO)
    lvl = logging.DEBUG if verbose else configured_level
    # Native-library stderr is bridged into logging, so application logs must
    # use stdout to avoid feeding the bridge back into itself.
    stream = logging.StreamHandler(sys.stdout)
    # Avoid escape sequences in redirected output and CI logs.
    use_color = bool(
        color
        and hasattr(stream.stream, "isatty")
        and stream.stream.isatty()
        and not os.environ.get("NO_COLOR")
    )
    stream.setFormatter(_PrettyFormatter(color=use_color, verbose=verbose))
    handlers = [stream]
    log_path = (
        Path(logfile)
        if logfile
        else Path(__file__).resolve().parents[1] / "logs" / "realtime_cv.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(file_handler)
    logging.basicConfig(level=lvl, handlers=handlers, force=True)
    try:
        from .noise_control import start_native_stderr_bridge

        start_native_stderr_bridge()
    except Exception:
        pass


logger = logging.getLogger("realtime_cv")


def current_milli() -> int:
    return int(time.time() * 1000)


def bbox_center(
        bbox: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float],
) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if areaA + areaB - interArea == 0:
        return 0.0
    return interArea / float(areaA + areaB - interArea)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_point(
        x: float, y: float, width: int, height: int
) -> Tuple[float, float]:
    return x / width, y / height


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
