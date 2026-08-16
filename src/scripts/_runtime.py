"""Shared logging bootstrap and safe output adapter for utility scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.noise_control import configure_library_noise
from src.utils import setup_logging


class ScriptOutput:
    """Send script diagnostics through project logging instead of stdout."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __call__(
            self,
            *values: Any,
            sep: str = " ",
            end: str = "\n",
            **_ignored: Any,
    ) -> None:
        self._emit(logging.INFO, *values, sep=sep, end=end)

    def debug(self, *values: Any, sep: str = " ", **_ignored: Any) -> None:
        """Record low-level script diagnostics."""
        self._emit(logging.DEBUG, *values, sep=sep)

    def info(self, *values: Any, sep: str = " ", **_ignored: Any) -> None:
        """Record normal script progress."""
        self._emit(logging.INFO, *values, sep=sep)

    def warning(self, *values: Any, sep: str = " ", **_ignored: Any) -> None:
        """Record a non-fatal script problem."""
        self._emit(logging.WARNING, *values, sep=sep)

    def error(self, *values: Any, sep: str = " ", **_ignored: Any) -> None:
        """Record a script failure."""
        self._emit(logging.ERROR, *values, sep=sep)

    def _emit(
            self,
            level: int,
            *values: Any,
            sep: str = " ",
            end: str = "\n",
    ) -> None:
        """Format output safely and send it to the configured logger."""
        message = sep.join(str(value) for value in values).rstrip()
        if not message:
            return
        # Benchmark progress previously rewrote one terminal line. Keep it
        # available for debugging without flooding normal runtime logs.
        if end != "\n":
            self._logger.debug("%s", message)
        else:
            self._logger.log(level, "%s", message)


def configure_script_output(module_name: str) -> ScriptOutput:
    """Install standard logging before a script emits diagnostic output."""
    configure_library_noise()
    setup_logging()
    if module_name == "__main__":
        source_file = getattr(sys.modules.get("__main__"), "__file__", None)
        logger_name = Path(source_file).stem if source_file else "script"
    else:
        logger_name = module_name.removeprefix("src.")
    return ScriptOutput(logging.getLogger(f"nirt_shooterbot.{logger_name}"))
