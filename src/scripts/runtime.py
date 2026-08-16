"""Shared logging bootstrap and safe output adapter for utility scripts."""

from __future__ import annotations

import logging
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
        message = sep.join(str(value) for value in values).rstrip()
        if not message:
            return
        # Benchmark progress previously rewrote one terminal line. Keep it
        # available for debugging without flooding normal runtime logs.
        if end != "\n":
            self._logger.debug("%s", message)
        else:
            self._logger.info("%s", message)


def configure_script_output(module_name: str) -> ScriptOutput:
    """Install standard logging before a script emits diagnostic output."""
    configure_library_noise()
    setup_logging()
    logger_name = module_name.removeprefix("src.")
    return ScriptOutput(logging.getLogger(f"realtime_cv.{logger_name}"))
