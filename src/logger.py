"""Compatibility shim: expose setup_logging and logger from utils."""

from .utils import setup_logging, logger

__all__ = ["setup_logging", "logger"]
