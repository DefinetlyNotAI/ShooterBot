"""Keep third-party diagnostics inside the application's logging system."""

from __future__ import annotations

import logging
import os
import threading
import warnings


_native_stderr_lock = threading.Lock()
_native_stderr_started = False


def _native_log_level(message: str) -> int:
    """Map common native-library prefixes to Python logging levels."""
    upper = message.upper()
    if upper.startswith(("FATAL", "ERROR", "E")):
        return logging.ERROR
    if "WARNING" in upper or upper.startswith("W"):
        return logging.WARNING
    if upper.startswith("DEBUG"):
        return logging.DEBUG
    return logging.INFO


def start_native_stderr_bridge() -> None:
    """Route native stderr diagnostics through the project's logger once."""
    global _native_stderr_started
    with _native_stderr_lock:
        if _native_stderr_started or os.environ.get(
            "NIRT_CAPTURE_NATIVE_STDERR", "1"
        ) == "0":
            return
        try:
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 2)
            os.close(write_fd)
        except OSError:
            return
        _native_stderr_started = True

    def forward() -> None:
        library_logger = logging.getLogger("nirt_shooterbot.library")
        try:
            with os.fdopen(
                read_fd, "r", encoding="utf-8", errors="replace"
            ) as stream:
                for line in stream:
                    message = line.strip()
                    if message:
                        library_logger.log(_native_log_level(message), "%s", message)
        except OSError:
            return

    threading.Thread(
        target=forward,
        name="native-stderr-logger",
        daemon=True,
    ).start()


def configure_library_noise() -> None:
    """Set native-library quiet flags and route Python warnings to logging."""
    # These must be set before TensorFlow/MediaPipe/absl are imported.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "default")

    warnings.filterwarnings("default")
    warnings.filterwarnings(
        "ignore", message="pkg_resources is deprecated as an API"
    )
    # TensorFlow emits this as a multiline UserWarning. Match the originating
    # module and stable prefix instead of relying on a single-line substring.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"tensorflow\.lite\.python\.interpreter",
        message=r"[\s\S]*tf\.lite\.Interpreter is deprecated[\s\S]*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"[\s\S]*LiteRT interpreter[\s\S]*",
    )
    logging.captureWarnings(True)
    for name in (
            "absl",
            "tensorflow",
            "tensorflow Lite",
            "transformers",
            "huggingface_hub",
            "urllib3",
            "mediapipe",
            "matplotlib",
            "PIL",
            "torch",
            "ultralytics",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


configure_library_noise()
