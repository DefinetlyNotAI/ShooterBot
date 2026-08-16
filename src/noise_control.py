"""Keep third-party diagnostics inside the application's logging system."""

from __future__ import annotations

import logging
import os
import warnings


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
