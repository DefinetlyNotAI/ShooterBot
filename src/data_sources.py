"""Approved external data sources used by the installer and runtime."""

from __future__ import annotations

from typing import Final

PYTORCH_CUDA_WHEEL_INDEX: Final = "https://download.pytorch.org/whl/cu124"

MODEL_DOWNLOAD_URLS: Final[dict[str, str]] = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
    "yolo26n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt",
    "yolo26n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolov11m-face.pt": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt",
    "yolov11l-face.pt": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt",
    "yolov10n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-face.pt",
}

OPENCV_FACE_PROTO_URL: Final = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
OPENCV_FACE_MODEL_URL: Final = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# GitHub release downloads may redirect to these GitHub-controlled hosts.
TRUSTED_DOWNLOAD_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "github-releases.githubusercontent.com",
        "download.pytorch.org",
    }
)

# Avoid filling disks if a remote source responds with an unexpectedly large
# file. Model downloads for this project are substantially smaller.
MAX_EXTERNAL_DOWNLOAD_BYTES: Final = 2 * 1024 * 1024 * 1024
