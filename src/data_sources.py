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

# SHA-256 pins for model artifacts that are distributed by the project.
# A model without a pin is deliberately never downloaded automatically.
MODEL_SHA256: Final[dict[str, str]] = {
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "yolov8m.pt": "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
    "yolo26n-face.pt": "c6a5405127a2e351292315a6a8084ea3e790dbec25b9d16a8e80d1e3f866efe1",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "yolov11m-face.pt": "6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661",
    "yolov11l-face.pt": "ba0000a5945ef6c4b5841b06478ff9a55649073d01cf98f382dd4991100c9514",
    "yolov10n-face.pt": "58bc4397f6e5a1cd69411cf46615e60b9bd89a00c1dfd92307f873f626528c18",
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

# Avoid filling disks if a remote source responds with an unexpectedly large file.
# Model downloads for this project are substantially smaller.
MAX_EXTERNAL_DOWNLOAD_BYTES: Final = 2 * 1024 * 1024 * 1024
