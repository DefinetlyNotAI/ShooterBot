"""Face detector wrapper supporting multiple backends.

- If a .pt model path is provided, will use ultralytics YOLO for face detection (high accuracy).
- Otherwise falls back to OpenCV DNN SSD face detector (downloaded to cache).

FaceDetector.predict returns list of dicts with keys: bbox, confidence, class_id=-1, label='face'.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from .data_sources import (
    MAX_EXTERNAL_DOWNLOAD_BYTES,
    OPENCV_FACE_MODEL_URL,
    OPENCV_FACE_PROTO_URL,
    TRUSTED_DOWNLOAD_HOSTS,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / ".cache" / "nirt_shooterbot_models"

logger = logging.getLogger("nirt_shooterbot.face")

PROTO_URL = OPENCV_FACE_PROTO_URL
MODEL_URL = OPENCV_FACE_MODEL_URL


def _download(url: str, dst: Path) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.hostname.lower() not in TRUSTED_DOWNLOAD_HOSTS
    ):
        raise ValueError("Face model source is not an approved HTTPS host")
    if dst.exists():
        return
    logger.info(f"Downloading {url} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(dst.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "NIRT-ShooterRobot"})
        with urllib.request.urlopen(request, timeout=60) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if (
                    final.scheme != "https"
                    or final.hostname is None
                    or final.hostname.lower() not in TRUSTED_DOWNLOAD_HOSTS
            ):
                raise ValueError("Face model redirect resolved to an unapproved host")
            with temporary.open("wb") as stream:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > MAX_EXTERNAL_DOWNLOAD_BYTES:
                        raise ValueError("Face model download exceeds the safety size limit")
                    stream.write(chunk)
        os.replace(temporary, dst)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class FaceDetector:
    def __init__(
            self,
            cache_dir: Path | None = None,
            conf: float = 0.5,
            model_path: Optional[str] = None,
    ):
        """If model_path endswith .pt, use ultralytics YOLO for face detection. Otherwise, use OpenCV DNN SSD."""
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conf = conf
        self.model_path = model_path
        self.backend = "opencv"
        self.net = None
        self.yolo = None
        # if user provided a path to a .pt model, try ultralytics
        if model_path:
            # allow names: try adding .pt if missing and searching common locations
            candidate = str(model_path)
            if not candidate.lower().endswith(".pt"):
                candidate_pt = candidate + ".pt"
            else:
                candidate_pt = candidate
            # Keep model discovery inside the project model/cache directories.
            candidates = [
                MODEL_DIR / Path(candidate_pt).name,
                self.cache_dir / Path(candidate_pt).name,
            ]
            found = None
            for c in candidates:
                if Path(c).exists():
                    found = str(Path(c).resolve())
                    break
            if found is None:
                # also try raw model_path if it's a full path
                if Path(model_path).exists():
                    found = str(Path(model_path).resolve())
            if found:
                try:
                    from ultralytics import YOLO

                    self.yolo = YOLO(found)
                    self.backend = "yolo"
                    logger.info("Loaded YOLO face model %s", found)
                except Exception:
                    logger.warning(
                        "Failed to load YOLO face model %s; falling back to OpenCV DNN",
                        found,
                    )
                    self.yolo = None
                    self.backend = "opencv"
            else:
                logger.info(
                    "face_model %s not found on disk; will use OpenCV DNN if available",
                    model_path,
                )
        if self.backend == "opencv":
            self.proto = self.cache_dir / "deploy.prototxt"
            self.model = (
                    self.cache_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            )
            if not self.proto.exists() or not self.model.exists():
                try:
                    _download(PROTO_URL, self.proto)
                    _download(MODEL_URL, self.model)
                except Exception as e:
                    logger.warning("Could not download face model: %s", e)
            # Try OpenCV DNN first
            if self.model.exists() and self.proto.exists():
                try:
                    self.net = cv2.dnn.readNetFromCaffe(
                        str(self.proto), str(self.model)
                    )
                    logger.info("Loaded OpenCV DNN face detector")
                except Exception as e:
                    logger.warning(
                        "Failed to initialize OpenCV DNN face detector: %s", e
                    )
                    self.net = None
            # If DNN not available, try Haar cascade as last-resort (works in headless builds)
            if self.net is None:
                try:
                    cascade_path = (
                            cv2.data.haarcascades
                            + "haarcascade_frontalface_default.xml"
                    )
                    if Path(cascade_path).exists():
                        self.cascade = cv2.CascadeClassifier(cascade_path)
                        # noinspection PyUnresolvedReferences
                        if not self.cascade.empty():
                            logger.info(
                                "Loaded Haar cascade face detector as fallback"
                            )
                        else:
                            self.cascade = None
                    else:
                        self.cascade = None
                except Exception as e:
                    logger.debug("Haar cascade not available: %s", e)
                    self.cascade = None

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]
        if self.backend == "yolo" and self.yolo is not None:
            try:
                # run YOLO on a resized frame for speed, then map coordinates back
                small = (
                    cv2.resize(frame, (640, int(640 * h / w)))
                    if w > h
                    else cv2.resize(frame, (int(640 * w / h), 640))
                )
                results = self.yolo.predict(
                    small, conf=self.conf, verbose=False
                )
                for r in results:
                    boxes = getattr(r, "boxes", [])
                    for b in boxes:
                        xyxy = (
                            b.xyxy[0].cpu().numpy()
                            if hasattr(b.xyxy[0], "cpu")
                            else np.array(b.xyxy[0])
                        )
                        # scale back to original size
                        sx, sy = small.shape[1], small.shape[0]
                        scale_x = w / sx
                        scale_y = h / sy
                        x1, y1, x2, y2 = [
                            float(xyxy[0]) * scale_x,
                            float(xyxy[1]) * scale_y,
                            float(xyxy[2]) * scale_x,
                            float(xyxy[3]) * scale_y,
                        ]
                        conf_score = (
                            float(b.conf[0])
                            if hasattr(b, "conf")
                            else float(b.conf)
                        )
                        if conf_score >= self.conf:
                            out.append(
                                {
                                    "bbox": [
                                        int(x1),
                                        int(y1),
                                        int(x2),
                                        int(y2),
                                    ],
                                    "confidence": conf_score,
                                    "class_id": -1,
                                    "label": "face",
                                }
                            )
            except Exception:
                logger.debug("YOLO face predict failed", exc_info=True)
                return out
            return out
        # fallback to OpenCV DNN
        # If DNN available, use it
        if getattr(self, "net", None) is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            # noinspection PyUnresolvedReferences
            self.net.setInput(blob)
            # noinspection PyUnresolvedReferences
            detections = self.net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > self.conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    out.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_id": -1,
                            "label": "face",
                        }
                    )
            return out
        # If cascade available, use it
        if getattr(self, "cascade", None) is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # noinspection PyUnresolvedReferences
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, wbox, hbox in faces:
                x1, y1, x2, y2 = x, y, x + wbox, y + hbox
                out.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": 0.8,
                        "class_id": -1,
                        "label": "face",
                    }
                )
            return out
        # nothing available
        return out
