"""Inference module wrapping Ultralytics YOLO and optional additional detectors."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np
from ultralytics.engine.results import Results

from .utils import iou

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

logger = logging.getLogger("realtime_cv.inference")


def choose_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA unavailable, falling back to CPU. "
                    "torch.cuda.is_available() returned False. "
                    "torch version=%s, CUDA build=%s",
                    torch.__version__,
                    torch.version.cuda,
                )

                if (
                        hasattr(torch.cuda, "is_built")
                        and not torch.cuda.is_built()
                ):
                    logger.warning(
                        "PyTorch was installed without CUDA support. "
                        "Install a CUDA-enabled PyTorch build."
                    )
                else:
                    device_count = torch.cuda.device_count()
                    logger.warning(
                        "CUDA devices detected by PyTorch: %s", device_count
                    )

                return "cpu"

            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                logger.debug("Could not enable cuDNN benchmark", exc_info=True)

            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)

            logger.info(
                "Using CUDA device: %s (index=%s/%s), torch CUDA=%s",
                gpu_name,
                current_device,
                gpu_count,
                torch.version.cuda,
            )

            return "cuda"

        except ImportError:
            logger.warning("PyTorch is not installed, falling back to CPU")
            return "cpu"

        except Exception:
            logger.exception(
                "Failed while checking CUDA availability, falling back to CPU"
            )
            return "cpu"

    logger.info("Using manually selected device: %s", device)
    return device


class YOLODetector:
    """Wrapper around ultralytics YOLO model with automatic model selection based on device."""

    def __init__(
            self,
            model: str = "yolov8n.pt",
            device: str = "auto",
            cache_dir: Optional[str] = None,
            prefer_high_quality: bool = True,
    ):
        try:
            from ultralytics import YOLO
        except Exception:
            logger.error(
                "ultralytics package is required. Install with `pip install ultralytics`"
            )
            raise

        self.imgsz = 640
        self.device = choose_device(device)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else ROOT / ".cache" / "realtime_cv_models"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # model selection: support 'auto' to pick model based on device
        selected_model = model
        if isinstance(model, str) and model == "auto":
            # choose model: tiny for cpu, medium/extra for cuda depending on preference
            if self.device.startswith("cuda") and prefer_high_quality:
                selected_model = "yolov8x.pt"  # top-tier
            elif self.device.startswith("cuda"):
                selected_model = "yolov8m.pt"
            else:
                selected_model = "yolov8n.pt"
        self.model_name = resolve_model_path(str(selected_model))
        # If running on CPU, prefer smaller model for realtime; downgrade very large models automatically
        if self.device.startswith("cpu") and self.model_name in (
                "yolov8x.pt",
                "yolov8m.pt",
        ):
            logger.warning(
                "Running on CPU: switching model %s -> yolov8n.pt for better realtime performance",
                self.model_name,
            )
            self.model_name = str(MODEL_DIR / "yolov8n.pt")
        logger.info(
            f"Loading YOLO model {self.model_name} on device {self.device}"
        )
        # Let ultralytics handle downloading; instantiate model.
        #  If download fails, attempt to fall back to local yolov8n.pt
        try:
            self.model = YOLO(self.model_name)
            try:
                # move model to device
                self.model.to(self.device)
            except Exception:
                pass
            # if running on CUDA, use half precision for best throughput
            if self.device.startswith("cuda"):
                try:
                    self.model.to(self.device)

                    logger.info(
                        "Enabled FP16 (half precision) for model on CUDA"
                    )

                except Exception:
                    logger.debug(
                        "Could not enable half precision for model",
                        exc_info=True,
                    )

                # CUDA warmup
                try:
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

                    self.model.predict(
                        dummy,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                        quantize="fp16",
                    )

                    logger.info("CUDA warmup complete")

                except Exception:
                    logger.debug("CUDA warmup failed", exc_info=True)
        except Exception:
            import traceback

            logger.exception(
                "Failed to load requested model %s via ultralytics, attempting local fallback.",
                self.model_name,
            )
            local_fallback = MODEL_DIR / "yolov8n.pt"
            if local_fallback.exists():
                try:
                    self.model = YOLO(str(local_fallback))
                    logger.info(
                        "Loaded local fallback model %s", str(local_fallback)
                    )

                    try:
                        self.model.to(self.device)
                    except Exception:
                        logger.debug(
                            "Could not move fallback model to %s",
                            self.device,
                            exc_info=True,
                        )
                    if self.device.startswith("cuda"):
                        try:
                            self.model.to(self.device)
                            logger.info("Model moved to CUDA")
                        except Exception:
                            logger.debug(
                                "Could not move fallback model to CUDA",
                                exc_info=True,
                            )
                except Exception as e:
                    logger.exception(
                        "Failed to load local fallback model: %s", e
                    )
                    raise
            else:
                logger.error(
                    "No fallback model found at %s. Install ultralytics and ensure network access or place a model at "
                    "that path.",
                    str(local_fallback),
                )
                raise

    def predict(
            self, frame: np.ndarray, conf: float = 0.25
    ) -> List[Dict[str, Any]]:

        h, w = map(int, frame.shape[:2])
        proc = frame
        scale_x = 1.0
        scale_y = 1.0

        # Limit resolution for realtime inference
        max_dimension = 640

        if w > max_dimension or h > max_dimension:
            if w >= h:
                new_w = max_dimension
                new_h = int(h * max_dimension / w)
            else:
                new_h = max_dimension
                new_w = int(w * max_dimension / h)

            proc = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            scale_x = w / new_w
            scale_y = h / new_h

        try:
            results = self.model.predict(
                proc,
                conf=conf,
                imgsz=self.imgsz,
                verbose=False,
                stream=False,
                device=self.device,
                quantize="fp16" if self.device.startswith("cuda") else None,
            )

        except Exception:
            logger.exception("YOLO inference failed")
            return []

        detections = []

        for result in results:
            if not isinstance(result, Results):
                logger.error(
                    "YOLO returned unexpected result type: %s",
                    type(result).__name__,
                )
                continue

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                try:
                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = xyxy

                    detections.append(
                        {
                            "bbox": [
                                float(x1 * scale_x),
                                float(y1 * scale_y),
                                float(x2 * scale_x),
                                float(y2 * scale_y),
                            ],
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "label": self.model.names[int(box.cls[0])],
                        }
                    )

                except Exception:
                    continue

        return detections


class InferenceResult(TypedDict):
    detections: List[Dict[str, Any]]
    inference_time_ms: float
    timestamp: float


class InferenceWorker:
    """Background inference worker to offload model prediction and reduce UI lag.
    Submit frames with submit(camera_idx, frame, ts), and read the latest results via get(camera_idx).
    """

    def __init__(self, detector_manager: "DetectorManager"):
        import threading
        from queue import Queue

        self.detector_manager = detector_manager
        # keep queue size small to minimize latency; drop frames when busy
        self._queue: "Queue" = Queue(maxsize=1)
        self._results: Dict[int, InferenceResult] = {}
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        import threading

        if self._running:
            return
        self._running = True
        # create a fresh thread each time start is called
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

        try:
            self.detector_manager.shutdown()
        except Exception:
            pass

        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
        except Exception:
            pass

    def thread_is_alive(self) -> bool:
        """Return whether the inference worker thread is currently alive."""
        return self._thread.is_alive()

    def submit(self, camera_idx: int, frame: Any, timestamp: float) -> None:
        try:
            self._queue.put_nowait((camera_idx, frame.copy(), timestamp))
        except Exception:
            # drop if queue full
            pass

    def _loop(self):
        import time
        from queue import Empty

        # noinspection PyProtectedMember
        while self._running and not self.detector_manager._shutdown:
            try:
                camera_idx, frame, ts = self._queue.get(timeout=0.02)
            except Empty:
                time.sleep(0.002)
                continue
            t0 = time.time()
            try:
                dets = self.detector_manager.predict(frame)
            except Exception:
                logger.exception(
                    "InferenceWorker: detector_manager.predict crashed"
                )
                dets = []
            t1 = time.time()
            self._results[camera_idx] = {
                "detections": dets,
                "inference_time_ms": (t1 - t0) * 1000.0,
                "timestamp": ts,
            }

    def get(self, camera_idx: int) -> InferenceResult | None:
        return self._results.get(camera_idx)


def resolve_model_path(model: str) -> str:
    """
    Resolve model paths relative to project ROOT/models directory.
    """

    path = Path(model)

    # Already absolute
    if path.is_absolute():
        return str(path)

    # Already starts with models/
    if path.parts and path.parts[0].lower() == MODEL_DIR.name.lower():
        return str(ROOT / path)

    # Plain filename
    return str(MODEL_DIR / path)


class DetectorManager:
    """Manage multiple detectors (YOLO + optional face detector) and provide filtering and text search.

    Supports loading multiple YOLO models concurrently (e.g., primary YOLO, small fallback, and a face-specific model)
    and runs them in parallel for lower latency when multiple CPU cores are available.
    """

    def __init__(self, config: Any, cache_dir: Optional[str] = None):
        self._shutdown = False
        self.config = config
        self.cache_dir = cache_dir
        # detectors structured as name -> {'type': str, 'obj': detector_obj, 'priority': int}
        self.detectors: Dict[str, Dict[str, Any]] = {}
        self.device = "cpu"

        # Load main YOLO models list: support single model or a list in config.inference.multi_models
        prefer_hq = getattr(config.inference, "prefer_high_quality", True)
        models_to_load: List[str] = []

        configured_models = getattr(config.inference, "model", [])

        if isinstance(configured_models, list):
            models_to_load = [str(m) for m in configured_models]
        elif configured_models:
            models_to_load = [str(configured_models)]

        if getattr(config.inference, "auto_load_models_from_cwd", True):
            max_models = int(
                getattr(config.inference, "max_models_to_load", 0) or 0
            )
            discovered = []
            seen_names = {Path(m).name.lower() for m in models_to_load}
            for search_root in (MODEL_DIR,):
                if not search_root.exists():
                    continue
                for candidate in sorted(search_root.glob("*.pt")):
                    candidate_name = candidate.name.lower()
                    if candidate_name in seen_names:
                        continue
                    discovered.append(str(candidate))
                    seen_names.add(candidate_name)
                    if 0 < max_models <= len(models_to_load) + len(discovered):
                        break
                if 0 < max_models <= len(models_to_load) + len(discovered):
                    break
            models_to_load.extend(discovered)

        # ensure a small fallback exists for CPU realtime
        if not any(Path(m).name == "yolov8n.pt" for m in models_to_load):
            models_to_load.append(str(MODEL_DIR / "yolov8n.pt"))

        # instantiate YOLODetector objects for each requested model name
        idx = 0
        loaded_model_names = set()
        for m in models_to_load:
            try:
                if "face" in Path(m).stem.lower():
                    continue

                m = resolve_model_path(m)

                yd = YOLODetector(
                    m,
                    config.inference.device,
                    cache_dir=cache_dir,
                    prefer_high_quality=prefer_hq,
                )

                # skip duplicate model names (e.g., two configs mapping to yolov8n.pt on CPU)
                base = Path(getattr(yd, "model_name", str(m))).name.lower()
                if base in loaded_model_names:
                    logger.info("Skipping duplicate model load for %s", base)
                    continue
                loaded_model_names.add(base)
                key = f"yolo_{idx}"
                # default priority: prefer face-specific models if name contains 'face' or provided as face_model
                pr = 2
                try:
                    if "n" in str(m):
                        pr = 1
                    if "face" in str(m).lower():
                        pr = 3
                except Exception:
                    pass
                self.detectors[key] = {
                    "type": "yolo",
                    "obj": yd,
                    "priority": pr,
                }
                self.device = yd.device
                idx += 1
            except Exception:
                logger.warning("Failed to load YOLO model %s; skipping", m)

        # If user provided a face model name without setting face_model, try to auto-discover face-specific .pt files
        # in cwd or cache
        face_candidate = getattr(config.inference, "face_model", None)

        # If discovered, and not already loaded, append
        if isinstance(face_candidate, str) and face_candidate:
            try:
                # normalize and resolve candidate path/name to avoid relative path issues
                cand = Path(resolve_model_path(str(face_candidate)))
                # if not absolute/existing, try repo-root relative paths
                if not cand.exists():
                    # repo root is package parent
                    repo_root = Path(__file__).resolve().parents[1]
                    alt = repo_root / cand.name
                    if alt.exists():
                        cand = alt
                cand_path = str(cand)
                cand_name = Path(cand_path).name.lower()

                if not cand.exists():
                    logger.warning(
                        "Face model is missing: %s; skipping invalid YOLO face load",
                        cand_path,
                    )
                    cand_path = None

                # robust duplicate check: compare against loaded YOLODetector.model_name when possible
                def _is_same_model(obj_meta):
                    try:
                        _obj = obj_meta.get("obj")
                        # YOLODetector exposes model_name attribute
                        mname = getattr(_obj, "model_name", None)
                        if (
                                isinstance(mname, str)
                                and mname
                                and Path(str(mname)).name.lower() == cand_name
                        ):
                            return True
                    except Exception:
                        pass
                    return False

                already = any(
                    _is_same_model(v) for k, v in self.detectors.items()
                )
                if cand_path and not already:
                    # load face-specific model as a YOLO detector as well (user requested using all models
                    # simultaneously)
                    yd = YOLODetector(
                        cand_path,
                        config.inference.device,
                        cache_dir=cache_dir,
                        prefer_high_quality=prefer_hq,
                    )
                    key = f"yolo_{idx}_face"
                    # give face-specific model highest priority
                    self.detectors[key] = {
                        "type": "yolo",
                        "obj": yd,
                        "priority": 10,
                    }
                    self.device = yd.device
                    idx += 1
                    logger.info("Loaded additional face YOLO model as %s", key)
            except Exception:
                logger.warning(
                    "Failed to auto-load face model %s", face_candidate
                )

        # optional fallback face detector
        # Only load this if a YOLO face model was not already loaded above.
        disabled = set(
            getattr(config.inference, "disabled_detectors", []) or []
        )

        has_yolo_face = any(
            meta.get("priority") == 10 for meta in self.detectors.values()
        )

        if "face" not in disabled and not has_yolo_face:
            try:
                from .face_detector import FaceDetector

                fd = FaceDetector(
                    cache_dir=None,
                    conf=getattr(config.inference, "face_confidence", 0.5),
                    model_path=getattr(config.inference, "face_model", None),
                )

                if (getattr(fd, "net", None) is not None) or (
                        getattr(fd, "yolo", None) is not None
                ):
                    self.detectors["face"] = {
                        "type": "face",
                        "obj": fd,
                        "priority": 3,
                    }

                    self.device = self.device or "cpu"

            except Exception:
                logger.debug("Face detector not available: %s", exc_info=True)
                pass

        # set per-detector run intervals to avoid CPU thrashing when many heavy models are present on CPU
        self._last_run: Dict[str, float] = {}
        self._intervals: Dict[str, float] = {}
        for name, meta in self.detectors.items():
            try:
                obj = meta.get("obj")
                model_name = (
                    getattr(obj, "model_name", "") if obj is not None else ""
                )
                # if running on CPU and model looks heavy (contains 'x' or 'm') or it's a face-specific large model,
                # throttle it
                if self.device.startswith("cpu"):
                    if (
                            "face" in name.lower()
                            or "face" in str(model_name).lower()
                    ):
                        interval = getattr(
                            self.config.inference, "face_model_interval", 0.4
                        )
                    elif (
                            "x" in str(model_name).lower()
                            or "m" in str(model_name).lower()
                    ):
                        interval = getattr(
                            self.config.inference, "heavy_model_interval", 0.3
                        )
                    else:
                        interval = 0.0
                else:
                    interval = 0.0
            except Exception:
                interval = 0.0
            self._intervals[name] = float(interval)

    def _dedupe_by_iou(
            self, dets: List[Dict[str, Any]], iou_thr: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Greedy deduplication with class-priority support.

        - Sort by confidence descending. - For overlapping detections (IoU >= iou_thr), keep the one with higher
        class priority (configured via config.inference.class_priority) or fallback to higher confidence.
        """
        if not dets:
            return []
        # build class priority mapping (higher index -> higher priority)
        priority_list = (
                getattr(self.config.tracking, "class_priority", None)
                or getattr(self.config.inference, "class_priority", [])
                or []
        )
        priority_map = {
            name.lower(): i for i, name in enumerate(priority_list[::-1])
        }  # reverse so earlier in list = higher priority

        def class_prio(_d):
            name = (_d.get("class_name") or _d.get("label") or "").lower()
            return priority_map.get(name, 0)

        def _bbox_tuple(
                det: Dict[str, Any],
        ) -> Tuple[float, float, float, float]:
            x1, y1, x2, y2 = det["bbox"]
            return float(x1), float(y1), float(x2), float(y2)

        # sort by confidence descending as baseline
        dets_sorted = sorted(
            dets,
            key=lambda x: (class_prio(x), float(x.get("confidence", 0.0))),
            reverse=True,
        )
        keep: List[Dict[str, Any]] = []
        for d in dets_sorted:
            placed = False
            for k in list(keep):
                if iou(_bbox_tuple(d), _bbox_tuple(k)) >= iou_thr:
                    # decide by class priority first
                    pd = class_prio(d)
                    pk = class_prio(k)
                    if pd > pk:
                        # replace
                        keep.remove(k)
                        keep.append(d)
                        placed = True
                        break
                    elif pd == pk:
                        # keep higher confidence (dets_sorted ensures higher confidence first)
                        placed = True
                        break
                    else:
                        placed = True
                        break
            if not placed:
                keep.append(d)
        return keep

    def _filter_detections(
            self, dets: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Apply confidence, class and size filters from config.

        Accepts class names or ids in config.inference.classes and config.inference.ignored_classes.
        """
        h, w = frame_shape[:2]
        out = []
        # prepare allowed set (can be names or ids)
        allowed = None
        if getattr(self.config.inference, "classes", None):
            allowed = set(self.config.inference.classes)
        raw_ignored = (
                getattr(self.config.inference, "ignored_classes", []) or []
        )
        ignored_names = set()
        ignored_ids = set()
        # translate ignored entries: if "string", map to COCO id when possible
        try:
            from .coco import COCO_CLASSES

            name_to_id = {n: i for i, n in enumerate(COCO_CLASSES)}
        except Exception:
            name_to_id = {}
        for item in raw_ignored:
            if isinstance(item, str):
                item_l = item.lower()
                if item_l.isdigit():
                    ignored_ids.add(int(item_l))
                else:
                    # map known name
                    if item_l in name_to_id:
                        ignored_ids.add(name_to_id[item_l])
                    else:
                        ignored_names.add(item_l)
            elif isinstance(item, int):
                ignored_ids.add(item)
        for d in dets:
            conf = float(d.get("confidence", 0.0))
            if conf < getattr(
                    self.config.inference, "confidence_threshold", 0.25
            ):
                continue
            cls = int(d.get("class_id", -1))
            cls_name = (
                str(d.get("class_name", "")).lower()
                if d.get("class_name")
                else str(d.get("label", "")).lower()
            )
            # allowed filtering (support names)
            if allowed is not None:
                # if allowed contains strings, compare names
                if any(isinstance(x, str) for x in allowed):
                    if cls_name not in {s.lower() for s in allowed}:
                        continue
                else:
                    if cls not in allowed:
                        continue
            # ignored filtering by id or name
            if cls in ignored_ids or (cls_name and cls_name in ignored_names):
                continue
            x1, y1, x2, y2 = d["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            # size filtering can be added to config in percentage of frame
            min_area = getattr(self.config.inference, "min_area", 0)
            if min_area and area < (min_area * w * h):
                continue
            # normalize bbox and center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d["center"] = [cx, cy]
            d["normalized_center"] = [cx / w, cy / h]
            d["bbox_normalized"] = [x1 / w, y1 / h, x2 / w, y2 / h]
            out.append(d)
        # deduplicate overlapping detections keeping highest-confidence
        out = self._dedupe_by_iou(
            out,
            iou_thr=getattr(self.config.inference, "nms_iou_threshold", 0.5),
        )
        return out

    def shutdown(self):
        self._shutdown = True

    def predict(self, frame: Any) -> List[Dict[str, Any]]:
        if self._shutdown:
            return []

        dets: List[Dict[str, Any]] = []

        if not self.detectors:
            return dets

        run_mode = getattr(self.config.inference, "run_mode", "cascade")

        yolo_candidates = [
            (name, meta)
            for name, meta in self.detectors.items()
            if meta.get("type") == "yolo"
        ]

        face_yolo_candidates = [
            (name, meta)
            for name, meta in yolo_candidates
            if meta.get("priority") == 10
        ]

        object_yolo_candidates = [
            (name, meta)
            for name, meta in yolo_candidates
            if meta.get("priority") != 10
        ]

        to_run = []

        if run_mode == "cascade":
            if object_yolo_candidates:
                best_object = max(
                    object_yolo_candidates,
                    key=lambda item: item[1].get("priority", 0),
                )
                to_run.append(best_object)

            if face_yolo_candidates:
                best_face = max(
                    face_yolo_candidates,
                    key=lambda item: item[1].get("priority", 0),
                )
                to_run.append(best_face)

        else:
            to_run.extend(yolo_candidates)

        if self._shutdown:
            return []

        now = time.time()

        for name, meta in to_run:

            if self._shutdown:
                break

            interval = self._intervals.get(name, 0.0)
            last = self._last_run.get(name, 0.0)

            if (now - last) < interval:
                continue

            obj = meta["obj"]
            d_type = meta["type"]

            try:
                start = time.perf_counter()

                if d_type == "yolo":
                    res = obj.predict(frame)
                else:
                    res = obj.predict(frame)

                elapsed = (time.perf_counter() - start) * 1000

                logger.debug("%s inference %.2fms", name, elapsed)

                self._last_run[name] = now

                for r in res:
                    r["detector"] = name
                    r["detector_priority"] = int(meta.get("priority", 0))

                    if meta.get("type") == "face":
                        r["label"] = r.get("label", "face")
                        r["class_id"] = r.get("class_id", -1)

                    dets.append(r)

            except Exception:
                logger.exception("Detector %s failed", name)

        # Force face-specific models to be labeled correctly
        try:
            for d in dets:
                det_name = str(d.get("detector", "")).lower()

                if (
                        "face" in det_name
                        or int(d.get("detector_priority", 0)) >= 10
                ):
                    d["label"] = "face"
                    d["class_name"] = "face"
                    d["class_id"] = -1

            from .coco import COCO_CLASSES

            for d in dets:
                if d.get("class_name"):
                    continue

                cid = int(d.get("class_id", -1))

                if 0 <= cid < len(COCO_CLASSES):
                    d["class_name"] = COCO_CLASSES[cid]
                else:
                    d["class_name"] = d.get("label", "unknown")

        except Exception:
            pass

        dets = self._filter_detections(dets, frame.shape)

        # Debug fake face injection
        if (
                getattr(self.config, "debug", None)
                and getattr(self.config.debug, "simulate_camera", False)
                and getattr(self.config.debug, "inject_fake_face", False)
                and not dets
        ):
            h, w = frame.shape[:2]

            bw = int(w * 0.2)
            bh = int(h * 0.25)

            x1 = w // 2 - bw // 2
            y1 = h // 2 - bh // 2

            dets.append(
                {
                    "bbox": [x1, y1, x1 + bw, y1 + bh],
                    "confidence": 0.95,
                    "class_id": -1,
                    "label": "face",
                    "class_name": "face",
                    "detector": "sim",
                    "detector_priority": 5,
                }
            )

        return dets

    @staticmethod
    def text_search(
            query: str, method: str = "fuzzy"
    ) -> List[Tuple[str, int, float]]:
        try:
            from .coco import find_closest_class

            return find_closest_class(query, method=method)
        except Exception:
            return []
