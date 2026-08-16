"""Camera handling using OpenCV with support for multiple sources and simulation mode."""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraStream:
    """Threaded camera capture that yields frames."""

    def __init__(
            self,
            source=0,
            width: int = 1280,
            height: int = 720,
            fps: int = 30,
            simulate: bool = False,
            sim_video: str = "",
            inject_fake_face: bool = False,
            backend_preference: Optional[list] = None,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.simulate = simulate
        self.sim_video = sim_video
        # when injecting fake face in inference, don't draw an extra circle on the frame
        self.inject_fake_face = inject_fake_face
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._queue: Queue = Queue(maxsize=4)
        self._running = False
        # store backend preference provided by caller (list of friendly names or constants)
        self.backend_preference = backend_preference

    def start(self) -> None:
        import logging, platform

        logger = logging.getLogger("realtime_cv.camera")
        if self._running:
            return
        # simulation path
        if self.simulate:
            if self.sim_video:
                logger.info("Opening simulation video %s", self.sim_video)
                self._cap = cv2.VideoCapture(self.sim_video)
            else:
                logger.info("Starting synthetic simulation camera")
                self._cap = None
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._thread.start()
            return

        # try to open real camera with multiple backends for robustness
        # allow user to provide backend_preference as friendly names
        user_pref = None
        if hasattr(self, "backend_preference") and self.backend_preference:
            user_pref = self.backend_preference
        backends = []
        sys_plat = platform.system().lower()
        if user_pref:
            # map friendly names to cv2 constants where possible
            # build mapping of friendly names to cv2 constants, but only include constants present in this cv2 build
            raw_map = {
                "dshow": "CAP_DSHOW",
                "msmf": "CAP_MSMF",
                "vfw": "CAP_VFW",
                "v4l2": "CAP_V4L2",
                "ffmpeg": "CAP_FFMPEG",
                "avfoundation": "CAP_AVFOUNDATION",
                "qt": "CAP_QT",
                "any": "CAP_ANY",
            }
            name_map = {}
            for k, attr_name in raw_map.items():
                val = getattr(cv2, attr_name, None)
                if val is not None:
                    name_map[k] = val
            for item in user_pref:
                it = str(item).lower()
                if it in name_map:
                    backends.append(name_map[it])
                else:
                    # if it's numeric, try to append as-is
                    try:
                        backends.append(int(item))
                    except Exception:
                        pass
        if not backends:
            if "windows" in sys_plat:
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
            elif "linux" in sys_plat:
                backends = [cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_ANY]
            elif "darwin" in sys_plat:
                backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_QT]
            else:
                backends = [cv2.CAP_ANY]

        tried = []
        opened = False
        # attempt numeric index first, then string
        candidates = []
        try:
            candidates.append(int(self.source))
        except Exception:
            candidates.append(self.source)
        for src in candidates:
            for b in backends:
                try:
                    logger.debug(
                        "Trying VideoCapture src=%s backend=%s", src, b
                    )
                    cap = (
                        cv2.VideoCapture(src, b)
                        if isinstance(b, int)
                        else cv2.VideoCapture(str(src))
                    )
                    # set small timeout/readiness test
                    if cap is not None and cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        # try grab frame
                        ret, _ = cap.read()
                        if ret:
                            logger.info(
                                "Opened camera source %s with backend %s",
                                src,
                                b,
                            )
                            self._cap = cap
                            opened = True
                            break
                        else:
                            logger.debug(
                                "Opened capture but read() failed for src=%s backend=%s",
                                src,
                                b,
                            )
                            try:
                                cap.release()
                            except Exception:
                                pass
                    else:
                        if cap:
                            try:
                                cap.release()
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug(
                        "Backend attempt failed for src=%s backend=%s: %s",
                        src,
                        b,
                        e,
                    )
                tried.append((src, b))
            if opened:
                break
        if not opened:
            logger.warning(
                "Could not open camera %s with any backend; tried: %s; falling back to simulation",
                self.source,
                tried,
            )
            # helpful diagnostics
            try:
                from pathlib import Path

                cwd = Path.cwd()
                files = [p.name for p in cwd.iterdir() if p.is_file()][:50]
                logger.debug(
                    "Current working directory files (first 50): %s", files
                )
                # list any *.pt files as possible models
                pt_files = [p.name for p in cwd.glob("*.pt")]
                logger.info("Detected .pt model files in cwd: %s", pt_files)
            except Exception:
                pass
            self.simulate = True
            self._cap = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        t0 = time.time()
        sim_pos = 0.0
        sim_dir = 1
        while self._running:
            if self.simulate and not self.sim_video:
                # generate a synthetic frame (neutral background). If inference injects a fake face,
                # do not draw an extra circle here to avoid duplicate visuals.
                frame = 30 * (
                    np.ones((self.height, self.width, 3), dtype="uint8")
                )
                # moving simulated face position for non-injected visual only
                sim_pos += 0.05 * sim_dir
                if sim_pos > 1.0:
                    sim_pos = 1.0
                    sim_dir = -1
                if sim_pos < 0.0:
                    sim_pos = 0.0
                    sim_dir = 1
                if not self.inject_fake_face:
                    cx = int(self.width * (0.3 + 0.4 * sim_pos))
                    cy = int(self.height * 0.45)
                    cv2.circle(
                        frame,
                        (cx, cy),
                        int(min(self.width, self.height) * 0.08),
                        (180, 180, 200),
                        -1,
                    )
                ret = True
            else:
                ret, frame = (
                    self._cap.read()
                    if self._cap is not None
                    else (False, None)
                )
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self._queue.put_nowait((time.time(), frame))
            except Exception:
                # drop frame if queue full
                pass

    def read(self, timeout: float = 0.5) -> Optional[Tuple[float, any]]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


class CameraManager:
    """Manage multiple CameraStream instances."""

    def __init__(self):
        self.streams: List[CameraStream] = []

    def add(self, stream: CameraStream) -> None:
        self.streams.append(stream)
        stream.start()

    def read_all(self) -> List[Tuple[int, float, any]]:
        out = []
        for idx, s in enumerate(self.streams):
            item = s.read(timeout=0.01)
            if item:
                out.append((idx, item[0], item[1]))
        return out

    def stop_all(self) -> None:
        for s in self.streams:
            s.stop()
