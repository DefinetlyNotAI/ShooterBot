"""Robust multi-object tracker with optional Kalman filter + Hungarian assignment.

Defaults to a lightweight IoU greedy tracker when scipy is not available.
Tracks provide persistent IDs, velocity (px/s), history for trajectories, and automatic pruning of lost tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, cast

from .coco import COCO_CLASSES
from .utils import bbox_center, iou

Detection = Dict[str, Any]


def _detection_bbox(detection: Detection) -> Tuple[float, float, float, float]:
    bbox = detection["bbox"]

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid detection bbox: {bbox!r}")

    return (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )


@dataclass
class Track:
    id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    confidence: float
    last_seen: float
    history: List[Tuple[float, Tuple[float, float]]] = field(
        default_factory=list
    )
    velocity: Tuple[float, float] = (0.0, 0.0)
    lost: int = 0
    hits: int = 1
    age: int = 0
    confirmed: bool = False
    # Kalman state (if used)
    kf_state: Optional[List[float]] = None
    kf_cov: Optional[List[List[float]]] = None
    appearance: Optional[List[float]] = None


class Tracker:
    def __init__(
            self,
            max_lost: int = 30,
            iou_threshold: float = 0.3,
            track_only: List[str | int] | None = None,
            use_kalman: bool = True,
            min_hits: int = 1,
            max_age: int = 30,
            class_priority: List[str] | None = None,
            remember_faces: bool = False,
            face_memory_threshold: float = 0.78,
            face_memory_max_age: int = 300,
    ):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}
        # track_only: list of class_name strings or class_ids to be tracked; None => track all
        self.track_only = track_only
        # try to use Hungarian if available
        self.use_kalman = use_kalman
        self.min_hits = max(1, int(min_hits))
        self.max_age = max(1, int(max_age))
        self.class_priority = class_priority or []
        self.remember_faces = bool(remember_faces)
        self.face_memory_threshold = float(face_memory_threshold)
        self.face_memory_max_age = int(face_memory_max_age)
        self._face_gallery: Dict[int, Tuple[List[float], float]] = {}
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            self._hungarian_available = True
        except Exception:
            self._hungarian_available = False
        # Kalman parameters
        self._dt = 1.0 / 30.0

    def _should_track(self, det: Detection) -> bool:
        if self.track_only is None:
            return True
        cls_name = det.get("class_name")
        cls_id = det.get("class_id")
        for t in self.track_only:
            if isinstance(t, str) and cls_name == t:
                return True
            if isinstance(t, int) and cls_id == t:
                return True
        return False

    @staticmethod
    def _class_name(track: Track) -> str:
        if 0 <= int(track.class_id) < len(COCO_CLASSES):
            return COCO_CLASSES[int(track.class_id)]
        return ""

    def _priority_score(self, class_name: str) -> int:
        if not self.class_priority:
            return 0
        lowered = [str(item).lower() for item in self.class_priority]
        try:
            index = lowered.index(str(class_name).lower())
        except ValueError:
            return 0
        return len(lowered) - index

    def rank_track(self, track: Track) -> tuple[int, float, int, float]:
        class_name = self._class_name(track)
        return (
            self._priority_score(class_name),
            float(track.confidence),
            int(track.hits),
            float(track.last_seen),
        )

    def select_primary_track(
            self, tracks: List[Track], preferred_id: int | None = None
    ) -> Optional[Track]:
        confirmed_tracks = [track for track in tracks if track.confirmed]
        if not confirmed_tracks:
            return None

        if preferred_id is not None:
            for track in confirmed_tracks:
                if track.id == preferred_id:
                    return track

        return max(confirmed_tracks, key=self.rank_track)

    def _spawn_track(self, detection: Detection, timestamp: float) -> Track:
        bbox = _detection_bbox(detection)
        center = bbox_center(bbox)

        tr = Track(
            id=self._next_id,
            bbox=bbox,
            class_id=int(detection.get("class_id", -1)),
            confidence=float(detection.get("confidence", 0.0)),
            last_seen=timestamp,
            history=[(timestamp, center)],
            confirmed=self.min_hits <= 1,
            appearance=detection.get("appearance"),
        )

        self._next_id += 1

        if self.use_kalman:
            self._init_kf(tr)

        return tr

    @staticmethod
    def _appearance_similarity(
            a: Optional[List[float]],
            b: Optional[List[float]],
    ) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else -1.0

    def _remember_track(self, track: Track, timestamp: float) -> None:
        if (
                self.remember_faces
                and track.appearance
                and self._class_name(track) == ""
        ):
            self._face_gallery[track.id] = (track.appearance, timestamp)

    def _remember_lost_face(self, track: Track, timestamp: float) -> None:
        """Capture identity as soon as a face is missed, not only after deletion."""
        if self.remember_faces and track.appearance:
            self._face_gallery[track.id] = (track.appearance, timestamp)

    def _revive_face(
            self,
            detection: Detection,
            timestamp: float,
    ) -> Optional[Track]:
        appearance = detection.get("appearance")

        if not self.remember_faces or not appearance:
            return None

        best_id: Optional[int] = None
        best_score = self.face_memory_threshold

        for tid, (feature, seen) in list(self._face_gallery.items()):
            if timestamp - seen > self.face_memory_max_age:
                del self._face_gallery[tid]
                continue

            score = self._appearance_similarity(feature, appearance)

            if score >= best_score:
                best_id = tid
                best_score = score

        if best_id is None:
            return None

        tr = self._spawn_track(detection, timestamp)
        tr.id = best_id
        tr.appearance = appearance

        del self._face_gallery[best_id]

        return tr

    # --- simple Kalman utilities (constant velocity) ---
    @staticmethod
    def _init_kf(track: Track):
        # state: [cx, cy, vx, vy]
        cx, cy = bbox_center(track.bbox)
        state = [cx, cy, 0.0, 0.0]
        cov = [[1e3 if i < 2 else 1e3 for _ in range(4)] for i in range(4)]
        track.kf_state = state
        track.kf_cov = cov

    def _predict_kf(self, track: Track, dt: float):
        if track.kf_state is None:
            self._init_kf(track)
            return track.kf_state
        x, y, vx, vy = track.kf_state
        # simple linear motion
        x += vx * dt
        y += vy * dt
        track.kf_state = [x, y, vx, vy]
        return track.kf_state

    def _update_kf(
            self,
            track: Track,
            meas: Tuple[float, float],
            dt: float,
    ) -> None:
        # naive update: set velocity from difference
        if track.kf_state is None:
            self._init_kf(track)

        state = track.kf_state

        if state is None:
            return

        px, py, vx, vy = state
        mx, my = meas

        new_vx = (mx - px) / max(1e-3, dt)
        new_vy = (my - py) / max(1e-3, dt)

        track.kf_state = [
            mx,
            my,
            0.7 * vx + 0.3 * new_vx,
            0.7 * vy + 0.3 * new_vy,
            ]

    def update(
            self,
            detections: List[Detection],
            timestamp: float,
    ) -> List[Track]:
        # Only consider detections that should be tracked for association, but keep all in input for visualization
        dets = [d for d in detections if self._should_track(d)]
        n_det = len(dets)
        tracks_list = list(self.tracks.items())
        n_tr = len(tracks_list)

        # If no tracks exist, create tracks for all detections
        if n_tr == 0:
            for d in dets:
                tr = self._spawn_track(d, timestamp)
                self.tracks[tr.id] = tr
            return [tr for tr in self.tracks.values() if tr.confirmed]

        # Predict step for Kalman
        if self.use_kalman:
            # estimate dt since last update for each track
            for tid, tr in tracks_list:
                last = tr.last_seen if tr.last_seen else timestamp
                dt = max(1e-3, timestamp - last)
                self._predict_kf(tr, dt)

        # Build cost matrix (IoU-based) between predicted track bboxes and detections
        import numpy as _np

        cost = _np.full((n_tr, n_det), 1e6, dtype=float)
        for i, (tid, tr) in enumerate(tracks_list):
            # predicted center
            # pred_cx, pred_cy = bbox_center(tr.bbox)
            for j, d in enumerate(dets):
                try:
                    # use IoU as similarity
                    cost[i, j] = 1.0 - iou(tr.bbox, _detection_bbox(d))
                except Exception:
                    cost[i, j] = 1.0

        matches: List[Tuple[int, int]] = []
        if self._hungarian_available and n_tr and n_det:
            from scipy.optimize import linear_sum_assignment

            assignment = cast(
                Tuple[_np.ndarray, _np.ndarray],
                linear_sum_assignment(cost),
            )

            row_ind, col_ind = assignment

            for r, c in zip(row_ind, col_ind):
                if (
                        r < n_tr
                        and c < n_det
                        and cost[r, c] <= (1.0 - self.iou_threshold)
                ):
                    matches.append((r, c))
        else:
            # greedy matching by lowest cost
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            flat: List[Tuple[float, int, int]] = []
            for i in range(n_tr):
                for j in range(n_det):
                    flat.append((cost[i, j], i, j))
            flat.sort(key=lambda x: x[0])
            for cst, i, j in flat:
                if i in used_rows or j in used_cols:
                    continue
                if cst <= (1.0 - self.iou_threshold):
                    matches.append((i, j))
                    used_rows.add(i)
                    used_cols.add(j)

        matched_tr_ids: set[int] = set()
        matched_det_idx: set[int] = set()

        # Update matched tracks
        for i, j in matches:
            tid, tr = tracks_list[i]
            d = dets[j]
            matched_tr_ids.add(tid)
            matched_det_idx.add(j)
            # update kf and track
            center = bbox_center(_detection_bbox(d))
            if self.use_kalman:
                dt = max(1e-3, timestamp - tr.last_seen)
                self._update_kf(tr, center, dt)
                # read velocity from kf_state
                if tr.kf_state:
                    vx, vy = tr.kf_state[2], tr.kf_state[3]
                    tr.velocity = (vx, vy)
            else:
                prev_center = bbox_center(tr.bbox)
                dt = max(1e-3, timestamp - tr.last_seen)
                vx = (center[0] - prev_center[0]) / dt
                vy = (center[1] - prev_center[1]) / dt
                tr.velocity = (vx, vy)
            tr.bbox = _detection_bbox(d)
            tr.confidence = float(d.get("confidence", tr.confidence))
            tr.class_id = int(d.get("class_id", tr.class_id))
            if d.get("appearance"):
                tr.appearance = d["appearance"]
            tr.last_seen = timestamp
            tr.history.append((timestamp, center))
            tr.lost = 0
            tr.age = 0
            tr.hits += 1
            tr.confirmed = tr.confirmed or tr.hits >= self.min_hits

        # mark unmatched tracks as lost
        for tid, tr in tracks_list:
            if tid in matched_tr_ids:
                continue
            self._remember_lost_face(tr, timestamp)
            tr.lost += 1
            tr.age += 1
            if tr.lost > self.max_lost or tr.age > self.max_age:
                self._remember_track(tr, timestamp)
                try:
                    del self.tracks[tid]
                except KeyError:
                    pass

        # create new tracks for unmatched detections
        for j, d in enumerate(dets):
            if j in matched_det_idx:
                continue
            tr = self._revive_face(d, timestamp) or self._spawn_track(
                d, timestamp
            )
            self.tracks[tr.id] = tr

        return [tr for tr in self.tracks.values() if tr.confirmed]

    def get_active_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.confirmed]
