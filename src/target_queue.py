"""Stateful multi-target selection and hit tracking.

The tracker owns short-lived motion tracks; this module owns the longer-lived
shooting state.  A target is queued once it is observed, remains selectable
while its tracker ID exists, and is either cycled or permanently excluded after a shot.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .tracker import Track


class TargetQueue:
    def __init__(self, cycle_remember: bool = True) -> None:
        self.cycle_remember = cycle_remember
        self._queue: deque[int] = deque()
        self._queued: set[int] = set()
        self.hit_ids: set[int] = set()
        self.current_id: int | None = None

    def sync(self, tracks: Iterable[Track]) -> None:
        """Add visible tracks and remove IDs no longer owned by the tracker."""
        # A missed/lost track is treated as out of range for target selection;
        # it may be re-added automatically if the detector sees it again.
        tracks = list(tracks)
        tracks_by_id = {
            track.id: track
            for track in tracks
            if track.confirmed and track.lost == 0
        }
        live_ids = set(tracks_by_id)

        excluded = self.hit_ids if not self.cycle_remember else set()
        self._queue = deque(
            tid
            for tid in self._queue
            if tid in live_ids and tid not in excluded
        )
        self._queued = set(self._queue)
        if self.current_id is not None and self.current_id not in live_ids:
            self.current_id = None

        for track in tracks:
            if (
                    track.confirmed
                    and track.lost == 0
                    and (self.cycle_remember or track.id not in self.hit_ids)
                    and track.id not in self._queued
            ):
                self._queue.append(track.id)
                self._queued.add(track.id)

    def select_next(self) -> int | None:
        """Lock the next unhit target, preserving the lock while it exists."""
        if self.current_id is not None and self.current_id in self._queued:
            return self.current_id
        self.current_id = self._queue[0] if self._queue else None
        return self.current_id

    def mark_shot(self, target_id: int | None = None) -> int | None:
        """Mark a target hit and advance to the next target exactly once."""
        target_id = target_id if target_id is not None else self.current_id
        if target_id is None or target_id in self.hit_ids:
            return self.select_next()
        self._queue = deque(tid for tid in self._queue if tid != target_id)
        self._queued.discard(target_id)
        if self.cycle_remember:
            self._queue.append(target_id)
            self._queued.add(target_id)
        else:
            self.hit_ids.add(target_id)
        if self.current_id == target_id:
            self.current_id = None
        return self.select_next()

    def target_ids(self) -> list[int]:
        return list(self._queue)

    def is_hit(self, target_id: int) -> bool:
        return target_id in self.hit_ids
