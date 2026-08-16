"""Visualization utilities using OpenCV to draw detections, tracks, and debug info."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .tracker import Track


def _rounded_box(frame, x1, y1, x2, y2, color, thickness=2, radius=8):
    # draw anti-aliased rectangle with optional rounded-feel corners by drawing a rect and small corner circles
    cv2.rectangle(
        frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA
    )
    # corner accents for a polished look
    cv2.circle(frame, (x1, y1), radius // 3, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x2, y1), radius // 3, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x1, y2), radius // 3, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x2, y2), radius // 3, color, -1, lineType=cv2.LINE_AA)


def draw_center_ui(
        frame: np.ndarray, serial_center: tuple | None = None, color=(0, 200, 200)
) -> None:
    # draw static center crosshair and moving dot if serial_center provided
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # crosshair
    cv2.line(
        frame,
        (cx - 20, cy),
        (cx + 20, cy),
        (180, 180, 180),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (cx, cy - 20),
        (cx, cy + 20),
        (180, 180, 180),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(frame, (cx, cy), 4, (200, 200, 200), 1, lineType=cv2.LINE_AA)
    # moving dot
    if serial_center:
        try:
            sx = int(max(0, min(1, serial_center[0])) * w)
            sy = int(max(0, min(1, serial_center[1])) * h)
            cv2.circle(frame, (sx, sy), 6, color, -1, lineType=cv2.LINE_AA)
        except Exception:
            pass
    else:
        # small static indicator at center
        cv2.circle(
            frame, (cx, cy), 3, (120, 120, 120), -1, lineType=cv2.LINE_AA
        )


def draw_detection(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        confidence: float = 0.0,
        color=(0, 255, 0),
        font_scale=0.5,
        thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    # shadow for depth
    overlay = frame.copy()
    shadow_color = (15, 15, 15)
    cv2.rectangle(
        overlay, (x1 + 3, y1 + 3), (x2 + 3, y2 + 3), shadow_color, -1
    )
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    # polished border
    _rounded_box(frame, x1, y1, x2, y2, color, thickness=thickness)
    # semi-transparent label background with rounded feel
    text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad_x = 8
    pad_y = 6
    # Keep detection text above the box where possible. Track labels are
    # placed below the box, preventing the two tag panels from colliding.
    label_h = h + pad_y * 2
    rx1, ry1 = x1, max(0, y1 - label_h)
    rx2, ry2 = x1 + w + pad_x * 2, y1
    overlay = frame.copy()
    # darker translucent panel
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # thin colored accent bar
    cv2.rectangle(
        frame, (rx1, ry2 - 6), (rx1 + int((rx2 - rx1) * 0.2), ry2), color, -1
    )
    cv2.putText(
        frame,
        text,
        (rx1 + pad_x, ry2 - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def draw_track(
        frame: np.ndarray, track: Track, color=(255, 0, 0), font_scale=0.5
) -> None:
    x1, y1, x2, y2 = map(int, track.bbox)
    # bolder border for tracked object
    _rounded_box(frame, x1, y1, x2, y2, color, thickness=3)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(frame, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)
    label = f"ID:{track.id} | {track.confidence:.2f}"
    # small flat panel for label near top-left of bbox for consistency
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    overlay = frame.copy()
    panel_x1 = x1
    panel_y1 = min(frame.shape[0] - 1, y2 + 4)
    panel_x2 = x1 + w + 12
    panel_y2 = min(frame.shape[0] - 1, panel_y1 + int(h + 12))
    if panel_y2 <= panel_y1:
        panel_y1 = max(0, y1 - int(h + 12))
        panel_y2 = y1
    cv2.rectangle(
        overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), color, -1
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(
        frame,
        label,
        (panel_x1 + 6, panel_y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # draw refined trajectory (smoother, fading)
    pts = [tuple(map(int, p)) for (_, p) in track.history[-60:]]
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts))
        thickness = max(1, int(5 * alpha))
        col = (
            int(color[0] * alpha + 30 * (1 - alpha)),
            int(color[1] * alpha + 30 * (1 - alpha)),
            int(color[2] * alpha + 30 * (1 - alpha)),
        )
        cv2.line(
            frame, pts[i - 1], pts[i], col, thickness, lineType=cv2.LINE_AA
        )


def draw_top_left_panel(
        frame: np.ndarray,
        tracked: dict | None = None,
        looking_for: list | None = None,
) -> None:
    # Draw a compact top-left panel showing the tracked object information and thumbnail
    h, w = frame.shape[:2]
    panel_w = min(300, max(230, int(w * 0.22)))
    panel_h = 125
    margin = 10
    x1, y1 = margin, margin
    x2, y2 = x1 + panel_w, y1 + panel_h
    overlay = frame.copy()
    # dark translucent background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 20, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    # header
    cv2.putText(
        frame,
        "TRACK",
        (x1 + 8, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # show what we are looking for (smaller)
    lf = ", ".join(looking_for) if looking_for else "any"
    max_chars = max(12, int((panel_w - 16) / 8))
    if len(lf) > max_chars:
        lf = lf[: max_chars - 3].rstrip() + "..."
    cv2.putText(
        frame,
        f"Looking: {lf}",
        (x1 + 8, y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    if tracked:
        label = tracked.get("class_name", "unknown")
        conf = tracked.get("confidence", 0.0)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1 + 8, y1 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # draw thumbnail area inside panel
        bbox = tracked.get("bbox")
        if bbox:
            bx1, by1, bx2, by2 = map(int, bbox)
            try:
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(w - 1, bx2)
                by2 = min(h - 1, by2)
                thumb = frame[by1:by2, bx1:bx2].copy()
                if thumb.size != 0 and (bx2 - bx1) > 4 and (by2 - by1) > 4:
                    th = panel_h - 64
                    tw = panel_w - 16
                    thumb = cv2.resize(
                        thumb, (tw, th), interpolation=cv2.INTER_AREA
                    )
                    ty = y1 + 58
                    tx = x1 + 8
                    frame[ty: ty + th, tx: tx + tw] = thumb
                    cv2.rectangle(
                        frame,
                        (tx - 1, ty - 1),
                        (tx + tw + 1, ty + th + 1),
                        (200, 200, 200),
                        1,
                    )
            except Exception:
                pass
    else:
        cv2.putText(
            frame,
            "No active target",
            (x1 + 8, y1 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def draw_tracking_queue(
        frame: np.ndarray, target_ids: list[int], current_id: int | None = None
) -> None:
    """Draw the optional operator submenu for the target rotation queue."""
    if not target_ids:
        lines = ["QUEUE", "empty"]
    else:
        lines = ["QUEUE"] + [
            f"{'>' if target_id == current_id else ' '} P{target_id}"
            for target_id in target_ids
        ]
    h, w = frame.shape[:2]
    x1, y1 = 10, 10
    panel_w = max(150, min(240, int(w * 0.20)))
    x1 = max(10, w - panel_w - 10)
    panel_h = 24 + 18 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (18, 20, 22), -1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for index, line in enumerate(lines):
        color = (
            (220, 220, 220)
            if index == 0
            else (180, 255, 180) if line.startswith(">") else (190, 190, 190)
        )
        cv2.putText(
            frame,
            line,
            (x1 + 8, y1 + 20 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def show_info(
        frame: np.ndarray,
        fps: float,
        inference_time_ms: float,
        device: str = "cpu",
        detections: int = 0,
        tracks: int = 0,
        font_scale_override: float | None = None,
) -> None:
    # Bottom-left translucent panel for system info (compact, more numeric stats)
    h, w = frame.shape[:2]
    panel_w = 320
    panel_h = 100
    x1, y1 = 8, h - panel_h - 8
    x2, y2 = x1 + panel_w, y1 + panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    info_color = (220, 220, 220)
    line_h = 18
    small_scale = 0.45 if font_scale_override is None else font_scale_override
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (x1 + 8, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        info_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Infer: {inference_time_ms:.1f}ms",
        (x1 + 8, y1 + 18 + line_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        info_color,
        1,
        cv2.LINE_AA,
    )
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        cv2.putText(
            frame,
            f"CPU: {cpu:.0f}%",
            (x1 + 8, y1 + 18 + 2 * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            small_scale,
            info_color,
            1,
            cv2.LINE_AA,
        )
    except Exception:
        cv2.putText(
            frame,
            f"Device: {device}",
            (x1 + 8, y1 + 18 + 2 * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            small_scale,
            info_color,
            1,
            cv2.LINE_AA,
        )
    # extra stats
    cv2.putText(
        frame,
        f"Det: {detections}",
        (x1 + 160, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        info_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Tracks: {tracks}",
        (x1 + 160, y1 + 18 + line_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        info_color,
        1,
        cv2.LINE_AA,
    )
    # GPU memory and utilization (try torch and pynvml)
    try:
        # noinspection PyPackageRequirements
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 * 1024)
            cv2.putText(
                frame,
                f"GPU mem(MB): {mem:.0f}",
                (x1 + 8, y1 + 18 + 3 * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_scale,
                info_color,
                1,
                cv2.LINE_AA,
            )
    except Exception:
        pass
    try:
        # try pynvml for GPU utilization
        # noinspection PyPackageRequirements
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gput = util.gpu
        memu = util.memory
        cv2.putText(
            frame,
            f"GPU util:{gput}% mem:{memu}%",
            (x1 + 160, y1 + 18 + 3 * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            small_scale,
            info_color,
            1,
            cv2.LINE_AA,
        )
    except Exception:
        pass
