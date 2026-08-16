"""Optional, lazy-loaded face emotion and hand gesture features."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("realtime_cv.features")


def face_appearance(
    frame: np.ndarray,
    bbox: Any,
) -> list[float] | None:
    x1, y1, x2, y2 = map(int, bbox)

    height, width = frame.shape[:2]

    crop = frame[
        max(0, y1) : min(height, y2),
        max(0, x1) : min(width, x2),
    ]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256],
    ).flatten()

    norm = np.linalg.norm(hist)

    if not norm:
        return None

    return (hist / norm).astype(float).tolist()


class OptionalFeatures:
    def __init__(self, config: Any) -> None:
        self.emotion: Any = None
        self.hands: Any = None

        features = getattr(config, "features", config)

        self.gesture_map: dict[str, Any] = (
            getattr(features, "hand_gesture_map", {}) or {}
        )

        if getattr(features, "emotion_tracking", False):
            self._initialize_emotion(features)

        if getattr(features, "hand_tracking", False):
            self._initialize_hands()

    def _initialize_emotion(self, features: Any) -> None:
        backend = getattr(
            features,
            "emotion_backend",
            "fer",
        ).lower()

        if backend != "fer":
            raise ValueError("Unsupported emotion_backend; use 'fer'")

        try:
            from fer.fer import FER

            if not callable(FER):
                raise RuntimeError(
                    "fer.FER is unavailable in the installed fer package"
                )

            self.emotion = FER(mtcnn=False)

        except Exception as exc:
            logger.warning(
                "Emotion tracking disabled: backend unavailable (%s)",
                exc,
            )

    def _initialize_hands(self) -> None:
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    "mediapipe.solutions is unavailable; "
                    "install mediapipe==0.10.21"
                )

            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        except Exception as exc:
            logger.warning(
                "Hand tracking disabled: backend unavailable (%s)",
                exc,
            )

    def annotate_emotions(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> None:
        if self.emotion is None:
            return

        for detection in detections:
            class_name_value = detection.get(
                "class_name",
                detection.get("label", ""),
            )

            if not isinstance(class_name_value, str):
                class_name_value = ""

            class_name = class_name_value.lower()

            if class_name != "face":
                continue

            try:
                x1, y1, x2, y2 = map(
                    int,
                    detection["bbox"],
                )

                height, width = frame.shape[:2]

                crop = frame[
                    max(0, y1) : min(height, y2),
                    max(0, x1) : min(width, x2),
                ]

                if crop.size == 0:
                    continue

                result: Any = self.emotion.detect_emotions(crop)

                if not result:
                    continue

                first_result = result[0]

                if not isinstance(first_result, dict):
                    continue

                scores = first_result.get("emotions", {})

                if not isinstance(scores, dict) or not scores:
                    continue

                emotion = max(
                    scores,
                    key=lambda name: float(scores[name]),
                )

                detection["emotion"] = str(emotion)
                detection["emotion_confidence"] = float(scores[emotion])

            except Exception:
                logger.debug(
                    "Emotion inference failed",
                    exc_info=True,
                )

    def detect_hands(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        if self.hands is None:
            return []

        try:
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # MediaPipe's Python type definitions do not correctly
            # describe the runtime HandsResult object.
            result: Any = self.hands.process(rgb)

            multi_hand_landmarks = result.multi_hand_landmarks or []

            multi_handedness = result.multi_handedness or []

            output: list[dict[str, Any]] = []

            for landmarks, handedness in zip(
                multi_hand_landmarks,
                multi_handedness,
            ):
                if not landmarks.landmark:
                    continue

                if not handedness.classification:
                    continue

                points = [
                    (
                        int(
                            np.clip(
                                point.x * frame.shape[1],
                                0,
                                frame.shape[1] - 1,
                            )
                        ),
                        int(
                            np.clip(
                                point.y * frame.shape[0],
                                0,
                                frame.shape[0] - 1,
                            )
                        ),
                    )
                    for point in landmarks.landmark
                ]

                if len(points) < 21:
                    continue

                classification = handedness.classification[0]

                hand_label = str(classification.label)

                # Index, middle, ring and little fingers.
                fingers = [
                    points[index][1] < points[index - 3][1]
                    for index in (8, 12, 16, 20)
                ]

                # MediaPipe handedness follows the mirrored/selfie
                # convention.
                thumb_extended = (
                    points[4][0] < points[3][0]
                    if hand_label == "Right"
                    else points[4][0] > points[3][0]
                )

                fingers.insert(
                    0,
                    thumb_extended,
                )

                finger_count = sum(fingers)

                gesture = self.gesture_map.get(
                    str(finger_count),
                    str(finger_count),
                )

                output.append(
                    {
                        "points": points,
                        "gesture": gesture,
                        "finger_count": finger_count,
                        "fingers": fingers,
                        "handedness": hand_label,
                    }
                )

            return output

        except Exception:
            logger.debug(
                "Hand inference failed",
                exc_info=True,
            )
            return []
