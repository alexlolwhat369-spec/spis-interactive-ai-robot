"""Current MediaPipe Tasks hand tracking used by capture and live demo."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import cv2
import mediapipe as mp

try:
    from .gesture_features import HandSample
except ImportError:  # Supports direct execution: python src/collect_samples.py
    from gesture_features import HandSample

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandTracker:
    """Thin wrapper around MediaPipe's supported Hand Landmarker Task API."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found: {model_path}. Run: python src/setup_assets.py"
            )
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._started_at = monotonic()

    def detect(self, frame_bgr: object) -> list[HandSample]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((monotonic() - self._started_at) * 1000)
        result = self._detector.detect_for_video(image, timestamp_ms)
        hands: list[HandSample] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0] if handedness else None
            label = (category.category_name or category.display_name) if category else "Unknown"
            hands.append(HandSample.from_mediapipe(landmarks, label or "Unknown"))
        return hands

    def close(self) -> None:
        self._detector.close()


def draw_hands(frame_bgr: object, hands: list[HandSample]) -> None:
    """Draw landmarks using OpenCV so the app has no legacy MediaPipe imports."""
    height, width = frame_bgr.shape[:2]
    for hand in hands:
        points = [(int(point[0] * width), int(point[1] * height)) for point in hand.points]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (255, 220, 0), 2)
        for point in points:
            cv2.circle(frame_bgr, point, 3, (255, 255, 255), -1)

