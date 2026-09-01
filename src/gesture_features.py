"""Convert MediaPipe hand landmarks into portable numeric feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
MAX_HANDS = 2
FEATURE_SIZE = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK * MAX_HANDS

TRAINING_LABELS = (
    "wave",
    "thumbs_up",
    "peace",
    "stop",
    "heart",
    "middle_finger",
    "ok",
    "mohan",
)

POSE_HINTS = {
    "wave": "Open hand, fingers apart",
    "thumbs_up": "Thumb up, other fingers folded",
    "peace": "Index and middle fingers raised",
    "stop": "Open palm facing the camera",
    "heart": "Make one heart with both hands",
    "middle_finger": "Only the middle finger raised",
    "ok": "Touch thumb and index; other fingers raised",
    "mohan": "Two hands: make an M with two peace signs",
}


@dataclass(frozen=True)
class HandSample:
    """One hand returned by MediaPipe, kept independent from MediaPipe types."""

    points: np.ndarray
    label: str = "Unknown"

    @classmethod
    def from_mediapipe(cls, landmarks: Iterable[object], label: str) -> "HandSample":
        points = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
        )
        if points.shape != (LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK):
            raise ValueError("Each hand must contain exactly 21 x/y/z landmarks.")
        return cls(points=points, label=label)


def _hand_size(points: np.ndarray) -> float:
    distances = np.linalg.norm(points[:, :2] - points[0, :2], axis=1)
    return float(max(np.max(distances), 1e-6))


def _single_hand_features(hand: HandSample) -> np.ndarray:
    points = hand.points.copy()
    points -= points[0]
    points /= _hand_size(hand.points)

    # A left-hand gesture is mirrored so either hand can perform one-hand gestures.
    if hand.label.lower() == "left":
        points[:, 0] *= -1
    return points.reshape(-1)


def landmarks_to_features(hands: Sequence[HandSample]) -> np.ndarray:
    """Return a fixed 126-value vector for zero, one, or two detected hands.

    For one hand, its wrist is the origin and its size normalizes the vector.
    For two hands, both sets of landmarks preserve their relative position,
    which is necessary for a two-hand heart gesture.
    """
    if len(hands) > MAX_HANDS:
        raise ValueError("At most two hands are supported.")
    if not hands:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

    if len(hands) == 1:
        features = _single_hand_features(hands[0])
        return np.pad(features, (0, FEATURE_SIZE - features.size)).astype(np.float32)

    ordered = sorted(hands, key=lambda hand: float(hand.points[0, 0]))
    wrists = np.asarray([hand.points[0] for hand in ordered], dtype=np.float32)
    center = wrists.mean(axis=0)
    wrist_distance = float(np.linalg.norm(wrists[0, :2] - wrists[1, :2]))
    scale = max(wrist_distance, _hand_size(ordered[0].points), _hand_size(ordered[1].points))
    features = np.concatenate([(hand.points - center).reshape(-1) / scale for hand in ordered])
    return features.astype(np.float32)


def required_hands(label: str) -> int:
    """Return the required number of visible hands for a training label."""
    return 2 if label in {"heart", "mohan"} else 1
