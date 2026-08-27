"""Conservative live gate so ordinary hand movement is not treated as a gesture."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .gesture_features import required_hands
    from .gesture_model import Prediction
except ImportError:  # Supports direct execution: python src/live_demo.py
    from gesture_features import required_hands
    from gesture_model import Prediction


@dataclass
class GestureGate:
    distance_limit: float
    min_confidence: float = 0.8
    distance_ratio: float = 0.6
    activation_frames: int = 8
    release_frames: int = 4
    _candidate: str = "none"
    _candidate_frames: int = 0
    _active: str = "none"
    _missing_frames: int = 0
    _armed: bool = True

    def reset(self) -> None:
        """Forget a gesture while another interaction has priority."""
        self._candidate = "none"
        self._candidate_frames = 0
        self._active = "none"
        self._missing_frames = 0

    def suspend(self) -> None:
        """Require hands to leave the camera before accepting another gesture."""
        self.reset()
        self._armed = False

    def update(self, prediction: Prediction, hand_count: int) -> str:
        """Return a gesture only after it is close, valid, and stable."""
        if not self._armed:
            if hand_count == 0:
                self._missing_frames += 1
                if self._missing_frames >= self.release_frames:
                    self._armed = True
                    self.reset()
            else:
                self._missing_frames = 0
            return "none"

        candidate = prediction.label
        accepted = (
            candidate not in {"none", "unknown"}
            and prediction.confidence >= self.min_confidence
            and prediction.nearest_distance <= self.distance_limit * self.distance_ratio
            and hand_count >= required_hands(candidate)
        )
        if not accepted:
            self._candidate = "none"
            self._candidate_frames = 0
            self._missing_frames += 1
            if self._missing_frames >= self.release_frames:
                self._active = "none"
            return self._active

        self._missing_frames = 0
        if candidate == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = candidate
            self._candidate_frames = 1
        if self._candidate_frames >= self.activation_frames:
            self._active = candidate
        return self._active
