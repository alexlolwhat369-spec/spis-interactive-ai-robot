"""Small, deterministic quality gate for landmark sample collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CaptureGate:
    """Reject near-duplicate captures so a dataset contains useful variation."""

    min_distance: float = 0.18
    cooldown_ms: int = 350
    _last_features: np.ndarray | None = None
    _last_capture_ms: float | None = None

    def can_capture(self, features: np.ndarray, now_ms: float) -> tuple[bool, str]:
        values = np.asarray(features, dtype=np.float32)
        if self._last_capture_ms is not None:
            elapsed = now_ms - self._last_capture_ms
            if elapsed < self.cooldown_ms:
                return False, "wait a moment"
        if self._last_features is not None:
            distance = float(np.linalg.norm(values - self._last_features))
            if distance < self.min_distance:
                return False, "move hands slightly"
        self._last_features = values.copy()
        self._last_capture_ms = now_ms
        return True, "sample accepted"
