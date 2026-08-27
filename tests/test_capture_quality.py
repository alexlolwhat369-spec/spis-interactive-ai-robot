"""Tests for duplicate protection during gesture capture."""

from __future__ import annotations

import unittest

import numpy as np

from src.capture_quality import CaptureGate


class CaptureQualityTests(unittest.TestCase):
    def test_first_capture_is_accepted(self) -> None:
        gate = CaptureGate()
        accepted, _ = gate.can_capture(np.zeros(126), now_ms=0)
        self.assertTrue(accepted)

    def test_cooldown_rejects_immediate_capture(self) -> None:
        gate = CaptureGate(cooldown_ms=350)
        gate.can_capture(np.zeros(126), now_ms=0)
        accepted, reason = gate.can_capture(np.ones(126), now_ms=100)
        self.assertFalse(accepted)
        self.assertEqual(reason, "wait a moment")

    def test_near_duplicate_is_rejected_after_cooldown(self) -> None:
        gate = CaptureGate(min_distance=0.18, cooldown_ms=0)
        gate.can_capture(np.zeros(126), now_ms=0)
        accepted, reason = gate.can_capture(np.full(126, 0.001), now_ms=400)
        self.assertFalse(accepted)
        self.assertEqual(reason, "move hands slightly")

    def test_varied_capture_is_accepted(self) -> None:
        gate = CaptureGate(min_distance=0.18, cooldown_ms=0)
        gate.can_capture(np.zeros(126), now_ms=0)
        accepted, _ = gate.can_capture(np.full(126, 0.1), now_ms=400)
        self.assertTrue(accepted)
