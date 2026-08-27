from __future__ import annotations

import unittest

from src.gesture_gate import GestureGate
from src.gesture_model import Prediction


class GestureGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = GestureGate(distance_limit=10.0, activation_frames=3, release_frames=2)

    def test_requires_a_stable_close_gesture(self) -> None:
        prediction = Prediction("peace", 1.0, 3.0)
        self.assertEqual(self.gate.update(prediction, hand_count=1), "none")
        self.assertEqual(self.gate.update(prediction, hand_count=1), "none")
        self.assertEqual(self.gate.update(prediction, hand_count=1), "peace")

    def test_rejects_a_distant_or_low_vote_hand(self) -> None:
        self.assertEqual(self.gate.update(Prediction("peace", 1.0, 7.0), hand_count=1), "none")
        self.assertEqual(self.gate.update(Prediction("peace", 0.6, 2.0), hand_count=1), "none")

    def test_heart_requires_two_hands(self) -> None:
        heart = Prediction("heart", 1.0, 2.0)
        for _ in range(4):
            self.assertEqual(self.gate.update(heart, hand_count=1), "none")
        self.assertEqual(self.gate.update(heart, hand_count=2), "none")
        self.assertEqual(self.gate.update(heart, hand_count=2), "none")
        self.assertEqual(self.gate.update(heart, hand_count=2), "heart")

    def test_releases_after_hand_is_gone(self) -> None:
        peace = Prediction("peace", 1.0, 2.0)
        for _ in range(3):
            self.gate.update(peace, hand_count=1)
        self.assertEqual(self.gate.update(Prediction("none", 1.0, 0.0), hand_count=0), "peace")
        self.assertEqual(self.gate.update(Prediction("none", 1.0, 0.0), hand_count=0), "none")

    def test_reset_requires_a_gesture_to_be_stable_again(self) -> None:
        peace = Prediction("peace", 1.0, 2.0)
        for _ in range(3):
            self.gate.update(peace, hand_count=1)
        self.assertEqual(self.gate.update(peace, hand_count=1), "peace")

        self.gate.reset()
        self.assertEqual(self.gate.update(peace, hand_count=1), "none")
        self.assertEqual(self.gate.update(peace, hand_count=1), "none")
        self.assertEqual(self.gate.update(peace, hand_count=1), "peace")

    def test_suspend_requires_hands_to_leave_before_a_new_gesture(self) -> None:
        peace = Prediction("peace", 1.0, 2.0)
        self.gate.suspend()

        for _ in range(5):
            self.assertEqual(self.gate.update(peace, hand_count=1), "none")
        self.assertEqual(self.gate.update(Prediction("none", 1.0, 0.0), hand_count=0), "none")
        self.assertEqual(self.gate.update(Prediction("none", 1.0, 0.0), hand_count=0), "none")
        self.assertEqual(self.gate.update(peace, hand_count=1), "none")
        self.assertEqual(self.gate.update(peace, hand_count=1), "none")
        self.assertEqual(self.gate.update(peace, hand_count=1), "peace")
