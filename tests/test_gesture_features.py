"""Focused tests for portable landmark feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from src.gesture_features import FEATURE_SIZE, TRAINING_LABELS, HandSample, landmarks_to_features, required_hands


def make_hand(offset_x: float = 0.0) -> HandSample:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(offset_x, offset_x + 0.2, 21)
    points[:, 1] = np.linspace(0.0, 0.1, 21)
    return HandSample(points, "Right")


class GestureFeatureTests(unittest.TestCase):
    def test_no_hands_is_a_zero_vector(self) -> None:
        features = landmarks_to_features([])
        self.assertEqual(features.shape, (FEATURE_SIZE,))
        self.assertTrue(np.all(features == 0))

    def test_one_hand_has_fixed_feature_size(self) -> None:
        features = landmarks_to_features([make_hand()])
        self.assertEqual(features.shape, (FEATURE_SIZE,))
        self.assertFalse(np.all(features == 0))

    def test_two_hands_keep_relative_geometry(self) -> None:
        close = landmarks_to_features([make_hand(0.0), make_hand(0.25)])
        far = landmarks_to_features([make_hand(0.0), make_hand(0.7)])
        self.assertEqual(close.shape, (FEATURE_SIZE,))
        self.assertFalse(np.allclose(close, far))

    def test_heart_requires_two_hands(self) -> None:
        self.assertEqual(required_hands("heart"), 2)
        self.assertEqual(required_hands("wave"), 1)

    def test_new_training_labels_use_the_correct_number_of_hands(self) -> None:
        self.assertTrue({"middle_finger", "ok", "mohan"}.issubset(TRAINING_LABELS))
        for label in ("middle_finger", "ok"):
            self.assertEqual(required_hands(label), 1)
        self.assertEqual(required_hands("mohan"), 2)


if __name__ == "__main__":
    unittest.main()
