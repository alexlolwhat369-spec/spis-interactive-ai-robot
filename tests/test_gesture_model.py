"""Focused tests for the model saved and copied to Raspberry Pi."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gesture_model import GestureKNN


class GestureModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [4.0, 4.0], [4.1, 4.0], [4.0, 4.1]],
            dtype=np.float32,
        )
        self.labels = ["wave", "wave", "wave", "peace", "peace", "peace"]
        self.model = GestureKNN.fit(self.features, self.labels, k=3)

    def test_nearby_sample_is_classified(self) -> None:
        prediction = self.model.predict(np.asarray([0.04, 0.02], dtype=np.float32))
        self.assertEqual(prediction.label, "wave")
        self.assertGreater(prediction.confidence, 0.5)

    def test_distant_sample_is_rejected(self) -> None:
        prediction = self.model.predict(np.asarray([20.0, -20.0], dtype=np.float32))
        self.assertEqual(prediction.label, "unknown")

    def test_saved_model_keeps_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gesture_knn.npz"
            self.model.save(path)
            loaded = GestureKNN.load(path)
            self.assertEqual(loaded.predict(np.asarray([4.03, 4.02], dtype=np.float32)).label, "peace")

    def test_close_neighbour_outweighs_a_distant_majority(self) -> None:
        model = GestureKNN(
            features=np.asarray([[0.01, 0.0], [0.5, 0.0], [0.6, 0.0]], dtype=np.float32),
            labels=np.asarray(["close", "far", "far"]),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            labels_order=np.asarray(["close", "far"]),
            distance_limit=10.0,
            k=3,
        )

        prediction = model.predict(np.asarray([0.0, 0.0], dtype=np.float32))

        self.assertEqual(prediction.label, "close")
        self.assertGreater(prediction.confidence, 0.9)
