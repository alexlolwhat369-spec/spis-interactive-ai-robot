"""Tests ensuring the training dataset contains landmark numbers, not images."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_io import append_sample, load_dataset
from src.gesture_features import FEATURE_SIZE


class DatasetTests(unittest.TestCase):
    def test_saved_dataset_contains_only_label_and_numeric_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "landmarks.csv"
            append_sample(path, "wave", np.zeros(FEATURE_SIZE, dtype=np.float32))
            text = path.read_text(encoding="utf-8")
            features, labels = load_dataset(path)
            self.assertEqual(labels.tolist(), ["wave"])
            self.assertEqual(features.shape, (1, FEATURE_SIZE))
            self.assertNotIn("image", text.lower())
            self.assertNotIn("frame", text.lower())

    def test_wrong_feature_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "landmarks.csv"
            with self.assertRaises(ValueError):
                append_sample(path, "wave", [0.0] * (FEATURE_SIZE - 1))
