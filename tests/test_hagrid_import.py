"""Tests for the bounded, landmark-only HaGRID importer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_io import load_dataset
from src.import_hagrid import extract_samples, write_landmark_dataset


def hand(x_offset: float) -> list[list[float]]:
    return [[x_offset + index / 100.0, index / 200.0, 0.0] for index in range(21)]


class HaGRIDImportTests(unittest.TestCase):
    def test_extracts_only_supported_landmarks_with_a_per_class_limit(self) -> None:
        payload = {
            "like-1": {"labels": ["like"], "hand_landmarks": [hand(0.1)], "meta": {"race": ["ignored"]}},
            "like-2": {"labels": ["like"], "hand_landmarks": [hand(0.2)]},
            "peace": {"labels": ["peace"], "hand_landmarks": [hand(0.3)]},
            "heart": {"united_label": ["hand_heart"], "hand_landmarks": [hand(0.2), hand(0.7)]},
        }
        with tempfile.TemporaryDirectory() as directory:
            annotation_path = Path(directory) / "annotations.json"
            annotation_path.write_text(json.dumps(payload), encoding="utf-8")
            samples = extract_samples(annotation_path, max_per_class=1, seed=4)
            output_path = Path(directory) / "hagrid_landmarks.csv"
            counts = write_landmark_dataset(output_path, samples)
            features, labels = load_dataset(output_path)

        self.assertEqual(counts["thumbs_up"], 1)
        self.assertEqual(counts["peace"], 1)
        self.assertEqual(counts["heart"], 1)
        self.assertEqual(features.shape, (3, 126))
        self.assertEqual(sorted(labels.tolist()), ["heart", "peace", "thumbs_up"])

    def test_rejects_zero_sample_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation_path = Path(directory) / "annotations.json"
            annotation_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                extract_samples(annotation_path, max_per_class=0, seed=4)
