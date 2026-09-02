"""A small K-nearest-neighbour classifier that can run on a Raspberry Pi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Prediction:
    label: str
    confidence: float
    nearest_distance: float


@dataclass
class GestureKNN:
    features: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    labels_order: np.ndarray
    distance_limit: float
    k: int = 5

    @classmethod
    def fit(cls, features: np.ndarray, labels: Sequence[str], k: int = 5) -> "GestureKNN":
        features = np.asarray(features, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=str)
        if features.ndim != 2 or features.shape[0] != labels_array.size:
            raise ValueError("Features and labels must contain the same number of samples.")
        if features.shape[0] < 2:
            raise ValueError("At least two samples are needed to train the model.")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (features - mean) / scale
        labels_order = np.unique(labels_array)

        # Leave-one-out nearest-neighbour distances give a practical rejection limit.
        distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        distance_limit = float(np.percentile(nearest, 95) * 1.75)
        return cls(features, labels_array, mean, scale, labels_order, distance_limit, min(k, len(labels_array)))

    def predict(self, feature: np.ndarray) -> Prediction:
        feature = np.asarray(feature, dtype=np.float32)
        normalized_train = (self.features - self.mean) / self.scale
        normalized_feature = (feature - self.mean) / self.scale
        distances = np.linalg.norm(normalized_train - normalized_feature, axis=1)
        neighbour_indices = np.argsort(distances)[: self.k]
        neighbour_labels = self.labels[neighbour_indices]
        neighbour_distances = distances[neighbour_indices]
        # Nearby examples should count more than borderline neighbours. This
        # keeps KNN stable while avoiding a weak distant majority overruling a
        # very close pose, which is especially useful for similar open hands.
        weights = 1.0 / np.maximum(neighbour_distances, 1e-6)
        votes = {
            label: float(np.sum(weights[neighbour_labels == label]))
            for label in self.labels_order
        }
        label = max(votes, key=votes.get)
        nearest_distance = float(neighbour_distances[0])
        confidence = float(votes[label] / np.sum(weights))
        if nearest_distance > self.distance_limit:
            return Prediction("unknown", 0.0, nearest_distance)
        return Prediction(label, confidence, nearest_distance)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            mean=self.mean,
            scale=self.scale,
            labels_order=self.labels_order,
            distance_limit=np.asarray([self.distance_limit]),
            k=np.asarray([self.k]),
        )

    @classmethod
    def load(cls, path: Path) -> "GestureKNN":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                features=data["features"],
                labels=data["labels"].astype(str),
                mean=data["mean"],
                scale=data["scale"],
                labels_order=data["labels_order"].astype(str),
                distance_limit=float(data["distance_limit"][0]),
                k=int(data["k"][0]),
            )
