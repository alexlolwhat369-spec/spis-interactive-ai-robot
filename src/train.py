"""Train, evaluate, and save the lightweight gesture KNN model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .data_io import load_dataset
    from .gesture_model import GestureKNN
except ImportError:  # Supports direct execution: python src/train.py
    from data_io import load_dataset
    from gesture_model import GestureKNN

ROOT = Path(__file__).resolve().parents[1]


def stratified_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        test_size = max(1, int(round(len(indices) * test_fraction)))
        if len(indices) - test_size < 2:
            raise ValueError(f"'{label}' needs at least three samples for a train/test split.")
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])
    return np.asarray(train_indices), np.asarray(test_indices)


def write_confusion_matrix(path: Path, labels: list[str], actual: np.ndarray, predicted: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in labels] for _ in labels]
    positions = {label: index for index, label in enumerate(labels)}
    for truth, guess in zip(actual, predicted):
        if guess not in positions:
            continue
        matrix[positions[truth]][positions[guess]] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        writer.writerows([[label, *row] for label, row in zip(labels, matrix)])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a portable hand-gesture KNN model.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "landmarks.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "gesture_knn.npz")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    features, labels = load_dataset(args.dataset)
    counts = Counter(labels.tolist())
    if len(counts) < 2:
        raise ValueError("Capture at least two gesture classes before training.")
    train_indices, test_indices = stratified_split(labels, args.test_fraction, args.seed)
    model = GestureKNN.fit(features[train_indices], labels[train_indices], args.k)
    predictions = [model.predict(feature).label for feature in features[test_indices]]
    actual = labels[test_indices]
    accuracy = float(np.mean(actual == np.asarray(predictions)))
    label_order = sorted(counts)
    matrix = write_confusion_matrix(ROOT / "reports" / "confusion_matrix.csv", label_order, actual, predictions)

    per_class = {}
    for label in label_order:
        positions = np.flatnonzero(actual == label)
        per_class[label] = float(np.mean(np.asarray(predictions)[positions] == label))
    report = {
        "accuracy": accuracy,
        "samples_per_class": dict(sorted(counts.items())),
        "test_samples": int(len(test_indices)),
        "per_class_recall": per_class,
        "labels": label_order,
        "confusion_matrix": matrix,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Retrain using every captured sample for the model copied to Raspberry Pi.
    GestureKNN.fit(features, labels, args.k).save(args.model)
    print(json.dumps(report, indent=2))
    print(f"Saved portable model to {args.model}.")


if __name__ == "__main__":
    main()
