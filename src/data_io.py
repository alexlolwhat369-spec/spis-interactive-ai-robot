"""Read and append landmark-only CSV datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .gesture_features import FEATURE_SIZE
except ImportError:  # Supports direct execution: python src/train.py
    from gesture_features import FEATURE_SIZE


def append_sample(path: Path, label: str, features: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if len(features) != FEATURE_SIZE:
        raise ValueError(f"Expected {FEATURE_SIZE} features, got {len(features)}.")
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["label", *[f"f_{index}" for index in range(FEATURE_SIZE)]])
        writer.writerow([label, *np.asarray(features, dtype=np.float32).tolist()])


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The dataset is empty.")
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    features = np.asarray(
        [[float(row[f"f_{index}"]) for index in range(FEATURE_SIZE)] for row in rows],
        dtype=np.float32,
    )
    return features, labels
