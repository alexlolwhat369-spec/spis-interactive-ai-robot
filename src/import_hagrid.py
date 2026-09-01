"""Extract a small, landmark-only gesture dataset from HaGRID annotations.

The script never reads HaGRID images. It retains only the gesture label and
MediaPipe-style hand landmarks, and writes a new compact CSV for this project.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .gesture_features import HandSample, landmarks_to_features
except ImportError:  # Supports direct execution: python src/import_hagrid.py
    from gesture_features import HandSample, landmarks_to_features


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABELS = {
    "like": "thumbs_up",
    "peace": "peace",
    "stop": "stop",
    "middle_finger": "middle_finger",
    "ok": "ok",
    "hand_heart": "heart",
    "hand_heart2": "heart",
}


def _as_label_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _to_hand(points: object) -> HandSample | None:
    try:
        values = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if values.shape != (21, 3):
        return None
    return HandSample(points=values)


def _record_samples(record: object) -> Iterator[tuple[str, np.ndarray]]:
    """Yield only mapped gesture features. Metadata is deliberately ignored."""
    if not isinstance(record, dict):
        return
    raw_hands = record.get("hand_landmarks")
    if not isinstance(raw_hands, list):
        return
    hands = [hand for raw_hand in raw_hands if (hand := _to_hand(raw_hand)) is not None]
    if not hands:
        return

    united_labels = _as_label_list(record.get("united_label"))
    if any(label in ("hand_heart", "hand_heart2") for label in united_labels) and len(hands) >= 2:
        yield "heart", landmarks_to_features(hands[:2])

    labels = _as_label_list(record.get("labels"))
    for label, hand in zip(labels, hands):
        target = SOURCE_LABELS.get(label)
        if target and target != "heart":
            yield target, landmarks_to_features([hand])


def extract_samples(annotation_dir: Path, max_per_class: int, seed: int) -> dict[str, list[np.ndarray]]:
    """Reservoir-sample a balanced cap of landmarks from one or more JSON files."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be at least 1.")
    files = sorted(annotation_dir.rglob("*.json")) if annotation_dir.is_dir() else [annotation_dir]
    if not files:
        raise FileNotFoundError(f"No JSON annotation files found at {annotation_dir}.")

    rng = np.random.default_rng(seed)
    selected = {target: [] for target in set(SOURCE_LABELS.values())}
    seen = {target: 0 for target in selected}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for record in payload.values():
            for label, features in _record_samples(record):
                seen[label] += 1
                bucket = selected[label]
                if len(bucket) < max_per_class:
                    bucket.append(features)
                    continue
                replacement = int(rng.integers(seen[label]))
                if replacement < max_per_class:
                    bucket[replacement] = features
    return selected


def write_landmark_dataset(path: Path, samples: dict[str, list[np.ndarray]]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_size = 126
    counts = {label: len(values) for label, values in sorted(samples.items())}
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", *[f"f_{index}" for index in range(feature_size)]])
        for label in sorted(samples):
            for values in samples[label]:
                writer.writerow([label, *np.asarray(values, dtype=np.float32).tolist()])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small landmark-only HaGRID subset.")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "hagrid_landmarks.csv")
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = extract_samples(args.annotations_dir, args.max_per_class, args.seed)
    counts = write_landmark_dataset(args.output, samples)
    report = {
        "source": "HaGRID annotations only",
        "images_saved": 0,
        "identity_metadata_saved": 0,
        "max_per_class": args.max_per_class,
        "samples_per_class": counts,
        "labels": SOURCE_LABELS,
    }
    report_path = ROOT / "reports" / "hagrid_import.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved landmark-only dataset to {args.output}.")


if __name__ == "__main__":
    main()
