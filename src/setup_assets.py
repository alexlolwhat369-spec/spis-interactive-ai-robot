"""Download the one official MediaPipe model needed for local hand tracking."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official MediaPipe hand model.")
    parser.add_argument("--force", action="store_true", help="Replace an existing model file.")
    args = parser.parse_args()
    if MODEL_PATH.exists() and not args.force:
        print(f"Model already exists: {MODEL_PATH}")
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".task.part")
    print(f"Downloading hand model from {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        if temporary_path.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model is unexpectedly small.")
        temporary_path.replace(MODEL_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()

