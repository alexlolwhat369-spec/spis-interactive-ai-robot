"""Capture landmark-only gesture samples from a camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from .capture_quality import CaptureGate
    from .data_io import append_sample
    from .gesture_features import HandSample, landmarks_to_features, required_hands
    from .hand_tracker import HandTracker, draw_hands
except ImportError:  # Supports direct execution: python src/collect_samples.py
    from capture_quality import CaptureGate
    from data_io import append_sample
    from gesture_features import HandSample, landmarks_to_features, required_hands
    from hand_tracker import HandTracker, draw_hands

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "landmarks.csv"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture numeric hand-landmark samples.")
    parser.add_argument("--label", required=True, choices=["wave", "thumbs_up", "peace", "stop", "heart"])
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--auto", action="store_true", help="Capture varied samples automatically.")
    parser.add_argument("--cooldown-ms", type=int, default=350)
    parser.add_argument("--min-distance", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be at least 1.")
    needed = required_hands(args.label)
    captured = 0
    gate = CaptureGate(min_distance=args.min_distance, cooldown_ms=args.cooldown_ms)
    capture_status = "ready"
    # This is the same OpenCV camera opening path used for the working thumbs-up capture.
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    tracker = HandTracker(HAND_MODEL_PATH)
    try:
        while captured < args.samples:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read.")
            frame = cv2.flip(frame, 1)
            hand_samples = tracker.detect(frame)
            draw_hands(frame, hand_samples)

            valid = len(hand_samples) == needed
            color = (0, 220, 0) if valid else (0, 80, 255)
            control = "AUTO" if args.auto else "C capture"
            message = f"{args.label}: {captured}/{args.samples} | {control} | Q quit"
            cv2.putText(frame, message, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Show {needed} hand(s)", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, capture_status, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow("Gesture sample capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            should_capture = args.auto or key == ord("c")
            if should_capture and valid:
                features = landmarks_to_features(hand_samples)
                allowed, capture_status = gate.can_capture(features, time.monotonic() * 1000)
                if allowed:
                    append_sample(DATASET_PATH, args.label, features)
                    captured += 1
            elif should_capture:
                capture_status = f"show exactly {needed} hand(s)"

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {captured} numeric samples for '{args.label}' to {DATASET_PATH}.")


if __name__ == "__main__":
    main()
