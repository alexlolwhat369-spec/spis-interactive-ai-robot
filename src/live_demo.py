"""Run the trained gesture model with a live camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .robot_face import render_face
    from .robot_state import RobotController
except ImportError:  # Supports direct execution: python src/live_demo.py
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from robot_face import render_face
    from robot_state import RobotController

ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the trained robot gesture model.")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "gesture_knn.npz")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Train the model first: {args.model}")

    model = GestureKNN.load(args.model)
    gate = GestureGate(distance_limit=model.distance_limit)
    controller = RobotController()
    previous_label = "none"
    previous_face: np.ndarray | None = None
    face_transition_started = time.monotonic()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    tracker = HandTracker(HAND_MODEL_PATH)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand_samples = tracker.detect(frame)
            if not hand_samples:
                prediction = Prediction("none", 1.0, 0.0)
            else:
                prediction = model.predict(landmarks_to_features(hand_samples))
            label = gate.update(prediction, len(hand_samples))
            if label != previous_label:
                command = controller.from_gesture(label)
                previous_label = label
                face_transition_started = time.monotonic()
            else:
                command = controller.from_gesture("none") if label == "none" else controller.from_gesture(label)
            reaction = command.reaction

            draw_hands(frame, hand_samples)
            cv2.putText(frame, f"Gesture: {label}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
            cv2.putText(frame, f"Robot: {reaction}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
            if label not in {"none", "unknown"}:
                cv2.putText(frame, f"Confidence: {prediction.confidence:.0%}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
            now = time.monotonic()
            target_face = render_face(reaction, command.reply, time_seconds=now)
            progress = min(1.0, (now - face_transition_started) / 0.28)
            face = target_face if previous_face is None else cv2.addWeighted(previous_face, 1.0 - progress, target_face, progress, 0)
            if progress >= 1.0:
                previous_face = target_face
            cv2.imshow("SPIS Robot Face", face)
            cv2.imshow("SPIS Robot Gesture Demo - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
