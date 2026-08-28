"""Preview every animated robot reaction without requiring a camera."""

from __future__ import annotations

import argparse
import time

import cv2

try:
    from .robot_face import render_face
    from .robot_state import Reaction
except ImportError:  # Supports direct execution: python src/face_demo.py
    from robot_face import render_face
    from robot_state import Reaction


REACTIONS = list(Reaction)
SUBTITLES = {
    Reaction.IDLE: "",
    Reaction.LISTENING: "I am listening.",
    Reaction.THINKING: "Let me think about that.",
    Reaction.SPEAKING: "Hello! It is great to meet you.",
    Reaction.HAPPY: "That was fun!",
    Reaction.PROUD: "Nice work!",
    Reaction.CONFUSED: "Could you say that again?",
    Reaction.HEART: "Thank you!",
    Reaction.ANNOYED: "Let's keep it kind, please.",
    Reaction.CURIOUS: "That is interesting!",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the robot's animated expression system.")
    parser.add_argument("--seconds", type=float, default=2.8, help="Seconds per automatic reaction.")
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be greater than zero.")

    index = 0
    changed_at = time.monotonic()
    window_name = "SPIS Robot Expression Test"
    try:
        while True:
            now = time.monotonic()
            if now - changed_at >= args.seconds:
                index = (index + 1) % len(REACTIONS)
                changed_at = now
            reaction = REACTIONS[index]
            frame = render_face(reaction, SUBTITLES[reaction], time_seconds=now)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                break
            if key in {81, ord("a")}:
                index = (index - 1) % len(REACTIONS)
                changed_at = now
            if key in {83, ord("d")}:
                index = (index + 1) % len(REACTIONS)
                changed_at = now
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
