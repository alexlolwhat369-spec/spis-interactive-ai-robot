"""Show whether a selected microphone is receiving sound before running the robot."""

from __future__ import annotations

import argparse
import queue
import time

import cv2
import numpy as np
import sounddevice as sd


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a microphone's live input level.")
    parser.add_argument("--device", type=int, default=1, help="Input-device number from voice_demo.py --list-microphones.")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--window", action="store_true", help="Show a large graphical meter instead of only terminal bars.")
    args = parser.parse_args()
    info = sd.query_devices(args.device, "input")
    sample_rate = 16000
    sd.check_input_settings(device=args.device, samplerate=sample_rate, channels=1, dtype="float32")
    readings: queue.Queue[float] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        readings.put(float(np.max(np.abs(indata))))

    print(f"Testing microphone {args.device}: {info['name']}")
    print("Speak normally now. A bar means this microphone hears you.")
    window_name = "SPIS Robot Microphone Test"
    if args.window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 480)
    deadline = time.monotonic() + args.seconds
    try:
        with sd.InputStream(
            device=args.device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while time.monotonic() < deadline:
                try:
                    level = readings.get(timeout=0.05 if args.window else 0.25)
                except queue.Empty:
                    level = 0.0
                bars = "#" * min(30, int(level * 180))
                print(f"{level:0.3f} {bars}")
                if args.window:
                    _show_meter(window_name, level, args.device, info["name"], deadline - time.monotonic())
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if args.window:
            cv2.destroyWindow(window_name)
    print("Microphone test complete.")


def _show_meter(window_name: str, level: float, device: int, name: str, seconds_remaining: float) -> None:
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(frame, "Microphone check", (42, 74), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (86, 230, 255), 2)
    cv2.putText(frame, f"Device {device}: {name[:46]}", (42, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(frame, "Speak normally. Green means the microphone hears you.", (42, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 220, 220), 1)
    cv2.rectangle(frame, (42, 240), (758, 328), (95, 95, 95), 2)
    width = int(min(1.0, level * 8.0) * 712)
    color = (60, 230, 90) if level >= 0.015 else (0, 130, 255)
    cv2.rectangle(frame, (44, 242), (44 + width, 326), color, -1)
    cv2.putText(frame, f"Level: {level:.3f}", (42, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"{max(0, seconds_remaining):.0f} seconds remaining | Q closes", (42, 432), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (175, 175, 175), 1)
    cv2.imshow(window_name, frame)


if __name__ == "__main__":
    main()
