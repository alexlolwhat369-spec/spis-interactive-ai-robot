"""Measure webcam frames without displaying or saving them."""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

try:
    from .camera_io import backend_candidates, configure_usb_camera, is_usable_frame
except ImportError:  # Supports direct execution: python src/diagnose_camera.py
    from camera_io import backend_candidates, configure_usb_camera, is_usable_frame


def probe_camera(index: int, reads: int = 30) -> list[dict[str, object]]:
    """Return non-identifying image statistics for each supported video backend."""
    results: list[dict[str, object]] = []
    for name, backend, use_mjpeg in backend_candidates():
        camera = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        opened = bool(camera.isOpened())
        frames: list[np.ndarray] = []
        if opened:
            configure_usb_camera(camera, use_mjpeg)
            for _ in range(reads):
                ok, frame = camera.read()
                if ok and frame is not None:
                    frames.append(frame)
        camera.release()
        if not frames:
            results.append({"backend": name, "opened": opened, "frames": 0, "usable": False})
            continue
        means = [float(np.mean(frame)) for frame in frames]
        deviations = [float(np.std(frame)) for frame in frames]
        results.append(
            {
                "backend": name,
                "opened": opened,
                "frames": len(frames),
                "mean_brightness": round(float(np.mean(means)), 2),
                "mean_variation": round(float(np.mean(deviations)), 2),
                "usable": any(is_usable_frame(frame) for frame in frames),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect webcam signal statistics without saving video.")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps({"camera": args.camera, "results": probe_camera(args.camera)}, indent=2))


if __name__ == "__main__":
    main()
