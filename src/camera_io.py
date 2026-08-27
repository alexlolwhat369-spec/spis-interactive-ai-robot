"""Open a Windows camera using a backend that can actually deliver frames."""

from __future__ import annotations

import os
from collections.abc import Callable

import cv2
import numpy as np


CameraFactory = Callable[[int, int | None], cv2.VideoCapture]


def backend_candidates(platform_name: str | None = None) -> list[tuple[str, int | None, bool]]:
    """Try OpenCV's original automatic choice before Windows-specific fallbacks."""
    platform_name = platform_name or os.name
    if platform_name == "nt":
        return [
            ("OpenCV default", None, False),
            ("DirectShow MJPEG 640x480", cv2.CAP_DSHOW, True),
            ("DirectShow", cv2.CAP_DSHOW, False),
            ("Media Foundation", cv2.CAP_MSMF, False),
        ]
    return [("default", None, False)]


def _default_factory(index: int, backend: int | None) -> cv2.VideoCapture:
    return cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)


def configure_usb_camera(camera: cv2.VideoCapture, use_mjpeg: bool) -> None:
    """Request a conservative UVC camera mode before reading frames."""
    if not use_mjpeg:
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def is_usable_frame(frame: np.ndarray | None) -> bool:
    """Reject empty and uniformly black backend placeholder frames."""
    if frame is None or frame.size == 0:
        return False
    return bool(float(np.mean(frame)) >= 8.0 and float(np.std(frame)) >= 3.0)


def open_working_camera(
    index: int,
    *,
    factory: CameraFactory = _default_factory,
    platform_name: str | None = None,
    attempts_per_backend: int = 30,
) -> tuple[cv2.VideoCapture, np.ndarray, str]:
    """Return an open camera, its first frame, and the backend name.

    A camera that opens but cannot produce a frame is released before trying the
    next backend. This is common with Windows webcam drivers.
    """
    for backend_name, backend, use_mjpeg in backend_candidates(platform_name):
        camera = factory(index, backend)
        if not camera.isOpened():
            camera.release()
            continue
        configure_usb_camera(camera, use_mjpeg)
        for _ in range(attempts_per_backend):
            ok, frame = camera.read()
            if ok and is_usable_frame(frame):
                return camera, frame, backend_name
        camera.release()
    raise RuntimeError(
        f"Camera {index} opened but delivered only black or empty video. Close camera apps, "
        "allow desktop-app camera access in Windows Privacy settings, then retry."
    )
