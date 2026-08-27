"""Tests for camera backend selection without accessing physical hardware."""

from __future__ import annotations

import unittest

import numpy as np

from src.camera_io import backend_candidates, is_usable_frame


class CameraBackendTests(unittest.TestCase):
    def test_windows_tries_opencv_default_before_specific_backends(self) -> None:
        names = [name for name, _, _ in backend_candidates("nt")]
        self.assertEqual(
            names,
            ["OpenCV default", "DirectShow MJPEG 640x480", "DirectShow", "Media Foundation"],
        )

    def test_non_windows_uses_default_backend(self) -> None:
        self.assertEqual([name for name, _, _ in backend_candidates("posix")], ["default"])

    def test_black_placeholder_frame_is_rejected(self) -> None:
        self.assertFalse(is_usable_frame(np.zeros((20, 20, 3), dtype=np.uint8)))

    def test_visible_frame_is_accepted(self) -> None:
        frame = np.full((20, 20, 3), 10, dtype=np.uint8)
        frame[:, :10, 0] = 30
        self.assertTrue(is_usable_frame(frame))

    def test_dark_low_variation_frame_is_rejected(self) -> None:
        self.assertFalse(is_usable_frame(np.full((20, 20, 3), 7, dtype=np.uint8)))
