"""Small Windows push-to-talk key reader with no third-party keyboard hook."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable


class SpaceKey:
    """Read the physical Space key while the robot windows are visible."""

    def __init__(self, pressed_reader: Callable[[], bool] | None = None) -> None:
        self._pressed_reader = pressed_reader or self._windows_space_is_down

    def is_down(self) -> bool:
        return self._pressed_reader()

    @staticmethod
    def _windows_space_is_down() -> bool:
        if os.name != "nt":
            return False
        # GetAsyncKeyState reads the held state, unlike cv2.waitKey which only
        # reports key presses and cannot reliably detect a release.
        state = ctypes.WinDLL("user32", use_last_error=True).GetAsyncKeyState(0x20)
        return bool(state & 0x8000)
