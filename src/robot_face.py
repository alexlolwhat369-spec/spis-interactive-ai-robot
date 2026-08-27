"""A cohesive animated expression system for the robot display."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

try:
    from .robot_state import Reaction
except ImportError:  # Supports direct execution: python src/live_demo.py
    from robot_state import Reaction


SCREEN = (18, 10, 3)
CYAN = (255, 232, 90)
CYAN_DIM = (110, 74, 20)
SUBTITLE = (240, 244, 242)


def _add_glow(frame: np.ndarray, marks: np.ndarray, strength: float = 0.65) -> None:
    halo = cv2.GaussianBlur(marks, (0, 0), 13)
    cv2.addWeighted(frame, 1.0, halo, strength, 0, frame)
    cv2.add(frame, marks, frame)


def _screen(width: int, height: int, phase: float) -> np.ndarray:
    frame = np.full((height, width, 3), SCREEN, dtype=np.uint8)
    # A restrained LED texture keeps the screen alive without competing with the face.
    for y in range(18, height, 8):
        intensity = 7 if int(y / 8 + phase * 2) % 3 else 11
        cv2.line(frame, (0, y), (width, y), (intensity, intensity // 2, 0), 1)
    return frame


def _round_eye(marks: np.ndarray, center: tuple[int, int], radius: tuple[int, int], angle: int = 0) -> None:
    cv2.ellipse(marks, center, radius, angle, 0, 360, CYAN, -1, cv2.LINE_AA)


def _arc(marks: np.ndarray, center: tuple[int, int], radius: tuple[int, int], start: int, end: int, thickness: int = 8) -> None:
    cv2.ellipse(marks, center, radius, 0, start, end, CYAN, thickness, cv2.LINE_AA)


def _line(marks: np.ndarray, start: tuple[int, int], end: tuple[int, int], thickness: int = 8) -> None:
    cv2.line(marks, start, end, CYAN, thickness, cv2.LINE_AA)


def _heart_outline(marks: np.ndarray, center: tuple[int, int], scale: float) -> None:
    cx, cy = center
    # One continuous parametric curve avoids the doubled top lobes from the
    # previous polygon-and-arcs construction.
    angle = np.linspace(0.0, math.tau, 180)
    size = 5.0 * scale
    x = 16.0 * np.sin(angle) ** 3
    y = 13.0 * np.cos(angle) - 5.0 * np.cos(2.0 * angle) - 2.0 * np.cos(3.0 * angle) - np.cos(4.0 * angle)
    points = np.column_stack((cx + x * size, cy - y * size)).astype(np.int32)
    cv2.polylines(marks, [points], True, CYAN, 8, cv2.LINE_AA)


def _voice_wave(marks: np.ndarray, center: tuple[int, int], phase: float) -> None:
    cx, cy = center
    # Rounded equalizer bars read as speech without the harsh spikes of a waveform.
    for index, offset in enumerate(range(-72, 73, 24)):
        height = int(14 + 22 * (0.5 + 0.5 * math.sin(phase * 6 + index * 0.9)))
        top, bottom = cy - height, cy + height
        cv2.line(marks, (cx + offset, top), (cx + offset, bottom), CYAN, 8, cv2.LINE_AA)
        cv2.circle(marks, (cx + offset, top), 4, CYAN, -1, cv2.LINE_AA)
        cv2.circle(marks, (cx + offset, bottom), 4, CYAN, -1, cv2.LINE_AA)


def _eyes_for(reaction: Reaction, marks: np.ndarray, cx: int, cy: int, phase: float) -> None:
    bob = int(math.sin(phase * 1.5) * 3)
    left = (cx - 104, cy - 38 + bob)
    right = (cx + 104, cy - 38 + bob)

    if reaction == Reaction.IDLE:
        _round_eye(marks, left, (24, 19))
        _round_eye(marks, right, (24, 19))
        _arc(marks, (cx, cy + 59), (47, 25), 20, 160, 7)
        return
    if reaction == Reaction.LISTENING:
        _round_eye(marks, left, (22, 22))
        _round_eye(marks, right, (22, 22))
        pulse = int(12 + 7 * (math.sin(phase * 5) + 1))
        for x in (cx - 27, cx, cx + 27):
            cv2.line(marks, (x, cy + 46 - pulse), (x, cy + 46 + pulse), CYAN, 6, cv2.LINE_AA)
        return
    if reaction == Reaction.SPEAKING:
        _round_eye(marks, left, (25, 16))
        _round_eye(marks, right, (25, 16))
        _voice_wave(marks, (cx, cy + 64), phase)
        return
    if reaction == Reaction.HAPPY:
        _line(marks, (cx - 138, cy - 27), (cx - 104, cy - 62), 8)
        _line(marks, (cx - 104, cy - 62), (cx - 70, cy - 27), 8)
        _line(marks, (cx + 70, cy - 27), (cx + 104, cy - 62), 8)
        _line(marks, (cx + 104, cy - 62), (cx + 138, cy - 27), 8)
        _arc(marks, (cx, cy + 47), (62, 35), 18, 162, 8)
        return
    if reaction == Reaction.PROUD:
        _arc(marks, left, (35, 20), 200, 342, 8)
        _arc(marks, right, (35, 20), 200, 342, 8)
        _arc(marks, (cx, cy + 42), (74, 42), 18, 162, 9)
        _line(marks, (cx - 152, cy - 80), (cx - 124, cy - 95), 5)
        _line(marks, (cx + 124, cy - 95), (cx + 152, cy - 80), 5)
        return
    if reaction == Reaction.THINKING:
        _arc(marks, left, (30, 18), 18, 162, 8)
        _arc(marks, right, (30, 18), 18, 162, 8)
        for index, x in enumerate((cx - 30, cx, cx + 30)):
            radius = 6 + int((math.sin(phase * 4 + index * 1.8) + 1) * 3)
            cv2.circle(marks, (x, cy + 58), radius, CYAN, -1, cv2.LINE_AA)
        return
    if reaction == Reaction.CONFUSED:
        _round_eye(marks, (left[0], left[1] - 9), (20, 16), -15)
        _arc(marks, (right[0], right[1] + 6), (31, 17), 18, 160, 8)
        _arc(marks, (cx, cy + 55), (40, 23), 200, 340, 7)
        return
    if reaction == Reaction.HEART:
        _heart_outline(marks, (cx, cy + 5), 0.88 + 0.06 * math.sin(phase * 4.5))
        return


def _subtitle(frame: np.ndarray, text: str, height: int) -> None:
    if text:
        cv2.putText(frame, text[:70], (34, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SUBTITLE, 2, cv2.LINE_AA)


def render_face(reaction: Reaction, subtitle: str = "", width: int = 800, height: int = 480, time_seconds: float | None = None) -> np.ndarray:
    """Render a single animated face frame with no dark pupils or eye cut-outs."""
    phase = time.monotonic() if time_seconds is None else time_seconds
    frame = _screen(width, height, phase)
    marks = np.zeros_like(frame)
    _eyes_for(reaction, marks, width // 2, height // 2 - 4, phase)
    _add_glow(frame, marks)
    _subtitle(frame, subtitle, height)
    return frame
