"""A cohesive animated expression system for the robot display."""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from .robot_state import Reaction
except ImportError:  # Supports direct execution: python src/live_demo.py
    from robot_state import Reaction


SCREEN = (18, 10, 3)
CYAN = (255, 232, 90)
CYAN_DIM = (110, 74, 20)
STAR = (80, 230, 255)
CORAL = (75, 85, 255)
MINT = (170, 255, 135)
SUBTITLE = (240, 244, 242)
ROOT = Path(__file__).resolve().parents[1]
MOHAN_IMAGE_PATH = ROOT / "assets" / "mohan.jpg"


def _add_glow(frame: np.ndarray, marks: np.ndarray, strength: float = 0.65) -> None:
    halo = cv2.GaussianBlur(marks, (0, 0), 13)
    cv2.addWeighted(frame, 1.0, halo, strength, 0, frame)
    cv2.add(frame, marks, frame)


def _screen(width: int, height: int, phase: float) -> np.ndarray:
    del phase
    return np.full((height, width, 3), SCREEN, dtype=np.uint8)


def _round_eye(marks: np.ndarray, center: tuple[int, int], radius: tuple[int, int], angle: int = 0) -> None:
    cv2.ellipse(marks, center, radius, angle, 0, 360, CYAN, -1, cv2.LINE_AA)


def _arc(
    marks: np.ndarray,
    center: tuple[int, int],
    radius: tuple[int, int],
    start: int,
    end: int,
    thickness: int = 8,
    color: tuple[int, int, int] = CYAN,
) -> None:
    cv2.ellipse(marks, center, radius, 0, start, end, color, thickness, cv2.LINE_AA)
    for angle in (start, end):
        radians = math.radians(angle)
        point = (
            int(center[0] + radius[0] * math.cos(radians)),
            int(center[1] + radius[1] * math.sin(radians)),
        )
        cv2.circle(marks, point, max(2, thickness // 2), color, -1, cv2.LINE_AA)


def _line(
    marks: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int = 8,
    color: tuple[int, int, int] = CYAN,
) -> None:
    cv2.line(marks, start, end, color, thickness, cv2.LINE_AA)
    radius = max(2, thickness // 2)
    cv2.circle(marks, start, radius, color, -1, cv2.LINE_AA)
    cv2.circle(marks, end, radius, color, -1, cv2.LINE_AA)


def _star(marks: np.ndarray, center: tuple[int, int], outer_radius: int) -> None:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer_radius if index % 2 == 0 else outer_radius * 0.43
        points.append((int(center[0] + math.cos(angle) * radius), int(center[1] + math.sin(angle) * radius)))
    cv2.fillPoly(marks, [np.array(points, dtype=np.int32)], STAR, cv2.LINE_AA)


def _star_outline(marks: np.ndarray, center: tuple[int, int], outer_radius: int) -> None:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer_radius if index % 2 == 0 else outer_radius * 0.46
        points.append((int(center[0] + math.cos(angle) * radius), int(center[1] + math.sin(angle) * radius)))
    cv2.polylines(marks, [np.array(points, dtype=np.int32)], True, CYAN, 8, cv2.LINE_AA)


def _heart_outline(
    marks: np.ndarray,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = CYAN,
    thickness: int = 8,
) -> None:
    cx, cy = center
    # One continuous parametric curve avoids the doubled top lobes from the
    # previous polygon-and-arcs construction.
    angle = np.linspace(0.0, math.tau, 180)
    size = 5.0 * scale
    x = 16.0 * np.sin(angle) ** 3
    y = 13.0 * np.cos(angle) - 5.0 * np.cos(2.0 * angle) - 2.0 * np.cos(3.0 * angle) - np.cos(4.0 * angle)
    points = np.column_stack((cx + x * size, cy - y * size)).astype(np.int32)
    cv2.polylines(marks, [points], True, color, thickness, cv2.LINE_AA)


def _closed_eye(marks: np.ndarray, center: tuple[int, int], radius: tuple[int, int] = (43, 26)) -> None:
    _arc(marks, center, radius, 200, 340, 14)


def _check(marks: np.ndarray, center: tuple[int, int], scale: float = 1.0) -> None:
    cx, cy = center
    _line(marks, (int(cx - 38 * scale), int(cy - 1 * scale)), (int(cx - 10 * scale), int(cy + 27 * scale)), 14, MINT)
    _line(marks, (int(cx - 10 * scale), int(cy + 27 * scale)), (int(cx + 46 * scale), int(cy - 36 * scale)), 14, MINT)


def _crown(marks: np.ndarray, center: tuple[int, int], pulse: float) -> None:
    cx, cy = center
    lift = int(3 * pulse)
    points = np.array(
        [
            (cx - 58, cy + 21 - lift),
            (cx - 44, cy - 31 - lift),
            (cx - 11, cy - 1 - lift),
            (cx + 14, cy - 39 - lift),
            (cx + 46, cy + 1 - lift),
            (cx + 60, cy - 25 - lift),
            (cx + 52, cy + 30 - lift),
            (cx - 52, cy + 30 - lift),
        ],
        dtype=np.int32,
    )
    cv2.polylines(marks, [points], True, CYAN, 11, cv2.LINE_AA)


def _side_equalizer(marks: np.ndarray, x: int, cy: int, phase: float) -> None:
    for index, (offset, base_height) in enumerate(zip(range(-28, 29, 14), (6, 14, 25, 14, 6))):
        height = base_height + int(3 * math.sin(phase * 6 + index * 0.8))
        _line(marks, (x + offset, cy - height), (x + offset, cy + height), 7)


def _anger_eye(marks: np.ndarray, center: tuple[int, int], mirrored: bool = False) -> None:
    cx, cy = center
    direction = -1 if mirrored else 1
    points = np.array(
        [
            (cx - 49 * direction, cy - 6),
            (cx + 39 * direction, cy - 29),
            (cx + 31 * direction, cy + 24),
            (cx - 20 * direction, cy + 19),
        ],
        dtype=np.int32,
    )
    cv2.polylines(marks, [points], True, CYAN, 12, cv2.LINE_AA)


def _anger_mark(marks: np.ndarray, center: tuple[int, int], pulse: float) -> None:
    cx, cy = center
    reach = 13 + int(3 * pulse)
    _line(marks, (cx, cy), (cx - reach, cy - reach), 5, CORAL)
    _line(marks, (cx + 4, cy - 5), (cx + reach, cy - reach - 4), 5, CORAL)
    _line(marks, (cx + 5, cy + 3), (cx + reach + 3, cy + reach), 5, CORAL)


def _thinking_squint(marks: np.ndarray, center: tuple[int, int]) -> None:
    cx, cy = center
    points = np.array(
        [
            (cx - 45, cy + 9),
            (cx - 27, cy - 2),
            (cx + 3, cy - 10),
            (cx + 38, cy - 7),
            (cx + 22, cy + 1),
            (cx - 12, cy + 5),
        ],
        dtype=np.int32,
    )
    cv2.polylines(marks, [points], False, CYAN, 12, cv2.LINE_AA)
    cv2.circle(marks, tuple(points[0]), 6, CYAN, -1, cv2.LINE_AA)
    cv2.circle(marks, tuple(points[-1]), 6, CYAN, -1, cv2.LINE_AA)


def _voice_wave(marks: np.ndarray, center: tuple[int, int], phase: float) -> None:
    cx, cy = center
    base_heights = (10, 20, 34, 48, 34, 20, 10)
    for index, (offset, base_height) in enumerate(zip(range(-84, 85, 28), base_heights)):
        motion = 0.82 + 0.18 * (0.5 + 0.5 * math.sin(phase * 6 + index * 0.8))
        height = int(base_height * motion)
        _line(marks, (cx + offset, cy - height), (cx + offset, cy + height), 12)


@lru_cache(maxsize=1)
def _load_mohan_image() -> np.ndarray | None:
    image = cv2.imread(str(MOHAN_IMAGE_PATH))
    return image if image is not None and image.size else None


def _mohan_reaction(width: int, height: int, phase: float) -> np.ndarray:
    source = _load_mohan_image()
    if source is None:
        frame = _screen(width, height, phase)
    else:
        source_height, source_width = source.shape[:2]
        scale = max(width / source_width, height / source_height)
        resized = cv2.resize(
            source,
            (max(width, int(source_width * scale)), max(height, int(source_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        crop_x = max(0, (resized.shape[1] - width) // 2)
        face_center_y = int(resized.shape[0] * 0.36)
        crop_y = min(max(0, face_center_y - height // 2), resized.shape[0] - height)
        frame = resized[crop_y : crop_y + height, crop_x : crop_x + width].copy()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 78), SCREEN, -1)
    cv2.rectangle(overlay, (0, height - 76), (width, height), SCREEN, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, "MOHAN", (32, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.35, CYAN, 3, cv2.LINE_AA)
    return frame


def _eyes_for(reaction: Reaction, marks: np.ndarray, cx: int, cy: int, phase: float) -> None:
    bob = int(math.sin(phase * 1.5) * 3)
    left = (cx - 145, cy - 42 + bob)
    right = (cx + 145, cy - 42 + bob)

    if reaction == Reaction.IDLE:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        _arc(marks, (cx, cy + 58 + bob), (45, 27), 20, 160, 12)
        return
    if reaction == Reaction.LISTENING:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        _side_equalizer(marks, cx - 258, cy - 40, phase)
        _side_equalizer(marks, cx + 258, cy - 40, phase)
        return
    if reaction == Reaction.SPEAKING:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        _voice_wave(marks, (cx, cy + 72), phase)
        return
    if reaction == Reaction.HAPPY:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        _arc(marks, (cx, cy + 58), (64, 37), 18, 162, 13)
        cheek_width = 14 + int(2 * (1 + math.sin(phase * 3)))
        _line(marks, (cx - 226 - cheek_width, cy + 36), (cx - 226 + cheek_width, cy + 36), 11)
        _line(marks, (cx + 226 - cheek_width, cy + 36), (cx + 226 + cheek_width, cy + 36), 11)
        return
    if reaction == Reaction.PROUD:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        _arc(marks, (cx, cy + 59), (58, 31), 18, 162, 12)
        pulse = 0.5 + 0.5 * math.sin(phase * 3.5)
        _crown(marks, (cx, cy - 132), pulse)
        _line(marks, (cx - 105, cy - 128), (cx - 127, cy - 147), 8)
        _line(marks, (cx - 89, cy - 151), (cx - 103, cy - 174), 8)
        _line(marks, (cx + 105, cy - 128), (cx + 127, cy - 147), 8)
        _line(marks, (cx + 89, cy - 151), (cx + 103, cy - 174), 8)
        return
    if reaction == Reaction.THINKING:
        _closed_eye(marks, left)
        _thinking_squint(marks, right)
        for index, offset in enumerate(range(-30, 31, 15)):
            height = int(8 + 12 * (0.5 + 0.5 * math.sin(phase * 4 + index)))
            _line(marks, (cx + offset, cy - 132 - height), (cx + offset, cy - 132 + height), 7)
        return
    if reaction == Reaction.CONFUSED:
        _closed_eye(marks, left)
        _line(marks, (right[0] - 36, right[1]), (right[0] + 34, right[1]), 12)
        question_y = right[1] - 66 + int(5 * math.sin(phase * 3))
        cv2.putText(marks, "?", (right[0] + 48, question_y), cv2.FONT_HERSHEY_SIMPLEX, 1.55, CYAN, 10, cv2.LINE_AA)
        _line(marks, (cx - 40, cy + 71), (cx + 42, cy + 54), 12)
        return
    if reaction == Reaction.ANNOYED:
        shake = int(3 * math.sin(phase * 17))
        angry_left = (left[0] + shake, left[1])
        angry_right = (right[0] + shake, right[1])
        _anger_eye(marks, angry_left)
        _anger_eye(marks, angry_right, mirrored=True)
        _line(marks, (angry_left[0] - 50, angry_left[1] - 53), (angry_left[0] + 41, angry_left[1] - 21), 13, CORAL)
        _line(marks, (angry_right[0] - 41, angry_right[1] - 21), (angry_right[0] + 50, angry_right[1] - 53), 13, CORAL)
        _arc(marks, (cx + shake, cy + 81), (61, 34), 200, 340, 12)
        _anger_mark(marks, (cx + 236, cy - 126), 0.5 + 0.5 * math.sin(phase * 7))
        return
    if reaction == Reaction.CURIOUS:
        size = 51 + int((math.sin(phase * 4.0) + 1.0) * 4)
        _star_outline(marks, left, size)
        _star_outline(marks, right, size)
        _arc(marks, (cx, cy + 65), (63, 34), 18, 162, 12)
        return
    if reaction == Reaction.HEART:
        pulse = 0.68 + 0.045 * math.sin(phase * 4.5)
        _heart_outline(marks, left, pulse)
        _heart_outline(marks, right, pulse)
        _arc(marks, (cx, cy + 72), (58, 31), 18, 162, 12)
        for cheek_x in (cx - 244, cx + 214):
            _line(marks, (cheek_x - 7, cy + 36), (cheek_x + 5, cy + 21), 8, CORAL)
            _line(marks, (cheek_x + 12, cy + 40), (cheek_x + 24, cy + 25), 8, CORAL)
        return
    if reaction == Reaction.OK:
        _closed_eye(marks, left)
        _closed_eye(marks, right)
        scale = 1.02 + 0.06 * math.sin(phase * 4)
        _check(marks, (cx, cy + 66), scale)
        return


def _wrapped_lines(text: str, max_width: int, font_scale: float, thickness: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:2]


def _subtitle(frame: np.ndarray, text: str, width: int, height: int) -> None:
    if not text:
        return
    font_scale = 0.68
    thickness = 2
    lines = _wrapped_lines(text, width - 68, font_scale, thickness)
    start_y = height - 28 - (len(lines) - 1) * 30
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (34, start_y + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            SUBTITLE,
            thickness,
            cv2.LINE_AA,
        )


def _status_header(frame: np.ndarray, status: str, music_title: str | None, width: int) -> None:
    if status:
        cv2.circle(frame, (34, 30), 6, CYAN, -1, cv2.LINE_AA)
        cv2.putText(frame, status[:18], (51, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, SUBTITLE, 2, cv2.LINE_AA)
    if music_title:
        label = f"MUSIC  {music_title}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(
            frame,
            label[:42],
            (max(34, width - text_width - 28), 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            CYAN,
            1,
            cv2.LINE_AA,
        )


def _microphone_meter(frame: np.ndarray, level: float, width: int) -> None:
    level = max(0.0, min(1.0, level))
    x = width - 154
    for index in range(6):
        active = level >= (index + 1) / 7
        color = MINT if active else (34, 58, 66)
        bar_height = 8 + index * 5
        cv2.rectangle(frame, (x + index * 18, 62 - bar_height), (x + index * 18 + 10, 62), color, -1)


def render_face(
    reaction: Reaction,
    subtitle: str = "",
    width: int = 800,
    height: int = 480,
    time_seconds: float | None = None,
    *,
    speaking: bool = False,
    input_level: float = 0.0,
    status: str = "",
    music_title: str | None = None,
) -> np.ndarray:
    """Render a single animated face frame with no dark pupils or eye cut-outs."""
    phase = time.monotonic() if time_seconds is None else time_seconds
    if reaction == Reaction.MOHAN:
        frame = _mohan_reaction(width, height, phase)
        if speaking:
            marks = np.zeros_like(frame)
            _voice_wave(marks, (width // 2, height - 126), phase)
            _add_glow(frame, marks)
        _status_header(frame, status, music_title, width)
        _subtitle(frame, subtitle, width, height)
        return frame
    frame = _screen(width, height, phase)
    marks = np.zeros_like(frame)
    _eyes_for(reaction, marks, width // 2, height // 2 - 4, phase)
    if speaking and reaction != Reaction.SPEAKING:
        center_y = height // 2 - 4
        marks[center_y + 8 : center_y + 150] = 0
        _voice_wave(marks, (width // 2, center_y + 74), phase)
    _add_glow(frame, marks)
    _status_header(frame, status, music_title, width)
    if reaction == Reaction.LISTENING:
        _microphone_meter(frame, input_level, width)
    _subtitle(frame, subtitle, width, height)
    return frame
