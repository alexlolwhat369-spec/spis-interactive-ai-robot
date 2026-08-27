from __future__ import annotations

import unittest

from src.robot_face import render_face
from src.robot_state import Reaction


class RobotFaceTests(unittest.TestCase):
    def test_each_face_state_renders_pixels(self) -> None:
        for reaction in Reaction:
            frame = render_face(reaction, "Hello")
            self.assertEqual(frame.shape, (480, 800, 3))
            self.assertGreater(int(frame.sum()), 0)

    def test_animation_changes_pixels_over_time(self) -> None:
        first = render_face(Reaction.SPEAKING, "Hello", time_seconds=0.0)
        later = render_face(Reaction.SPEAKING, "Hello", time_seconds=0.18)
        self.assertGreater(int(abs(first.astype(int) - later.astype(int)).sum()), 0)

    def test_reactions_have_distinct_visual_signatures(self) -> None:
        signatures = {
            render_face(reaction, time_seconds=0.4).tobytes()
            for reaction in Reaction
        }
        self.assertEqual(len(signatures), len(Reaction))
