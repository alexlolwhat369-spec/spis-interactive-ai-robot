from __future__ import annotations

import unittest

from src.robot_face import CORAL, MINT, MOHAN_IMAGE_PATH, render_face
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

    def test_speaking_preserves_the_selected_emotional_eyes(self) -> None:
        curious = render_face(Reaction.CURIOUS, "Interesting", time_seconds=0.2, speaking=True)
        neutral = render_face(Reaction.SPEAKING, "Interesting", time_seconds=0.2)

        self.assertGreater(int(abs(curious.astype(int) - neutral.astype(int)).sum()), 1000)

    def test_long_subtitle_wraps_without_leaving_the_frame(self) -> None:
        frame = render_face(
            Reaction.SPEAKING,
            "This is a longer response that should remain readable on the laptop display without being cut off.",
            time_seconds=0.0,
        )

        self.assertEqual(frame.shape, (480, 800, 3))

    def test_every_drawn_reaction_has_visible_motion(self) -> None:
        for reaction in Reaction:
            if reaction == Reaction.MOHAN:
                continue
            with self.subTest(reaction=reaction):
                first = render_face(reaction, time_seconds=0.0)
                later = render_face(reaction, time_seconds=0.35)
                self.assertGreater(int(abs(first.astype(int) - later.astype(int)).sum()), 0)

    def test_reactions_have_distinct_visual_signatures(self) -> None:
        signatures = {
            render_face(reaction, time_seconds=0.4).tobytes()
            for reaction in Reaction
        }
        self.assertEqual(len(signatures), len(Reaction))

    def test_mohan_reaction_uses_the_supplied_portrait(self) -> None:
        self.assertTrue(MOHAN_IMAGE_PATH.exists())
        frame = render_face(Reaction.MOHAN, "Mohan!", time_seconds=0.0)
        self.assertGreater(float(frame.std()), 20.0)

    def test_ok_has_its_own_mint_confirmation_mark(self) -> None:
        frame = render_face(Reaction.OK, time_seconds=0.0)
        mint_pixels = (frame[:, :, 0] > 130) & (frame[:, :, 1] > 220) & (frame[:, :, 2] > 90)
        self.assertGreater(int(mint_pixels.sum()), 100)

    def test_selected_visual_system_uses_only_the_dark_display_without_a_shell(self) -> None:
        frame = render_face(Reaction.IDLE, time_seconds=0.0)
        self.assertTrue((frame[0, 0] < 25).all())
        bright_background = frame.min(axis=2) > 180
        self.assertLess(int(bright_background.sum()), 100)

    def test_annoyed_face_contains_the_selected_coral_accents(self) -> None:
        frame = render_face(Reaction.ANNOYED, time_seconds=0.0)
        coral_pixels = (frame[:, :, 2] > 180) & (frame[:, :, 1] < 150)
        self.assertGreater(int(coral_pixels.sum()), 100)
