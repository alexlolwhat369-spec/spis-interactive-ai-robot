"""Tests for the camera-free expression preview configuration."""

from __future__ import annotations

import unittest

from src.face_demo import REACTIONS, SUBTITLES
from src.robot_state import Reaction


class FaceDemoTests(unittest.TestCase):
    def test_every_reaction_has_a_preview_subtitle(self) -> None:
        self.assertEqual(REACTIONS, list(Reaction))
        self.assertEqual(set(SUBTITLES), set(Reaction))
