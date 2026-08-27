from __future__ import annotations

import unittest

from src.music import MusicSelector, Track


class MusicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = MusicSelector(
            [Track("Calm", "calm", "calm.wav"), Track("Warm", "warm", "warm.wav"), Track("Celebrate", "celebration", "win.wav")]
        )

    def test_heart_uses_warm_track(self) -> None:
        self.assertEqual(self.selector.choose(gesture="heart").category, "warm")

    def test_explicit_request_overrides_event(self) -> None:
        self.assertEqual(self.selector.choose(requested_category="calm", game_won=True).category, "calm")

