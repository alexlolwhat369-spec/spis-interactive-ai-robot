from __future__ import annotations

import unittest

from src.push_to_talk import SpaceKey


class PushToTalkTests(unittest.TestCase):
    def test_uses_the_injected_held_key_reader(self) -> None:
        held = {"value": False}
        key = SpaceKey(lambda: held["value"])

        self.assertFalse(key.is_down())
        held["value"] = True
        self.assertTrue(key.is_down())
