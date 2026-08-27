from __future__ import annotations

import unittest

from src.robot_state import Action, Reaction, RobotController


class RobotStateTests(unittest.TestCase):
    def test_heart_gesture_produces_heart_reaction(self) -> None:
        command = RobotController().from_gesture("heart")
        self.assertEqual(command.reaction, Reaction.HEART)
        self.assertEqual(command.action, Action.NONE)

    def test_stop_gesture_stops_without_storing_any_identity(self) -> None:
        command = RobotController().from_gesture("stop")
        self.assertEqual(command.reaction, Reaction.LISTENING)
        self.assertEqual(command.action, Action.STOP)

