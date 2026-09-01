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

    def test_new_gestures_have_explicit_reactions(self) -> None:
        controller = RobotController()
        self.assertEqual(controller.from_gesture("middle_finger").reaction, Reaction.ANNOYED)
        self.assertEqual(controller.from_gesture("ok").reaction, Reaction.OK)
        self.assertNotEqual(controller.from_gesture("ok").reaction, Reaction.PROUD)
        self.assertEqual(controller.from_gesture("mohan").reaction, Reaction.MOHAN)

    def test_unrecognized_gesture_cannot_show_mohan(self) -> None:
        command = RobotController().from_gesture("not_a_real_label")
        self.assertEqual(command.reaction, Reaction.CONFUSED)
