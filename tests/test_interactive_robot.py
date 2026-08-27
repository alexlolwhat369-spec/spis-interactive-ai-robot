from __future__ import annotations

import unittest

from src.interactive_robot import GestureFeedback, VoiceActivity, VoiceState, gestures_locked
from src.robot_state import Reaction, RobotCommand, RobotController


class InteractiveRobotTests(unittest.TestCase):
    def test_voice_state_is_safe_to_read_and_update(self) -> None:
        state = VoiceState()
        self.assertEqual(state.current().reaction, Reaction.IDLE)
        state.set(Reaction.LISTENING, "Listening...")
        self.assertEqual(state.current().reaction, Reaction.LISTENING)
        self.assertEqual(state.current().subtitle, "Listening...")

    def test_heart_feedback_stays_visible_after_hands_leave_camera(self) -> None:
        feedback = GestureFeedback(heart_hold_seconds=1.5)
        command = RobotController().from_gesture("heart")
        fallback = VoiceActivity(Reaction.IDLE, "Hold SPACE to talk")

        shown = feedback.choose("heart", command, fallback, now=10.0)
        still_shown = feedback.choose("none", RobotCommand("", Reaction.IDLE), fallback, now=11.0)
        released = feedback.choose("none", RobotCommand("", Reaction.IDLE), fallback, now=11.6)

        self.assertEqual(shown.reaction, Reaction.HEART)
        self.assertEqual(still_shown.reaction, Reaction.HEART)
        self.assertEqual(released.reaction, Reaction.IDLE)

    def test_conversation_activity_blocks_gesture_feedback(self) -> None:
        feedback = GestureFeedback(heart_hold_seconds=1.5)
        command = RobotController().from_gesture("heart")
        speaking = VoiceActivity(Reaction.SPEAKING, "Hello")

        shown = feedback.choose("heart", command, speaking, now=10.0)

        self.assertEqual(shown, speaking)
        self.assertTrue(gestures_locked(speaking))
        self.assertTrue(gestures_locked(VoiceActivity(Reaction.LISTENING), voice_busy=False))
        self.assertTrue(gestures_locked(VoiceActivity(Reaction.IDLE), voice_busy=True))
        self.assertFalse(gestures_locked(VoiceActivity(Reaction.IDLE)))
