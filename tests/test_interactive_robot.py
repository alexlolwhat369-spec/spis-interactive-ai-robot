from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.interactive_robot import GestureFeedback, VoiceActivity, VoiceState, VoiceWorker, gestures_locked
from src.conversation import ConversationResult
from src.music import MusicSelector, Track
from src.robot_runtime import SessionResult
from src.robot_state import Action, Reaction, RobotCommand, RobotController


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

    def test_mohan_portrait_stays_visible_briefly(self) -> None:
        feedback = GestureFeedback(mohan_hold_seconds=2.0)
        command = RobotController().from_gesture("mohan")
        fallback = VoiceActivity(Reaction.IDLE)

        shown = feedback.choose("mohan", command, fallback, now=5.0)
        held = feedback.choose("none", RobotCommand("", Reaction.IDLE), fallback, now=6.9)
        released = feedback.choose("none", RobotCommand("", Reaction.IDLE), fallback, now=7.1)

        self.assertEqual(shown.reaction, Reaction.MOHAN)
        self.assertEqual(held.reaction, Reaction.MOHAN)
        self.assertEqual(released.reaction, Reaction.IDLE)

    def test_missing_music_file_produces_truthful_spoken_feedback(self) -> None:
        class RecordingSpeaker:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def speak(self, text: str, reaction: str | None = None) -> bool:
                self.messages.append(text)
                return True

        speaker = RecordingSpeaker()
        selector = MusicSelector([Track("Missing", "calm", "assets/music/missing.wav")])
        with TemporaryDirectory() as directory:
            worker = VoiceWorker(object(), speaker, object(), 1.0, selector, Path(directory))
            worker._handle_music("calm")

        self.assertIn("no playable music file", speaker.messages[-1])

    def test_tts_exception_keeps_the_visible_subtitle_alive(self) -> None:
        class BrokenSpeaker:
            def speak(self, text: str, reaction: str | None = None) -> bool:
                raise RuntimeError("speaker disconnected")

        worker = VoiceWorker(object(), BrokenSpeaker(), object(), 1.0)

        worker._say("Visible reply", Reaction.HAPPY)

        self.assertEqual(worker.state.current(), VoiceActivity(Reaction.HAPPY, "Visible reply", speaking=True))

    def test_emotional_speech_locks_gestures(self) -> None:
        activity = VoiceActivity(Reaction.CURIOUS, "Really?", speaking=True)

        self.assertTrue(gestures_locked(activity))

    def test_unrelated_conversation_resumes_paused_music(self) -> None:
        class Listener:
            def listen_once(self, *args: object, **kwargs: object) -> str:
                return "Tell me a joke"

        class Speaker:
            def speak(self, *args: object, **kwargs: object) -> bool:
                return True

        class Session:
            expects_game_answer = False
            expects_music_category = False

            def respond(self, message: str) -> SessionResult:
                del message
                return SessionResult(ConversationResult(RobotCommand("A joke.", Reaction.HAPPY)), False)

        class Player:
            def __init__(self) -> None:
                self.resumed = False

            def resume(self) -> bool:
                self.resumed = True
                return True

        player = Player()
        worker = VoiceWorker(Listener(), Speaker(), Session(), 1.0, music_player=player)
        worker._music_paused_for_turn = True

        worker._listen_and_respond()

        self.assertTrue(player.resumed)

    def test_stop_music_control_reaches_the_owned_player(self) -> None:
        class Speaker:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def speak(self, text: str, *args: object) -> bool:
                self.messages.append(text)
                return True

        class Player:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> bool:
                self.stopped = True
                return True

        speaker = Speaker()
        player = Player()
        worker = VoiceWorker(object(), speaker, object(), 1.0, music_player=player)

        worker._handle_music_control(Action.STOP_MUSIC)

        self.assertTrue(player.stopped)
        self.assertEqual(speaker.messages[-1], "Music stopped.")

    def test_game_answer_turn_uses_the_guided_speech_vocabulary(self) -> None:
        class RecordingListener:
            def __init__(self) -> None:
                self.phrases: tuple[str, ...] | None = None

            def listen_once(self, max_seconds: float, stop_event: object, phrases: tuple[str, ...] | None = None) -> str:
                self.phrases = phrases
                return ""

        class GameSession:
            expects_game_answer = True

        listener = RecordingListener()
        worker = VoiceWorker(listener, object(), GameSession(), 1.0)

        worker._listen_and_respond()

        self.assertIsNotNone(listener.phrases)
        self.assertIn("probably not", listener.phrases)

    def test_semantic_game_turn_uses_dual_pass_guidance(self) -> None:
        class RecordingListener:
            def __init__(self) -> None:
                self.phrases: tuple[str, ...] | None = ("not called",)

            def listen_once(self, max_seconds: float, stop_event: object, phrases: tuple[str, ...] | None = None) -> str:
                self.phrases = phrases
                return ""

        class HybridGameSession:
            expects_game_answer = True
            supports_semantic_game_input = True

        listener = RecordingListener()
        worker = VoiceWorker(listener, object(), HybridGameSession(), 1.0)

        worker._listen_and_respond()

        self.assertIsNotNone(listener.phrases)
        self.assertIn("probably not", listener.phrases)
