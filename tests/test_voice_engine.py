from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from src.conversation import ConversationResult
from src.diagnostics import TurnDiagnostics
from src.robot_runtime import SessionResult, TurnRoute
from src.robot_state import Action, Reaction, RobotCommand
from src.voice_engine import VoiceEngine, mic_metrics


class MicMetricsTests(unittest.TestCase):
    def test_empty_audio_is_silent(self) -> None:
        self.assertEqual(mic_metrics(b""), (0.0, 0.0))

    def test_full_scale_audio_peaks_near_one(self) -> None:
        pcm = np.array([32767, -32768, 0, 16000], dtype=np.int16).tobytes()
        peak, average = mic_metrics(pcm)
        self.assertGreater(peak, 0.9)
        self.assertGreater(average, 0.0)
        self.assertLessEqual(average, peak)


class FakeMusic:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str | None]] = []

    def apply_action(self, action: object, category: str | None = None) -> dict:
        self.actions.append((str(action), category))
        return {}

    def state(self) -> dict:
        return {"available": True, "playing": True, "paused": False, "active": True, "title": "Calm"}


class FakeSpeaker:
    def synthesize_wav_bytes(self, text: str, reaction: str) -> bytes:
        return b"WAVEDATA"


class FakeSession:
    def __init__(self, result: object) -> None:
        self._result = result

    def respond(self, message: str) -> object:
        return self._result


def _engine(session: object, music: object, diagnostics: TurnDiagnostics) -> VoiceEngine:
    engine = VoiceEngine.__new__(VoiceEngine)
    engine._lock = threading.Lock()
    engine._diagnostics = diagnostics
    engine._music = music
    engine._session = session
    engine._speaker = FakeSpeaker()
    return engine


class VoiceEngineHandleTests(unittest.TestCase):
    @patch("src.voice_engine.wav_duration_seconds", return_value=0.0)
    def test_music_action_dispatched_and_diagnostics_recorded(self, _dur: object) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Playing calm.", Reaction.HAPPY, Action.PLAY_MUSIC), "calm"),
            game_active=False,
            route=TurnRoute.MUSIC_REQUEST,
        )
        music = FakeMusic()
        diagnostics = TurnDiagnostics()
        engine = _engine(FakeSession(result), music, diagnostics)
        engine.transcribe = lambda pcm: "play some calm music"

        out = engine.handle(b"\x01\x02\x03\x04")

        self.assertEqual(out["route"], "music_request")
        self.assertEqual(out["action"], "play_music")
        self.assertEqual(music.actions, [("play_music", "calm")])
        self.assertEqual(out["now_playing"]["title"], "Calm")
        snapshot = diagnostics.current()
        self.assertEqual(snapshot.route, "music_request")
        self.assertEqual(snapshot.action, "play_music")
        self.assertEqual(snapshot.transcript_source, "final")

    @patch("src.voice_engine.wav_duration_seconds", return_value=0.0)
    def test_no_speech_records_no_input(self, _dur: object) -> None:
        diagnostics = TurnDiagnostics()
        engine = _engine(FakeSession(None), FakeMusic(), diagnostics)
        engine.transcribe = lambda pcm: ""

        out = engine.handle(b"")

        self.assertEqual(out["route"], "no_input")
        self.assertEqual(out["action"], "none")
        self.assertEqual(diagnostics.current().route, "no_input")

    @patch("src.voice_engine.wav_duration_seconds", return_value=0.0)
    def test_conversation_turn_does_not_touch_music(self, _dur: object) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Hello there.", Reaction.HAPPY, Action.NONE)),
            game_active=False,
            route=TurnRoute.CONVERSATION,
        )
        music = FakeMusic()
        engine = _engine(FakeSession(result), music, TurnDiagnostics())
        engine.transcribe = lambda pcm: "hello"

        out = engine.handle(b"\x10\x00")

        self.assertEqual(out["route"], "conversation")
        self.assertEqual(music.actions, [])


if __name__ == "__main__":
    unittest.main()
