from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.conversation import ConversationResult
from src.diagnostics import TurnDiagnostics
from src.music import MusicSelector, Track
from src.robot_runtime import SessionResult, TurnRoute
from src.robot_state import Action, Reaction, RobotCommand
from src.voice_engine import VoiceEngine, mic_metrics


class FakeSpeaker:
    def synthesize_wav_bytes(self, text: str, reaction: str) -> bytes:
        self.last = (text, reaction)
        return b""


class FakePlayer:
    def __init__(self) -> None:
        self.active = False
        self.paused = False
        self.played: list[str] = []
        self.resumes = 0
        self.stops = 0

    @property
    def is_active(self) -> bool:
        return self.active

    def pause(self) -> bool:
        if not self.active or self.paused:
            return False
        self.paused = True
        return True

    def resume(self) -> bool:
        if not self.paused:
            return False
        self.paused = False
        self.resumes += 1
        return True

    def stop(self) -> bool:
        changed = self.active
        self.active = False
        self.paused = False
        self.stops += 1
        return changed

    def play(self, track: Track, root: Path) -> bool:
        del root
        self.active = True
        self.paused = False
        self.played.append(track.title)
        return True


class FakeWebMusic:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str | None]] = []
        self.title: str | None = "Calm"

    def apply_action(self, action: object, category: str | None = None) -> dict:
        self.actions.append((str(action), category))
        if str(action) == "stop_music":
            self.title = None
        return {"ok": True, **self.state()}

    def state(self) -> dict:
        return {
            "available": True,
            "categories": ["calm"],
            "title": self.title,
            "category": "calm" if self.title else None,
            "url": "/music/track/0" if self.title else None,
        }


class FakeSession:
    expects_game_answer = False
    expects_music_category = False

    def __init__(self, result: SessionResult | None) -> None:
        self.result = result

    def respond(self, message: str) -> SessionResult:
        self.message = message
        assert self.result is not None
        return self.result


def build_engine(
    directory: str,
    result: SessionResult | None,
    *,
    music: object,
    player: FakePlayer | None = None,
    diagnostics: TurnDiagnostics | None = None,
) -> VoiceEngine:
    engine = VoiceEngine(
        Path(directory),
        Path(directory) / "voice.onnx",
        session=FakeSession(result),
        speaker=FakeSpeaker(),
        recognizer_model=object(),
        recognizer_type=object(),
        music=music,
        music_player=player,
        project_root=Path(directory),
        diagnostics=diagnostics,
    )
    engine.transcribe = lambda data, phrases=None: ("hello", "guided" if phrases else "final", bool(phrases))  # type: ignore[method-assign]
    return engine


class MicMetricsTests(unittest.TestCase):
    def test_empty_audio_is_silent(self) -> None:
        self.assertEqual(mic_metrics(b""), (0.0, 0.0))

    def test_full_scale_audio_peaks_near_one(self) -> None:
        pcm = np.array([32767, -32768, 0, 16000], dtype=np.int16).tobytes()
        peak, average = mic_metrics(pcm)
        self.assertGreater(peak, 0.9)
        self.assertGreater(average, 0.0)
        self.assertLessEqual(average, peak)


class BrowserMusicVoiceTests(unittest.TestCase):
    def test_music_action_dispatched_and_diagnostics_recorded(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Playing calm.", Reaction.HAPPY, Action.PLAY_MUSIC), "calm"),
            game_active=False,
            route=TurnRoute.MUSIC_REQUEST,
        )
        with TemporaryDirectory() as directory:
            music = FakeWebMusic()
            diagnostics = TurnDiagnostics()
            engine = build_engine(directory, result, music=music, diagnostics=diagnostics)
            engine.transcribe = lambda pcm, phrases=None: ("play some calm music", "final", False)  # type: ignore[method-assign]

            out = engine.handle(b"\x01\x02\x03\x04")

        self.assertEqual(out["route"], "music_request")
        self.assertEqual(out["action"], "play_music")
        self.assertEqual(music.actions, [("play_music", "calm")])
        self.assertEqual(out["now_playing"]["title"], "Calm")
        snapshot = diagnostics.current()
        self.assertEqual(snapshot.route, "music_request")
        self.assertEqual(snapshot.action, "play_music")
        self.assertEqual(snapshot.transcript_source, "final")

    def test_no_speech_records_no_input(self) -> None:
        with TemporaryDirectory() as directory:
            diagnostics = TurnDiagnostics()
            engine = build_engine(directory, None, music=FakeWebMusic(), diagnostics=diagnostics)
            engine.transcribe = lambda pcm, phrases=None: ("", "none", False)  # type: ignore[method-assign]

            out = engine.handle(b"")

        self.assertEqual(out["route"], "no_input")
        self.assertEqual(out["action"], "none")
        self.assertEqual(diagnostics.current().route, "no_input")

    def test_conversation_turn_does_not_touch_music(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Hello there.", Reaction.HAPPY, Action.NONE)),
            game_active=False,
            route=TurnRoute.CONVERSATION,
        )
        with TemporaryDirectory() as directory:
            music = FakeWebMusic()
            engine = build_engine(directory, result, music=music)
            out = engine.handle(b"\x10\x00")

        self.assertEqual(out["route"], "conversation")
        self.assertEqual(music.actions, [])


class LocalMusicVoiceTests(unittest.TestCase):
    def local_engine(self, directory: str, result: SessionResult, player: FakePlayer) -> VoiceEngine:
        track_path = Path(directory) / "happy.wav"
        track_path.write_bytes(b"audio")
        selector = MusicSelector([Track("Happy Track", "happy", str(track_path))])
        return build_engine(directory, result, music=selector, player=player)

    def test_music_starts_after_browser_finishes_speaking(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("I will play it.", Reaction.HAPPY, Action.PLAY_MUSIC), "happy"),
            False,
            TurnRoute.MUSIC_CATEGORY,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            engine = self.local_engine(directory, result, player)
            response = engine.handle(b"audio")

            self.assertEqual(response["reply"], "Playing Happy Track.")
            self.assertEqual(player.played, [])
            self.assertTrue(engine.finish_turn())
            self.assertEqual(player.played, ["Happy Track"])

    def test_missing_music_file_never_claims_playback_started(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("I will play it.", Reaction.HAPPY, Action.PLAY_MUSIC), "happy"),
            False,
            TurnRoute.MUSIC_CATEGORY,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            engine = self.local_engine(directory, result, player)
            (Path(directory) / "happy.wav").unlink()

            response = engine.handle(b"audio")

            self.assertEqual(response["reaction"], "confused")
            self.assertIn("not installed", response["reply"])
            self.assertFalse(engine.finish_turn())

    def test_unrelated_reply_resumes_music_paused_for_microphone(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Hello.", Reaction.HAPPY)),
            False,
            TurnRoute.CONVERSATION,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            player.active = True
            engine = self.local_engine(directory, result, player)

            engine.begin_turn()
            engine.handle(b"audio")
            engine.finish_turn()

            self.assertEqual(player.resumes, 1)

    def test_game_turn_passes_guided_answer_vocabulary(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Next question?", Reaction.CURIOUS)),
            True,
            TurnRoute.GAME_ANSWER,
        )
        with TemporaryDirectory() as directory:
            engine = self.local_engine(directory, result, FakePlayer())
            engine._session.expects_game_answer = True
            captured: dict[str, object] = {}

            def transcribe(data: bytes, phrases: tuple[str, ...] | None = None) -> tuple[str, str, bool]:
                del data
                captured["phrases"] = phrases
                return "probably not", "guided", True

            engine.transcribe = transcribe  # type: ignore[method-assign]
            response = engine.handle(b"audio")

        self.assertIn("probably not", captured["phrases"])
        self.assertTrue(response["guided_used"])


if __name__ == "__main__":
    unittest.main()
