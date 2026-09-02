from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.conversation import ConversationResult
from src.music import MusicSelector, Track
from src.robot_runtime import SessionResult, TurnRoute
from src.robot_state import Action, Reaction, RobotCommand
from src.voice_engine import VoiceEngine


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


class FakeSession:
    expects_game_answer = False
    expects_music_category = False

    def __init__(self, result: SessionResult) -> None:
        self.result = result

    def respond(self, message: str) -> SessionResult:
        self.message = message
        return self.result


class VoiceEngineTests(unittest.TestCase):
    def build_engine(self, directory: str, result: SessionResult, player: FakePlayer) -> VoiceEngine:
        track_path = Path(directory) / "happy.wav"
        track_path.write_bytes(b"audio")
        selector = MusicSelector([Track("Happy Track", "happy", str(track_path))])
        engine = VoiceEngine(
            Path(directory),
            Path(directory) / "voice.onnx",
            session=FakeSession(result),
            speaker=FakeSpeaker(),
            recognizer_model=object(),
            recognizer_type=object(),
            music=selector,
            music_player=player,
            project_root=Path(directory),
        )
        engine.transcribe = lambda data, phrases=None: ("happy", "guided", bool(phrases))  # type: ignore[method-assign]
        return engine

    def test_music_starts_after_browser_finishes_speaking(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("I will play it.", Reaction.HAPPY, Action.PLAY_MUSIC), "happy"),
            False,
            TurnRoute.MUSIC_CATEGORY,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            engine = self.build_engine(directory, result, player)

            response = engine.handle(b"audio")

            self.assertEqual(response["reply"], "Playing Happy Track.")
            self.assertEqual(response["action"], "play_music")
            self.assertEqual(player.played, [])
            self.assertTrue(engine.finish_turn())
            self.assertEqual(player.played, ["Happy Track"])

    def test_missing_music_file_never_claims_that_playback_started(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("I will play it.", Reaction.HAPPY, Action.PLAY_MUSIC), "happy"),
            False,
            TurnRoute.MUSIC_CATEGORY,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            engine = self.build_engine(directory, result, player)
            (Path(directory) / "happy.wav").unlink()

            response = engine.handle(b"audio")

            self.assertEqual(response["reaction"], "confused")
            self.assertIn("not installed", response["reply"])
            self.assertFalse(engine.finish_turn())
            self.assertEqual(player.played, [])

    def test_unrelated_reply_resumes_music_paused_for_microphone(self) -> None:
        result = SessionResult(
            ConversationResult(RobotCommand("Hello.", Reaction.HAPPY)),
            False,
            TurnRoute.CONVERSATION,
        )
        with TemporaryDirectory() as directory:
            player = FakePlayer()
            player.active = True
            engine = self.build_engine(directory, result, player)

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
            player = FakePlayer()
            engine = self.build_engine(directory, result, player)
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
