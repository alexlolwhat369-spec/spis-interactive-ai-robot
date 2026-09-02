from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.robot_state import Reaction
from src.web_app import RobotWebRuntime, create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.overrides: list[tuple[Reaction, str, float]] = []
        self.cleared = 0

    def set_override(self, reaction: Reaction, subtitle: str, ttl: float) -> None:
        self.overrides.append((reaction, subtitle, ttl))

    def clear_override(self) -> None:
        self.cleared += 1

    def snapshot_state(self) -> dict:
        return {"gesture": "none", "reaction": "idle", "confidence": 0.0}

    def stream(self, kind: str):
        del kind
        return iter(())


class FakeEngine:
    available = True
    reason = ""

    def __init__(self) -> None:
        self.begins = 0
        self.finishes = 0

    def begin_turn(self) -> None:
        self.begins += 1

    def finish_turn(self) -> bool:
        self.finishes += 1
        return True

    def handle(self, data: bytes) -> dict:
        self.data = data
        return {
            "heard": "hello",
            "reply": "Hello!",
            "reaction": "happy",
            "route": "conversation",
            "action": "none",
            "provider_error": None,
            "transcript_source": "final",
            "guided_used": False,
            "audio_b64": "",
            "audio_seconds": 0.0,
        }


class WebAppTests(unittest.TestCase):
    def test_voice_lifecycle_pauses_then_finishes_audio_actions(self) -> None:
        runtime = FakeRuntime()
        engine = FakeEngine()
        client = create_app(runtime, engine).test_client()

        self.assertEqual(client.post("/voice/listening").status_code, 204)
        response = client.post("/voice", data=b"pcm")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.post("/voice/done").status_code, 204)

        self.assertEqual(engine.begins, 1)
        self.assertEqual(engine.finishes, 1)
        self.assertEqual(engine.data, b"pcm")
        self.assertEqual(runtime.overrides[-1][0], Reaction.HAPPY)
        self.assertEqual(runtime.cleared, 1)

    def test_voice_upload_rejects_an_unreasonably_large_recording(self) -> None:
        runtime = FakeRuntime()
        engine = FakeEngine()
        client = create_app(runtime, engine).test_client()

        response = client.post("/voice", data=b"x" * (2 * 1024 * 1024 + 1))

        self.assertEqual(response.status_code, 413)
        self.assertFalse(hasattr(engine, "data"))

    def test_web_runtime_uses_the_shared_robust_camera_opener(self) -> None:
        runtime = RobotWebRuntime.__new__(RobotWebRuntime)
        runtime._camera_index = 2
        first_frame = np.full((2, 2, 3), 20, dtype=np.uint8)
        camera = object()
        with patch("src.web_app.open_working_camera", return_value=(camera, first_frame, "DirectShow")) as opener:
            result = runtime._open_camera()

        self.assertEqual(result, (camera, first_frame, "DirectShow"))
        opener.assert_called_once_with(2)

    def test_mohan_sound_triggers_once_until_the_gesture_changes(self) -> None:
        class Player:
            def __init__(self) -> None:
                self.calls = 0

            def play(self, path: Path) -> bool:
                del path
                self.calls += 1
                return True

        runtime = RobotWebRuntime.__new__(RobotWebRuntime)
        runtime._sound_player = Player()
        runtime._mohan_sound = Path("mohan.mp3")
        runtime._previous_sound_label = "none"

        runtime._update_gesture_sound("mohan")
        runtime._update_gesture_sound("mohan")
        runtime._update_gesture_sound("none")
        runtime._update_gesture_sound("mohan")

        self.assertEqual(runtime._sound_player.calls, 2)


if __name__ == "__main__":
    unittest.main()
