from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.diagnostics import TurnDiagnostics
from src.music import MusicSelector, Track, WebMusicController
from src.robot_state import Reaction
from src.web_app import GESTURE_LABELS, RobotWebRuntime, create_app, gesture_catalog


class FakeRuntime:
    def __init__(self) -> None:
        self.overrides: list[tuple[Reaction, str, float]] = []
        self.cleared = 0

    def snapshot_state(self) -> dict:
        return {
            "gesture": "wave",
            "reaction": "happy",
            "confidence": 0.9,
            "camera_backend": "test camera",
        }

    def set_override(self, reaction: Reaction, subtitle: str, ttl: float) -> None:
        self.overrides.append((reaction, subtitle, ttl))

    def clear_override(self) -> None:
        self.cleared += 1

    def stream(self, kind: str):  # pragma: no cover - not exercised here
        del kind
        yield b""


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
            "mic_peak": 0.2,
            "mic_average": 0.1,
            "now_playing": None,
            "audio_b64": "",
            "audio_seconds": 0.0,
        }


def _client(project_root: Path, make_files: bool = False, sounds: dict | None = None):
    if make_files:
        (project_root / "calm.wav").write_bytes(b"RIFFfake-audio")
    selector = MusicSelector([Track("Calm", "calm", "calm.wav")])
    music = WebMusicController(selector, project_root)
    runtime = FakeRuntime()
    engine = FakeEngine()
    app = create_app(runtime, engine, music, TurnDiagnostics(), sounds)
    app.config.update(TESTING=True)
    return app.test_client(), music, runtime, engine


class GestureCatalogTests(unittest.TestCase):
    def test_catalog_matches_labels_and_excludes_wave(self) -> None:
        catalog = gesture_catalog()
        labels = [gesture["label"] for gesture in catalog]
        self.assertEqual(labels, list(GESTURE_LABELS))
        self.assertNotIn("wave", labels)
        by_label = {gesture["label"]: gesture for gesture in catalog}
        self.assertEqual(by_label["mohan"]["reply"], "Mohan!")
        self.assertEqual(by_label["mohan"]["reaction"], "mohan")
        self.assertTrue(all(gesture["reply"] for gesture in catalog))


class WebEndpointTests(unittest.TestCase):
    def test_voice_lifecycle_finishes_audio_actions(self) -> None:
        client, _, runtime, engine = _client(Path("/nonexistent"))

        self.assertEqual(client.post("/voice/listening").status_code, 204)
        response = client.post("/voice", data=b"pcm")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.post("/voice/done").status_code, 204)

        self.assertEqual(engine.begins, 1)
        self.assertEqual(engine.finishes, 1)
        self.assertEqual(engine.data, b"pcm")
        self.assertEqual(runtime.overrides[-1][0], Reaction.HAPPY)
        self.assertEqual(runtime.cleared, 1)

    def test_voice_upload_rejects_unreasonably_large_recording(self) -> None:
        client, _, _, engine = _client(Path("/nonexistent"))

        response = client.post("/voice", data=b"x" * (2 * 1024 * 1024 + 1))

        self.assertEqual(response.status_code, 413)
        self.assertFalse(hasattr(engine, "data"))

    def test_gestures_endpoint_returns_all_labels(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        body = client.get("/gestures").get_json()
        self.assertEqual(len(body["gestures"]), len(GESTURE_LABELS))
        self.assertNotIn("wave", [gesture["label"] for gesture in body["gestures"]])

    def test_state_merges_gesture_music_diagnostics_and_camera(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        body = client.get("/state").get_json()
        self.assertEqual(body["gesture"], "wave")
        self.assertEqual(body["camera_backend"], "test camera")
        self.assertIn("music", body)
        self.assertIn("diagnostics", body)
        self.assertEqual(body["music"]["categories"], ["calm"])
        self.assertEqual(body["diagnostics"]["route"], "idle")

    def test_music_state_reports_categories(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        body = client.get("/music/state").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(body["categories"], ["calm"])

    def test_play_missing_file_degrades_gracefully(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        body = client.post("/music/play", json={"category": "calm"}).get_json()
        self.assertFalse(body["ok"])
        self.assertIn("reason", body)
        self.assertIsNone(body["url"])

    def test_transport_endpoints_return_state(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        for endpoint in ("next", "stop"):
            response = client.post(f"/music/{endpoint}")
            self.assertEqual(response.status_code, 200, endpoint)
            self.assertIn("available", response.get_json())

    def test_gesture_sound_absent_by_default(self) -> None:
        client, _, _, _ = _client(Path("/nonexistent"))
        mohan = next(
            gesture for gesture in client.get("/gestures").get_json()["gestures"]
            if gesture["label"] == "mohan"
        )
        self.assertIsNone(mohan["sound"])
        self.assertEqual(client.get("/sound/mohan").status_code, 404)

    def test_gesture_sound_served_when_file_present(self) -> None:
        with TemporaryDirectory() as directory:
            whistle = Path(directory) / "mohan.mp3"
            whistle.write_bytes(b"ID3fake-whistle")
            client, _, _, _ = _client(Path(directory), sounds={"mohan": whistle})
            mohan = next(
                gesture for gesture in client.get("/gestures").get_json()["gestures"]
                if gesture["label"] == "mohan"
            )
            self.assertEqual(mohan["sound"], "/sound/mohan")
            served = client.get("/sound/mohan")
            self.assertEqual(served.status_code, 200)
            self.assertTrue(served.data.startswith(b"ID3"))
            served.close()
            self.assertEqual(client.get("/sound/unknown").status_code, 404)

    def test_play_returns_stream_url_and_track_serves_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            client, _, _, _ = _client(Path(directory), make_files=True)
            played = client.post("/music/play", json={"category": "calm"}).get_json()
            self.assertTrue(played["ok"])
            self.assertEqual(played["url"], "/music/track/0")

            audio = client.get(played["url"])
            self.assertEqual(audio.status_code, 200)
            self.assertTrue(audio.data.startswith(b"RIFF"))
            audio.close()
            self.assertEqual(client.get("/music/track/99").status_code, 404)


class CameraRuntimeTests(unittest.TestCase):
    def test_web_runtime_uses_shared_robust_camera_opener(self) -> None:
        runtime = RobotWebRuntime.__new__(RobotWebRuntime)
        runtime._camera_index = 2
        first_frame = np.full((2, 2, 3), 20, dtype=np.uint8)
        camera = object()
        with patch(
            "src.web_app.open_working_camera",
            return_value=(camera, first_frame, "DirectShow"),
        ) as opener:
            result = runtime._open_camera()

        self.assertEqual(result, (camera, first_frame, "DirectShow"))
        opener.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
