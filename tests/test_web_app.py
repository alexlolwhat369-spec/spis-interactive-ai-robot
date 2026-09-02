from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.diagnostics import TurnDiagnostics
from src.music import MusicSelector, Track, WebMusicController
from src.web_app import GESTURE_LABELS, create_app, gesture_catalog


class FakeRuntime:
    def snapshot_state(self) -> dict:
        return {"gesture": "wave", "reaction": "happy", "confidence": 0.9}

    def set_override(self, *args: object, **kwargs: object) -> None:
        pass

    def clear_override(self) -> None:
        pass

    def stream(self, kind: str):  # pragma: no cover - not exercised here
        yield b""


class FakeEngine:
    available = True
    reason = ""

    def handle(self, data: bytes) -> dict:
        return {"reaction": "happy", "reply": "hello", "audio_seconds": 0.0}


def _client(project_root: Path, make_files: bool = False, sounds: dict | None = None):
    if make_files:
        (Path(project_root) / "calm.wav").write_bytes(b"RIFFfake-audio")
    selector = MusicSelector([Track("Calm", "calm", "calm.wav")])
    music = WebMusicController(selector, Path(project_root))
    app = create_app(FakeRuntime(), FakeEngine(), music, TurnDiagnostics(), sounds)
    app.config.update(TESTING=True)
    return app.test_client(), music


class GestureCatalogTests(unittest.TestCase):
    def test_catalog_matches_labels_and_excludes_wave(self) -> None:
        catalog = gesture_catalog()
        labels = [g["label"] for g in catalog]
        self.assertEqual(labels, list(GESTURE_LABELS))
        self.assertNotIn("wave", labels)
        by_label = {g["label"]: g for g in catalog}
        self.assertEqual(by_label["mohan"]["reply"], "Mohan!")
        self.assertEqual(by_label["mohan"]["reaction"], "mohan")
        self.assertTrue(all(g["reply"] for g in catalog))


class WebEndpointTests(unittest.TestCase):
    def test_gestures_endpoint_returns_all_labels(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        body = client.get("/gestures").get_json()
        self.assertEqual(len(body["gestures"]), len(GESTURE_LABELS))
        self.assertNotIn("wave", [g["label"] for g in body["gestures"]])

    def test_state_merges_gesture_music_and_diagnostics(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        body = client.get("/state").get_json()
        self.assertEqual(body["gesture"], "wave")
        self.assertIn("music", body)
        self.assertIn("diagnostics", body)
        self.assertEqual(body["music"]["categories"], ["calm"])
        self.assertEqual(body["diagnostics"]["route"], "idle")

    def test_music_state_reports_categories(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        body = client.get("/music/state").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(body["categories"], ["calm"])

    def test_play_missing_file_degrades_gracefully(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        res = client.post("/music/play", json={"category": "calm"})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertIn("reason", body)
        self.assertIsNone(body["url"])

    def test_transport_endpoints_return_state(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        for endpoint in ("next", "stop"):
            res = client.post(f"/music/{endpoint}")
            self.assertEqual(res.status_code, 200, endpoint)
            self.assertIn("available", res.get_json())

    def test_gesture_sound_absent_by_default(self) -> None:
        client, _ = _client(Path("/nonexistent"))
        mohan = next(g for g in client.get("/gestures").get_json()["gestures"] if g["label"] == "mohan")
        self.assertIsNone(mohan["sound"])
        self.assertEqual(client.get("/sound/mohan").status_code, 404)

    def test_gesture_sound_served_when_file_present(self) -> None:
        with TemporaryDirectory() as directory:
            whistle = Path(directory) / "mohan.mp3"
            whistle.write_bytes(b"ID3fake-whistle")
            client, _ = _client(Path(directory), sounds={"mohan": whistle})
            mohan = next(g for g in client.get("/gestures").get_json()["gestures"] if g["label"] == "mohan")
            self.assertEqual(mohan["sound"], "/sound/mohan")
            served = client.get("/sound/mohan")
            self.assertEqual(served.status_code, 200)
            self.assertTrue(served.data.startswith(b"ID3"))
            self.assertEqual(client.get("/sound/unknown").status_code, 404)

    def test_play_returns_stream_url_and_track_serves_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            client, _ = _client(Path(directory), make_files=True)
            played = client.post("/music/play", json={"category": "calm"}).get_json()
            self.assertTrue(played["ok"])
            self.assertEqual(played["url"], "/music/track/0")

            audio = client.get(played["url"])
            self.assertEqual(audio.status_code, 200)
            self.assertTrue(audio.data.startswith(b"RIFF"))
            self.assertEqual(client.get("/music/track/99").status_code, 404)


if __name__ == "__main__":
    unittest.main()
