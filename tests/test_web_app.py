from __future__ import annotations

import unittest
import hashlib
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
    def test_frontend_build_is_served_with_assets(self) -> None:
        with TemporaryDirectory() as directory:
            dist = Path(directory)
            (dist / "assets").mkdir()
            (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
            (dist / "assets" / "app.js").write_text("console.log('robot');", encoding="utf-8")
            with patch("src.web_app.DIST_DIR", dist):
                client, _, _, _ = _client(dist)
                with client.get("/") as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(b'id="root"', response.data)
                with client.get("/assets/app.js") as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(b"robot", response.data)
                self.assertEqual(client.get("/assets/missing.js").status_code, 404)

    def test_missing_frontend_build_returns_actionable_error(self) -> None:
        with TemporaryDirectory() as directory, patch("src.web_app.DIST_DIR", Path(directory)):
            client, _, _, _ = _client(Path(directory))
            response = client.get("/")
            self.assertEqual(response.status_code, 503)
            self.assertIn(b"npm run build", response.data)

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
            heart = Path(directory) / "heart.wav"
            peace = Path(directory) / "peace.wav"
            ok_sound = Path(directory) / "ok.mp3"
            thumbs_up = Path(directory) / "thumbs-up.mp3"
            angry = Path(directory) / "angry.mp3"
            whistle.write_bytes(b"ID3fake-whistle")
            heart.write_bytes(b"RIFFfake-heart")
            peace.write_bytes(b"RIFFfake-peace")
            ok_sound.write_bytes(b"ID3fake-ok")
            thumbs_up.write_bytes(b"ID3fake-thumbs-up")
            angry.write_bytes(b"ID3fake-angry")
            sounds = {
                "peace": peace,
                "thumbs_up": thumbs_up,
                "heart": heart,
                "ok": ok_sound,
                "middle_finger": angry,
                "mohan": whistle,
            }
            client, _, _, _ = _client(Path(directory), sounds=sounds)
            gestures = {
                gesture["label"]: gesture
                for gesture in client.get("/gestures").get_json()["gestures"]
            }
            for label in sounds:
                version = hashlib.sha256(sounds[label].read_bytes()).hexdigest()[:16]
                self.assertEqual(gestures[label]["sound"], f"/sound/{label}?v={version}")
                self.assertEqual(gestures[label]["sound_name"], sounds[label].name)
                served = client.get(gestures[label]["sound"])
                self.assertEqual(served.status_code, 200)
                expected_header = b"RIFF" if label in {"heart", "peace"} else b"ID3"
                self.assertTrue(served.data.startswith(expected_header))
                self.assertEqual(served.data, sounds[label].read_bytes())
                self.assertEqual(served.mimetype, "audio/wav" if expected_header == b"RIFF" else "audio/mpeg")
                self.assertTrue(served.cache_control.no_store)
                served.close()
            self.assertEqual(client.get("/sound/unknown").status_code, 404)

    def test_replacing_sound_changes_url_even_at_same_size(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "peace.wav"
            path.write_bytes(b"RIFFold-audio")
            client, _, _, _ = _client(Path(directory), sounds={"peace": path})
            def sound_url():
                response = client.get("/gestures")
                self.assertTrue(response.cache_control.no_store)
                return next(g["sound"] for g in response.get_json()["gestures"] if g["label"] == "peace")
            old_url = sound_url()
            path.write_bytes(b"RIFFnew-audio")
            new_url = sound_url()
            self.assertNotEqual(old_url, new_url)
            with client.get(new_url) as response:
                self.assertEqual(response.data, b"RIFFnew-audio")

    def test_selected_effects_are_packaged_and_served_exactly(self) -> None:
        from src.web_app import (
            DEFAULT_HEART_SOUND, DEFAULT_PEACE_SOUND, DEFAULT_OK_SOUND,
            DEFAULT_ANGRY_SOUND, DEFAULT_THUMBS_UP_SOUND, DEFAULT_MOHAN_SOUND,
        )
        sounds = dict(heart=DEFAULT_HEART_SOUND, peace=DEFAULT_PEACE_SOUND,
                      ok=DEFAULT_OK_SOUND, middle_finger=DEFAULT_ANGRY_SOUND,
                      thumbs_up=DEFAULT_THUMBS_UP_SOUND, mohan=DEFAULT_MOHAN_SOUND)
        client, _, _, _ = _client(Path.cwd(), sounds=sounds)
        for label, path in sounds.items():
            with self.subTest(label=label):
                self.assertTrue(path.is_file(), f"Selected sound missing: {path.name}")
                with client.get(f"/sound/{label}") as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.data, path.read_bytes())

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

    def test_gesture_events_survive_browser_polling_gaps(self) -> None:
        runtime = RobotWebRuntime.__new__(RobotWebRuntime)
        runtime._gesture_event_id = 0
        runtime._gesture_event_label = "none"

        self.assertEqual(runtime._observe_gesture_event("heart"), 1)
        self.assertEqual(runtime._observe_gesture_event("heart"), 1)
        self.assertEqual(runtime._observe_gesture_event("unknown"), 1)
        self.assertEqual(runtime._observe_gesture_event("heart"), 1)
        self.assertEqual(runtime._observe_gesture_event("none"), 1)
        self.assertEqual(runtime._observe_gesture_event("heart"), 2)
        self.assertEqual(runtime._observe_gesture_event("ok"), 3)

    def test_web_reaction_uses_the_sound_duration(self) -> None:
        runtime = RobotWebRuntime.__new__(RobotWebRuntime)
        runtime._gesture_sound_durations = {"heart": 2.0, "ok": 1.0}
        runtime._effect_until = 0.0
        runtime._effect_activity = None
        runtime._effect_label = None
        runtime._effect_completed_label = None
        runtime._effect_consumed_event = 0

        started = runtime._timed_gesture_activity(
            "heart", 1, Reaction.HEART, "Love", now=10.0
        )
        held_after_release = runtime._timed_gesture_activity(
            "none", 1, Reaction.IDLE, "", now=11.9
        )
        ended_while_held = runtime._timed_gesture_activity(
            "heart", 1, Reaction.HEART, "Love", now=12.1
        )
        still_ended = runtime._timed_gesture_activity(
            "heart", 1, Reaction.HEART, "Love", now=12.5
        )
        replacement = runtime._timed_gesture_activity(
            "ok", 2, Reaction.OK, "Perfect", now=12.6
        )

        self.assertEqual(started, (Reaction.HEART, "Love"))
        self.assertEqual(held_after_release, (Reaction.HEART, "Love"))
        self.assertEqual(ended_while_held, (Reaction.IDLE, ""))
        self.assertEqual(still_ended, (Reaction.IDLE, ""))
        self.assertEqual(replacement, (Reaction.OK, "Perfect"))

    def test_browser_keeps_only_one_sound_effect_active(self) -> None:
        web = Path(__file__).parents[1] / "web" / "src"
        hook = (web / "hooks" / "useRobotState.ts").read_text(encoding="utf-8")
        voice = (web / "hooks" / "useVoice.ts").read_text(encoding="utf-8")

        self.assertIn("const playGestureSound = useCallback", hook)
        self.assertIn("const audio = new Audio(url)", hook)
        self.assertIn("stopGestureSound();", hook)
        self.assertIn("sfxUrl.current === url", hook)
        self.assertIn("s.gesture_event !== lastGestureEvent.current", hook)
        self.assertIn("stopGestureSound();", voice)
        self.assertNotIn("MAX_CONCURRENT_SFX", hook)
        self.assertNotIn("sfxAudio.src =", hook)


if __name__ == "__main__":
    unittest.main()
