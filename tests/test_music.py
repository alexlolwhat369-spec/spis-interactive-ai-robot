from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.music import MusicPlayer, MusicSelector, SoundEffectPlayer, Track, WebMusicController, play_track


class MusicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = MusicSelector(
            [Track("Calm", "calm", "calm.wav"), Track("Warm", "warm", "warm.wav"), Track("Celebrate", "celebration", "win.wav")]
        )

    def test_sound_effect_uses_a_separate_cached_mixer_sound(self) -> None:
        class FakeSound:
            def __init__(self, path: str) -> None:
                self.path = path
                self.volume = 0.0
                self.plays = 0

            def set_volume(self, volume: float) -> None:
                self.volume = volume

            def play(self) -> object:
                self.plays += 1
                return object()

        class FakeMixer:
            def __init__(self) -> None:
                self.created: list[FakeSound] = []

            def Sound(self, path: str) -> FakeSound:
                sound = FakeSound(path)
                self.created.append(sound)
                return sound

        class FakePygame:
            error = RuntimeError

            def __init__(self) -> None:
                self.mixer = FakeMixer()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "effect.mp3"
            path.write_bytes(b"test audio")
            pygame = FakePygame()
            player = SoundEffectPlayer(volume=0.65)
            player._pygame = pygame

            self.assertTrue(player.play(path))
            self.assertTrue(player.play(path))

        self.assertEqual(len(pygame.mixer.created), 1)
        self.assertEqual(pygame.mixer.created[0].volume, 0.65)
        self.assertEqual(pygame.mixer.created[0].plays, 2)

    def test_heart_uses_warm_track(self) -> None:
        self.assertEqual(self.selector.choose(gesture="heart").category, "warm")

    def test_missing_track_fails_cleanly_instead_of_claiming_to_play(self) -> None:
        with TemporaryDirectory() as directory:
            played = play_track(Track("Missing", "calm", "missing.wav"), Path(directory))

        self.assertFalse(played)

    def test_explicit_request_overrides_event(self) -> None:
        self.assertEqual(self.selector.choose(requested_category="calm", game_won=True).category, "calm")

    def test_tracks_in_the_same_category_rotate(self) -> None:
        selector = MusicSelector([Track("One", "energetic", "one.mp3"), Track("Two", "energetic", "two.mp3")])

        self.assertEqual(selector.choose(requested_category="energetic").title, "One")
        self.assertEqual(selector.choose(requested_category="energetic").title, "Two")
        self.assertEqual(selector.choose(requested_category="energetic").title, "One")

    def test_windows_mp3_starts_a_background_player(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "song.mp3"
            path.write_bytes(b"test audio placeholder")
            with patch("src.music.shutil.which", return_value="powershell.exe"), patch("src.music.subprocess.Popen") as popen:
                played = play_track(Track("Song", "happy", path.name), Path(directory))

        self.assertTrue(played)
        popen.assert_called_once()
        self.assertIn("-Sta", popen.call_args.args[0])

    def test_pygame_music_preserves_position_across_a_microphone_pause(self) -> None:
        class FakeMusic:
            def __init__(self) -> None:
                self.busy = False
                self.paused = False

            def load(self, path: str) -> None:
                self.path = path

            def play(self) -> None:
                self.busy = True

            def pause(self) -> None:
                self.paused = True

            def unpause(self) -> None:
                self.paused = False

            def stop(self) -> None:
                self.busy = False

            def get_busy(self) -> bool:
                return self.busy

        class FakeMixer:
            def __init__(self) -> None:
                self.music = FakeMusic()

            def get_init(self) -> bool:
                return True

        class FakePygame:
            error = RuntimeError

            def __init__(self) -> None:
                self.mixer = FakeMixer()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "song.mp3"
            path.write_bytes(b"placeholder")
            player = MusicPlayer()
            player._pygame = FakePygame()

            self.assertTrue(player.play(Track("Song", "happy", path.name), Path(directory)))
            self.assertEqual(player.current_title, "Song")
            self.assertTrue(player.pause())
            self.assertFalse(player.is_playing)
            self.assertTrue(player.resume())
            self.assertTrue(player.is_playing)


class WebMusicControllerTests(unittest.TestCase):
    """Browser-mode: the controller selects tracks + exposes stream URLs only."""

    def _controller(self, directory: str, missing: bool = False) -> WebMusicController:
        if not missing:
            for name in ("calm.wav", "warm.wav"):
                (Path(directory) / name).write_bytes(b"x")
        selector = MusicSelector([Track("Calm", "calm", "calm.wav"), Track("Warm", "warm", "warm.wav")])
        return WebMusicController(selector, Path(directory))

    def test_categories_follow_playlist_order(self) -> None:
        with TemporaryDirectory() as directory:
            controller = self._controller(directory)
            self.assertEqual(controller.categories(), ("calm", "warm"))
            self.assertEqual(controller.state()["categories"], ["calm", "warm"])

    def test_select_returns_title_and_stream_url(self) -> None:
        with TemporaryDirectory() as directory:
            controller = self._controller(directory)
            result = controller.select("warm")
            self.assertTrue(result["ok"])
            self.assertEqual(result["title"], "Warm")
            self.assertEqual(result["category"], "warm")
            self.assertEqual(result["url"], "/music/track/1")

    def test_missing_track_file_reports_reason(self) -> None:
        with TemporaryDirectory() as directory:
            controller = self._controller(directory, missing=True)
            result = controller.select("calm")
            self.assertFalse(result["ok"])
            self.assertIn("missing", result["reason"].lower())
            self.assertIsNone(result["url"])

    def test_apply_action_selects_and_clears(self) -> None:
        from src.robot_state import Action

        with TemporaryDirectory() as directory:
            controller = self._controller(directory)
            self.assertEqual(controller.apply_action(Action.PLAY_MUSIC, "warm")["title"], "Warm")
            # pause/resume are browser-only: server keeps the current selection.
            self.assertEqual(controller.apply_action(Action.PAUSE_MUSIC)["title"], "Warm")
            self.assertIsNone(controller.apply_action(Action.STOP_MUSIC)["title"])

    def test_skip_keeps_category(self) -> None:
        with TemporaryDirectory() as directory:
            controller = self._controller(directory)
            controller.select("warm")
            result = controller.skip()
            self.assertTrue(result["ok"])
            self.assertEqual(result["category"], "warm")

    def test_track_by_index_and_resolve_path(self) -> None:
        with TemporaryDirectory() as directory:
            controller = self._controller(directory)
            track = controller.track_by_index(0)
            self.assertEqual(track.title, "Calm")
            self.assertIsNotNone(controller.resolve_path(track))
            self.assertIsNone(controller.track_by_index(9))

    def test_unavailable_without_playlist(self) -> None:
        controller = WebMusicController(None, Path("."))
        self.assertFalse(controller.available)
        self.assertFalse(controller.select("calm")["ok"])
