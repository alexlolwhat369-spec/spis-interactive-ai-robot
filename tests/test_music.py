from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.music import MusicPlayer, MusicSelector, SoundEffectPlayer, Track, play_track


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
