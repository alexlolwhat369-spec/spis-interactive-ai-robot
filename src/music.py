"""Choose local music from explicit requests and robot events only."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Track:
    title: str
    category: str
    path: str


class MusicSelector:
    """Deterministic selector. It never receives face or identity information."""

    def __init__(self, tracks: Iterable[Track]) -> None:
        self._tracks = tuple(tracks)
        self._next_index: dict[str, int] = {}
        if not self._tracks:
            raise ValueError("At least one track is required.")

    @classmethod
    def from_file(cls, path: Path) -> "MusicSelector":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(Track(**track) for track in data["tracks"])

    def choose(
        self,
        *,
        requested_category: str | None = None,
        gesture: str | None = None,
        game_won: bool = False,
    ) -> Track:
        if requested_category:
            category = requested_category
        elif game_won or gesture == "thumbs_up":
            category = "celebration"
        elif gesture == "heart":
            category = "warm"
        else:
            category = "calm"
        matching = tuple(track for track in self._tracks if track.category == category)
        if matching:
            index = self._next_index.get(category, 0) % len(matching)
            self._next_index[category] = index + 1
            return matching[index]
        return self._tracks[0]


class SoundEffectPlayer:
    """Play short local effects on a mixer channel without stopping background music."""

    def __init__(self, volume: float = 0.8) -> None:
        self.volume = max(0.0, min(1.0, volume))
        self._pygame: object | None = None
        self._sounds: dict[Path, object] = {}
        self._lock = threading.Lock()

    def _load_pygame(self) -> object | None:
        if self._pygame is not None:
            return self._pygame
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._pygame = pygame
        except Exception:
            self._pygame = None
        return self._pygame

    def play(self, path: Path) -> bool:
        path = path.resolve()
        if not path.is_file():
            return False
        pygame = self._load_pygame()
        if pygame is None:
            return False
        try:
            with self._lock:
                sound = self._sounds.get(path)
                if sound is None:
                    sound = pygame.mixer.Sound(str(path))
                    sound.set_volume(self.volume)
                    self._sounds[path] = sound
                channel = sound.play()
            return channel is not None
        except (RuntimeError, OSError, pygame.error):
            return False


class MusicPlayer:
    """Own the active local player so speech commands can stop it reliably."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._windows_wav_active = False
        self._pygame: object | None = None
        self._backend: str | None = None
        self._current_track: Track | None = None
        self._project_root: Path | None = None
        self._paused = False
        self._lock = threading.Lock()

    def _load_pygame(self) -> object | None:
        if self._pygame is not None:
            return self._pygame
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._pygame = pygame
        except Exception:
            self._pygame = None
        return self._pygame

    @property
    def is_playing(self) -> bool:
        with self._lock:
            if self._backend == "pygame" and self._pygame is not None:
                return bool(self._pygame.mixer.music.get_busy()) and not self._paused
            process_active = self._process is not None and self._process.poll() is None
            return process_active or self._windows_wav_active

    @property
    def is_active(self) -> bool:
        with self._lock:
            self._refresh_finished_locked()
            return self._current_track is not None

    @property
    def current_title(self) -> str | None:
        with self._lock:
            self._refresh_finished_locked()
            return self._current_track.title if self._current_track is not None else None

    def _refresh_finished_locked(self) -> None:
        if self._current_track is None or self._paused:
            return
        finished = False
        if self._backend == "pygame" and self._pygame is not None:
            finished = not bool(self._pygame.mixer.music.get_busy())
        elif self._backend == "process" and self._process is not None:
            finished = self._process.poll() is not None
        if finished:
            self._current_track = None
            self._project_root = None
            self._backend = None

    def pause(self) -> bool:
        """Pause music for a microphone turn, preserving the track when possible."""
        with self._lock:
            if self._current_track is None or self._paused:
                return False
            if self._backend == "pygame" and self._pygame is not None:
                self._pygame.mixer.music.pause()
            else:
                self._stop_backend_locked()
            self._paused = True
            return True

    def resume(self) -> bool:
        """Resume a paused track; fallback players restart it from the beginning."""
        with self._lock:
            if not self._paused or self._current_track is None:
                return False
            if self._backend == "pygame" and self._pygame is not None:
                self._pygame.mixer.music.unpause()
                self._paused = False
                return True
            track = self._current_track
            project_root = self._project_root
            self._paused = False
        return bool(project_root is not None and self.play(track, project_root))

    def stop(self) -> bool:
        with self._lock:
            stopped = self._stop_backend_locked() or self._current_track is not None
            self._current_track = None
            self._project_root = None
            self._paused = False
            self._backend = None
            return stopped

    def _stop_backend_locked(self) -> bool:
        stopped = False
        if self._backend == "pygame" and self._pygame is not None:
            self._pygame.mixer.music.stop()
            stopped = True
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
            stopped = True
        if self._windows_wav_active:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
            self._windows_wav_active = False
            stopped = True
        return stopped

    def play(self, track: Track, project_root: Path) -> bool:
        path = Path(track.path)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            return False
        self.stop()
        pygame = self._load_pygame()
        played = self._play_pygame(path, pygame) if pygame is not None else False
        if not played:
            played = self._play_windows(path) if os.name == "nt" else self._play_linux(path)
        if played:
            with self._lock:
                self._current_track = track
                self._project_root = project_root
                self._paused = False
        return played

    def _play_pygame(self, path: Path, pygame: object) -> bool:
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
        except (RuntimeError, OSError, pygame.error):
            return False
        with self._lock:
            self._backend = "pygame"
        return True

    def _play_windows(self, path: Path) -> bool:
        if path.suffix.lower() == ".wav":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            with self._lock:
                self._windows_wav_active = True
            return True
        if path.suffix.lower() != ".mp3" or not shutil.which("powershell"):
            return False
        encoded = base64.b64encode(str(path.resolve()).encode("utf-8")).decode("ascii")
        script = (
            "Add-Type -AssemblyName PresentationCore; "
            f"$path = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')); "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            "$player.Open([Uri]$path); "
            "$deadline = [DateTime]::UtcNow.AddSeconds(5); "
            "while (-not $player.NaturalDuration.HasTimeSpan -and [DateTime]::UtcNow -lt $deadline) "
            "{ Start-Sleep -Milliseconds 100 }; "
            "$player.Play(); "
            "$seconds = if ($player.NaturalDuration.HasTimeSpan) "
            "{ $player.NaturalDuration.TimeSpan.TotalSeconds } else { 180 }; "
            "Start-Sleep -Milliseconds ([Math]::Max(1000, [int]($seconds * 1000))); "
            "$player.Close()"
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            ["powershell", "-NoProfile", "-Sta", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        with self._lock:
            self._process = process
            self._backend = "process"
        return True

    def _play_linux(self, path: Path) -> bool:
        commands = (
            ("mpg123", ["mpg123", "-q", str(path)]),
            ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
            ("cvlc", ["cvlc", "--intf", "dummy", "--play-and-exit", str(path)]),
            ("aplay", ["aplay", str(path)]),
        )
        for executable, command in commands:
            if shutil.which(executable) and (executable != "aplay" or path.suffix.lower() == ".wav"):
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with self._lock:
                    self._process = process
                    self._backend = "process"
                return True
        return False


def play_track(track: Track, project_root: Path) -> bool:
    """Compatibility wrapper for callers that do not need stop control."""
    return MusicPlayer().play(track, project_root)
