"""Offline microphone transcription using Vosk and the default audio input."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def text_from_result(raw_result: str) -> str:
    """Return a trimmed transcript from Vosk's JSON result."""
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError:
        return ""
    return str(data.get("text", "")).strip()


def partial_text_from_result(raw_result: str) -> str:
    """Return Vosk's best unfinished phrase without treating it as final."""
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError:
        return ""
    return str(data.get("partial", "")).strip()


def recognizer_vocabulary(phrases: Sequence[str] | None) -> str | None:
    """Build the optional Vosk grammar used for short, closed answers."""
    if not phrases:
        return None
    normalized = dict.fromkeys(phrase.strip().lower() for phrase in phrases if phrase.strip())
    return json.dumps([*normalized, "[unk]"])


class MicrophoneListener:
    """Listens for one completed English utterance; it never uploads audio."""

    def __init__(self, model_path: Path, device: int | None = None) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Speech model not found at {model_path}. Download the Vosk English model first."
            )
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as error:
            raise RuntimeError("Install the speech dependencies with: python -m pip install -r requirements.txt") from error

        self._sounddevice: Any = sd
        self._recognizer_type: Any = KaldiRecognizer
        self.model: Any = Model(str(model_path))
        self.device = device
        device_info = sd.query_devices(device, "input")
        self.device_name = str(device_info["name"])
        # The small English Vosk model is trained for 16 kHz mono audio.
        try:
            sd.check_input_settings(device=device, samplerate=16000, channels=1, dtype="int16")
            self.sample_rate = 16000
        except sd.PortAudioError:
            self.sample_rate = int(device_info["default_samplerate"])
        self._audio: queue.Queue[bytes] = queue.Queue()
        self.release_grace_seconds = 0.7
        self.input_level = 0.0

    @staticmethod
    def input_devices() -> list[tuple[int, str]]:
        try:
            import sounddevice as sd
        except ImportError:
            return []
        return [
            (index, str(device["name"]))
            for index, device in enumerate(sd.query_devices())
            if int(device["max_input_channels"]) > 0
        ]

    def listen_once(
        self,
        max_seconds: float = 12.0,
        stop_event: threading.Event | None = None,
        phrases: Sequence[str] | None = None,
    ) -> str:
        """Listen until speech completes, or until a push-to-talk key is released.

        With a stop event, final phrases are collected until the caller ends the
        turn. This prevents the robot from replying while the visitor is still
        holding Space and speaking.
        """
        vocabulary = recognizer_vocabulary(phrases)
        recognizer = (
            self._recognizer_type(self.model, self.sample_rate, vocabulary)
            if vocabulary is not None
            else self._recognizer_type(self.model, self.sample_rate)
        )
        deadline = time.monotonic() + max_seconds
        release_deadline: float | None = None
        transcripts: list[str] = []
        best_partial = ""
        while not self._audio.empty():
            self._audio.get_nowait()

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            usable_bytes = len(indata) - (len(indata) % np.dtype(np.int16).itemsize)
            samples = np.frombuffer(indata[:usable_bytes], dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                self.input_level = min(1.0, rms / 9000.0)
            self._audio.put(bytes(indata))

        with self._sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            # 50 ms blocks make short push-to-talk answers feel more immediate.
            blocksize=800,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while time.monotonic() < deadline:
                now = time.monotonic()
                if stop_event is not None and stop_event.is_set() and release_deadline is None:
                    release_deadline = now + self.release_grace_seconds
                try:
                    audio_chunk = self._audio.get(timeout=0.05)
                except queue.Empty:
                    if release_deadline is not None and time.monotonic() >= release_deadline:
                        break
                    continue
                if recognizer.AcceptWaveform(audio_chunk):
                    transcript = text_from_result(recognizer.Result())
                    if transcript:
                        transcripts.append(transcript)
                        best_partial = ""
                        if stop_event is None:
                            return transcript
                else:
                    partial_result = getattr(recognizer, "PartialResult", None)
                    if callable(partial_result):
                        partial = partial_text_from_result(partial_result())
                        if len(partial) >= len(best_partial):
                            best_partial = partial
                if release_deadline is not None and time.monotonic() >= release_deadline:
                    break
        final_transcript = text_from_result(recognizer.FinalResult())
        if final_transcript:
            transcripts.append(final_transcript)
        elif best_partial and not transcripts:
            transcripts.append(best_partial)
        self.input_level = 0.0
        return " ".join(transcripts).strip()


class WindowsMicrophoneListener:
    """Uses the installed Windows English recognizer for the laptop demo."""

    def __init__(self) -> None:
        if os.name != "nt" or not shutil.which("powershell"):
            raise RuntimeError("Windows speech recognition is unavailable on this system.")

    def listen_once(
        self,
        max_seconds: float = 12.0,
        stop_event: threading.Event | None = None,
        phrases: Sequence[str] | None = None,
    ) -> str:
        del stop_event, phrases  # Windows recognition cannot bind its default input to a held key.
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$culture = [System.Globalization.CultureInfo]::GetCultureInfo('en-US'); "
            "$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture); "
            "$engine.SetInputToDefaultAudioDevice(); "
            "$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
            "$done = New-Object System.Threading.ManualResetEvent($false); "
            "$script:heard = ''; "
            "$engine.add_SpeechRecognized({ param($sender, $event); $script:heard = $event.Result.Text; $done.Set() }); "
            "$engine.RecognizeAsync(); "
            f"$done.WaitOne([TimeSpan]::FromSeconds({max_seconds})) | Out-Null; "
            "$engine.RecognizeAsyncCancel(); $engine.Dispose(); [Console]::Write($script:heard)"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=max_seconds + 5,
            )
            return completed.stdout.strip()
        except subprocess.TimeoutExpired:
            return ""
