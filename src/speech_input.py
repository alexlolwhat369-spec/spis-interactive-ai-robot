"""Offline microphone transcription using Vosk and the default audio input."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def text_from_result(raw_result: str) -> str:
    """Return a trimmed transcript from Vosk's JSON result."""
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError:
        return ""
    return str(data.get("text", "")).strip()


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
        # The small English Vosk model is trained for 16 kHz mono audio.
        try:
            sd.check_input_settings(device=device, samplerate=16000, channels=1, dtype="int16")
            self.sample_rate = 16000
        except sd.PortAudioError:
            device_info = sd.query_devices(device, "input")
            self.sample_rate = int(device_info["default_samplerate"])
        self._audio: queue.Queue[bytes] = queue.Queue()

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

    def listen_once(self, max_seconds: float = 12.0, stop_event: threading.Event | None = None) -> str:
        """Listen until speech completes, or until a push-to-talk key is released.

        With a stop event, final phrases are collected until the caller ends the
        turn. This prevents the robot from replying while the visitor is still
        holding Space and speaking.
        """
        recognizer = self._recognizer_type(self.model, self.sample_rate)
        deadline = time.monotonic() + max_seconds
        transcripts: list[str] = []
        while not self._audio.empty():
            self._audio.get_nowait()

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            self._audio.put(bytes(indata))

        with self._sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while time.monotonic() < deadline and (stop_event is None or not stop_event.is_set()):
                try:
                    audio_chunk = self._audio.get(timeout=0.1)
                except queue.Empty:
                    continue
                if recognizer.AcceptWaveform(audio_chunk):
                    transcript = text_from_result(recognizer.Result())
                    if transcript:
                        transcripts.append(transcript)
                        if stop_event is None:
                            return transcript
        final_transcript = text_from_result(recognizer.FinalResult())
        if final_transcript:
            transcripts.append(final_transcript)
        return " ".join(transcripts).strip()


class WindowsMicrophoneListener:
    """Uses the installed Windows English recognizer for the laptop demo."""

    def __init__(self) -> None:
        if os.name != "nt" or not shutil.which("powershell"):
            raise RuntimeError("Windows speech recognition is unavailable on this system.")

    def listen_once(self, max_seconds: float = 12.0, stop_event: threading.Event | None = None) -> str:
        del stop_event  # Windows recognition cannot bind its default input to a held key.
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
