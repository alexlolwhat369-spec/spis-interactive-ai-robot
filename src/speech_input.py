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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpeechMetrics:
    duration_seconds: float = 0.0
    peak_level: float = 0.0
    average_level: float = 0.0
    transcript_source: str = "none"
    guided_used: bool = False


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


def select_transcript(full_text: str, guided_text: str, phrases: Sequence[str] | None) -> tuple[str, bool]:
    """Prefer a guided short answer, but preserve richer natural-language replies."""
    full_text = " ".join(full_text.split()).strip()
    guided_text = " ".join(guided_text.replace("[unk]", "").split()).strip()
    allowed = {" ".join(item.lower().split()) for item in phrases or ()}
    guided_is_known = guided_text.lower() in allowed
    full_is_richer = len(full_text.split()) > max(3, len(guided_text.split()) + 2)
    if guided_is_known and not full_is_richer:
        return guided_text, True
    return (full_text or guided_text), False


def transcribe_pcm16(
    model: object,
    recognizer_type: object,
    pcm16_le: bytes,
    *,
    sample_rate: int = 16000,
    phrases: Sequence[str] | None = None,
) -> tuple[str, str, bool]:
    """Decode a complete PCM turn with the same optional guided vocabulary.

    The browser sends a finished utterance instead of a live microphone stream.
    Running both recognizers over the same bytes keeps short game and music
    answers consistent with the desktop push-to-talk path.
    """
    vocabulary = recognizer_vocabulary(phrases)
    recognizer = recognizer_type(model, sample_rate)
    guided = recognizer_type(model, sample_rate, vocabulary) if vocabulary is not None else None
    for current in (recognizer, guided):
        set_words = getattr(current, "SetWords", None)
        if callable(set_words):
            set_words(False)

    full_parts: list[str] = []
    guided_parts: list[str] = []
    chunk_size = 4000
    for start in range(0, len(pcm16_le), chunk_size):
        chunk = pcm16_le[start : start + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            text = text_from_result(recognizer.Result())
            if text:
                full_parts.append(text)
        if guided is not None and guided.AcceptWaveform(chunk):
            text = text_from_result(guided.Result())
            if text:
                guided_parts.append(text)

    final_text = text_from_result(recognizer.FinalResult())
    if final_text:
        full_parts.append(final_text)
    if guided is not None:
        guided_final = text_from_result(guided.FinalResult())
        if guided_final:
            guided_parts.append(guided_final)

    full_text = " ".join(full_parts).strip()
    guided_text = " ".join(guided_parts).strip()
    selected, guided_used = select_transcript(full_text, guided_text, phrases)
    source = "guided" if guided_used else ("final" if full_text else "none")
    return selected, source, guided_used


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
        self.last_metrics = SpeechMetrics()

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
        recognizer = self._recognizer_type(self.model, self.sample_rate)
        guided_recognizer = (
            self._recognizer_type(self.model, self.sample_rate, vocabulary) if vocabulary is not None else None
        )
        deadline = time.monotonic() + max_seconds
        started = time.monotonic()
        release_deadline: float | None = None
        transcripts: list[str] = []
        guided_transcripts: list[str] = []
        best_partial = ""
        guided_best_partial = ""
        levels: list[float] = []
        while not self._audio.empty():
            self._audio.get_nowait()

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            usable_bytes = len(indata) - (len(indata) % np.dtype(np.int16).itemsize)
            samples = np.frombuffer(indata[:usable_bytes], dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                self.input_level = min(1.0, rms / 9000.0)
                levels.append(self.input_level)
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
                            self.last_metrics = SpeechMetrics(
                                time.monotonic() - started,
                                max(levels, default=0.0),
                                sum(levels) / len(levels) if levels else 0.0,
                                "final",
                            )
                            return transcript
                else:
                    partial_result = getattr(recognizer, "PartialResult", None)
                    if callable(partial_result):
                        partial = partial_text_from_result(partial_result())
                        if len(partial) >= len(best_partial):
                            best_partial = partial
                if guided_recognizer is not None:
                    if guided_recognizer.AcceptWaveform(audio_chunk):
                        guided = text_from_result(guided_recognizer.Result())
                        if guided:
                            guided_transcripts.append(guided)
                            guided_best_partial = ""
                    else:
                        guided_partial_result = getattr(guided_recognizer, "PartialResult", None)
                        if callable(guided_partial_result):
                            guided_partial = partial_text_from_result(guided_partial_result())
                            if len(guided_partial) >= len(guided_best_partial):
                                guided_best_partial = guided_partial
                if release_deadline is not None and time.monotonic() >= release_deadline:
                    break
        final_transcript = text_from_result(recognizer.FinalResult())
        if final_transcript:
            transcripts.append(final_transcript)
        elif best_partial and not transcripts:
            transcripts.append(best_partial)
        if guided_recognizer is not None:
            guided_final = text_from_result(guided_recognizer.FinalResult())
            if guided_final:
                guided_transcripts.append(guided_final)
            elif guided_best_partial and not guided_transcripts:
                guided_transcripts.append(guided_best_partial)
        self.input_level = 0.0
        full_text = " ".join(transcripts).strip()
        guided_text = " ".join(guided_transcripts).strip()
        selected, guided_used = select_transcript(full_text, guided_text, phrases)
        source = "guided" if guided_used else ("final" if final_transcript else "partial" if best_partial else "none")
        self.last_metrics = SpeechMetrics(
            time.monotonic() - started,
            max(levels, default=0.0),
            sum(levels) / len(levels) if levels else 0.0,
            source,
            guided_used,
        )
        return selected


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
