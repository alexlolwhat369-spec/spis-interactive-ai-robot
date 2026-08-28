"""Local speech synthesis and small, explainable delivery controls."""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SpeechStyle:
    """Piper controls for one short phrase; lower length means faster speech."""

    length_scale: float
    noise_scale: float
    noise_w_scale: float


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    style: SpeechStyle


NEUTRAL_STYLE = SpeechStyle(length_scale=0.90, noise_scale=0.80, noise_w_scale=0.90)
REACTION_STYLES = {
    "happy": SpeechStyle(length_scale=0.85, noise_scale=0.86, noise_w_scale=0.96),
    "proud": SpeechStyle(length_scale=0.80, noise_scale=0.92, noise_w_scale=1.02),
    "heart": SpeechStyle(length_scale=0.98, noise_scale=0.74, noise_w_scale=0.84),
    "annoyed": SpeechStyle(length_scale=0.94, noise_scale=0.70, noise_w_scale=0.80),
    "curious": SpeechStyle(length_scale=0.88, noise_scale=0.86, noise_w_scale=1.00),
    "confused": SpeechStyle(length_scale=1.00, noise_scale=0.72, noise_w_scale=0.82),
    "listening": SpeechStyle(length_scale=0.98, noise_scale=0.76, noise_w_scale=0.86),
    "thinking": SpeechStyle(length_scale=0.96, noise_scale=0.76, noise_w_scale=0.88),
    "speaking": NEUTRAL_STYLE,
    "idle": NEUTRAL_STYLE,
}
QUESTION_STYLE = SpeechStyle(length_scale=0.97, noise_scale=0.84, noise_w_scale=0.98)
EMPHATIC_STYLE = SpeechStyle(length_scale=0.78, noise_scale=0.96, noise_w_scale=1.06)

WINDOWS_VOICE_HINTS = ("natural", "neural", "aria", "jenny", "google", "samantha", "zira")


def clean_spoken_text(text: str) -> str:
    """Turn display-oriented Markdown into a short sentence suitable for TTS."""
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"[*_`#>]", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    return " ".join(cleaned.split())


def preferred_windows_voice(installed: list[str], requested: str = "auto") -> str | None:
    """Select an installed voice using the natural-voice preference from the web reference."""
    if not installed:
        return None
    if requested and requested.casefold() != "auto":
        exact = next((name for name in installed if name.casefold() == requested.casefold()), None)
        if exact:
            return exact
    for hint in WINDOWS_VOICE_HINTS:
        match = next((name for name in installed if hint in name.casefold()), None)
        if match:
            return match
    return installed[0]


def plan_speech(text: str, reaction: str | None = None) -> list[SpeechSegment]:
    """Split a reply into audible phrases and give each a deliberate delivery style.

    Piper does not understand SSML emotion tags, so the planner stays honest: it
    controls pace and natural variation, with punctuation marking an emphasis or
    a question. A future trained expressive voice can use this same segment plan.
    """
    cleaned = clean_spoken_text(text)
    if not cleaned:
        return []
    base_style = REACTION_STYLES.get((reaction or "").lower(), NEUTRAL_STYLE)
    phrases = re.findall(r"[^.!?]+[.!?]*", cleaned)
    if not phrases:
        phrases = [cleaned]
    segments: list[SpeechSegment] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        style = QUESTION_STYLE if phrase.endswith("?") else EMPHATIC_STYLE if phrase.endswith("!") else base_style
        segments.append(SpeechSegment(phrase, style))
    return segments


class LocalSpeaker:
    def __init__(self, voice_name: str = "auto", voice_rate: int = 4) -> None:
        self.voice_name = voice_name
        self.voice_rate = max(-10, min(10, voice_rate))

    def speak(self, text: str, reaction: str | None = None) -> bool:
        """Speak locally without sending the visitor's words to a cloud service."""
        text = clean_spoken_text(text)
        if not text:
            return True
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            words_per_minute = 175 + self.voice_rate * 9
            completed = subprocess.run([executable, "-v", "en-us", "-s", str(words_per_minute), text], check=False)
            return completed.returncode == 0
        if os.name == "nt" and shutil.which("powershell"):
            # Base64 keeps visitor text and the configurable voice out of the PowerShell source.
            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            encoded_voice = base64.b64encode(self.voice_name.encode("utf-8")).decode("ascii")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')); "
                f"$voiceName = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_voice}')); "
                "$names = @($voice.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }); "
                "$chosen = $null; "
                "if ($voiceName -and $voiceName.ToLowerInvariant() -ne 'auto') { "
                "$chosen = $names | Where-Object { $_.ToLowerInvariant() -eq $voiceName.ToLowerInvariant() } "
                "| Select-Object -First 1 }; "
                "if (-not $chosen) { foreach ($hint in @('natural','neural','aria','jenny','google','samantha','zira')) { "
                "$chosen = $names | Where-Object { $_.ToLowerInvariant().Contains($hint) } | Select-Object -First 1; "
                "if ($chosen) { break } } }; "
                "if (-not $chosen -and $names.Count -gt 0) { $chosen = $names[0] }; "
                "if ($chosen) { try { $voice.SelectVoice($chosen) } catch {} }; "
                f"$voice.Rate = {self.voice_rate}; "
                "$voice.Speak($text)"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return completed.returncode == 0
        return False


class PiperSpeaker:
    """A natural local neural voice with phrase-by-phrase delivery controls."""

    def __init__(
        self,
        model_path: Path,
        length_scale: float = NEUTRAL_STYLE.length_scale,
        noise_scale: float = NEUTRAL_STYLE.noise_scale,
        noise_w_scale: float = NEUTRAL_STYLE.noise_w_scale,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Piper voice model not found: {model_path}")
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as error:
            raise RuntimeError("Install the speech dependencies with: python -m pip install -r requirements.txt") from error
        self._voice = PiperVoice.load(str(model_path))
        self._default_style = SpeechStyle(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

    def synthesize_wav_bytes(self, text: str, reaction: str | None = None) -> bytes:
        """Render a reply to a single in-memory WAV without playing it.

        Same phrase-by-phrase delivery as ``speak()``; returns the WAV bytes so a
        caller (e.g. the web UI) can stream them to a browser instead of a speaker.
        Returns ``b""`` when there is nothing to say.
        """
        segments = plan_speech(text, reaction)
        if not segments:
            return b""
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            for segment in segments:
                style = segment.style if reaction is not None else self._default_style
                config = self._synthesis_config(style)
                # Piper initializes a WAV header for every synthesis call.
                # Keep each phrase in memory, then append its PCM frames to
                # one visitor-facing audio file.
                phrase_buffer = io.BytesIO()
                with wave.open(phrase_buffer, "wb") as phrase_wav:
                    self._voice.synthesize_wav(segment.text, phrase_wav, syn_config=config)
                phrase_buffer.seek(0)
                with wave.open(phrase_buffer, "rb") as phrase_wav:
                    if wav_file.getnframes() == 0:
                        wav_file.setparams(phrase_wav.getparams())
                    elif _audio_format(wav_file) != _audio_format(phrase_wav):
                        raise RuntimeError("Piper returned incompatible audio settings between phrases.")
                    wav_file.writeframes(phrase_wav.readframes(phrase_wav.getnframes()))
        return output.getvalue()

    def speak(self, text: str, reaction: str | None = None) -> bool:
        audio = self.synthesize_wav_bytes(text, reaction)
        if not audio:
            return True
        temporary_file = tempfile.NamedTemporaryFile(prefix="spis-robot-", suffix=".wav", delete=False)
        path = Path(temporary_file.name)
        temporary_file.close()
        try:
            path.write_bytes(audio)
            if os.name == "nt":
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return True
            player = shutil.which("aplay")
            if player:
                return subprocess.run([player, str(path)], check=False).returncode == 0
            return False
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _synthesis_config(style: SpeechStyle):
        from piper import SynthesisConfig

        return SynthesisConfig(
            length_scale=style.length_scale,
            noise_scale=style.noise_scale,
            noise_w_scale=style.noise_w_scale,
        )


def wav_duration_seconds(data: bytes) -> float:
    """Playback length of an in-memory WAV, used to time the robot's speaking state."""
    if not data:
        return 0.0
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _audio_format(wav_file: wave.Wave_read | wave.Wave_write) -> tuple[int, int, int, str, str]:
    """Return only header values that must agree before PCM frames are appended."""
    return (
        wav_file.getnchannels(),
        wav_file.getsampwidth(),
        wav_file.getframerate(),
        wav_file.getcomptype(),
        wav_file.getcompname(),
    )
