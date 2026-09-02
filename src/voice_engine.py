"""Server-side voice turn for the web UI: Vosk STT -> dialogue -> Piper TTS.

The browser records the microphone and posts raw 16 kHz mono PCM here, so speech
still stays on-device (no cloud STT). This reuses the same conversation session
and speech synthesis as the desktop ``voice_demo.py``; only the audio source and
sink differ (browser instead of the server's own mic/speakers).
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import numpy as np

try:
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .diagnostics import TurnDiagnostics
    from .robot_runtime import RobotDialogueSession
    from .robot_state import Action, Reaction
    from .speech import PiperSpeaker, wav_duration_seconds
    from .speech_input import text_from_result
except ImportError:  # Supports direct execution / test import
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from diagnostics import TurnDiagnostics
    from robot_runtime import RobotDialogueSession
    from robot_state import Action, Reaction
    from speech import PiperSpeaker, wav_duration_seconds
    from speech_input import text_from_result

ROOT = Path(__file__).resolve().parents[1]
OBJECT_CATALOG = ROOT / "data" / "object_catalog.json"
NO_SPEECH_REPLY = "I did not hear you. Please try again."
MUSIC_ACTIONS = frozenset(
    {Action.PLAY_MUSIC, Action.PAUSE_MUSIC, Action.RESUME_MUSIC, Action.NEXT_MUSIC, Action.STOP_MUSIC}
)


def mic_metrics(pcm16_le: bytes) -> tuple[float, float]:
    """Peak and RMS level (0..1) from a 16 kHz mono PCM16 utterance."""
    if not pcm16_le:
        return 0.0, 0.0
    samples = np.frombuffer(pcm16_le, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return round(peak, 3), round(rms, 3)


class VoiceEngine:
    """Load the speech + dialogue stack once and serve one voice turn at a time."""

    def __init__(
        self,
        vosk_model_dir: Path,
        piper_voice: Path,
        ollama_model: str = "spis-robot",
        use_ollama: bool = True,
        music: object | None = None,
        diagnostics: TurnDiagnostics | None = None,
    ) -> None:
        self.available = False
        self.reason = ""
        self._lock = threading.Lock()
        self._recognizer_type = None
        self._model = None
        self._music = music
        self._diagnostics = diagnostics or TurnDiagnostics()
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as error:
            self.reason = f"Vosk not installed: {error}"
            return
        try:
            if not vosk_model_dir.is_dir():
                raise FileNotFoundError(f"Vosk model not found at {vosk_model_dir}")
            self._model = Model(str(vosk_model_dir))
            self._recognizer_type = KaldiRecognizer
            self._speaker = PiperSpeaker(piper_voice)
            provider = OllamaConversationProvider(ollama_model) if use_ollama else RuleConversationProvider()
            self._session = RobotDialogueSession(provider, OBJECT_CATALOG)
        except (FileNotFoundError, RuntimeError) as error:
            self.reason = str(error)
            return
        self.available = True

    def transcribe(self, pcm16_le_16k: bytes) -> str:
        """Turn a full utterance of 16 kHz mono PCM16 into text."""
        recognizer = self._recognizer_type(self._model, 16000)
        recognizer.SetWords(False)
        chunk = 4000
        for start in range(0, len(pcm16_le_16k), chunk):
            recognizer.AcceptWaveform(pcm16_le_16k[start : start + chunk])
        return text_from_result(recognizer.FinalResult())

    def handle(self, pcm16_le_16k: bytes) -> dict:
        """Run one full turn: transcribe -> respond -> synthesize. Serialized."""
        with self._lock:
            peak, average = mic_metrics(pcm16_le_16k)
            self._diagnostics.begin()
            heard = self.transcribe(pcm16_le_16k)
            if not heard:
                self._diagnostics.no_input(mic_peak=peak, mic_average=average)
                reply, reaction = NO_SPEECH_REPLY, Reaction.CONFUSED
                provider_error, route, action = None, "no_input", str(Action.NONE)
            else:
                self._diagnostics.heard(
                    heard, mic_peak=peak, mic_average=average, transcript_source="final"
                )
                session_result = self._session.respond(heard)
                result = session_result.conversation
                reply = result.command.reply
                reaction = result.command.reaction
                provider_error = result.provider_error
                route = session_result.route.value
                action = result.command.action.value
                if self._music is not None and result.command.action in MUSIC_ACTIONS:
                    self._music.apply_action(result.command.action, result.music_category)
                self._diagnostics.complete(
                    route=route,
                    action=action,
                    reaction=str(reaction),
                    reply=reply,
                    provider_error=provider_error,
                )
            audio = self._speaker.synthesize_wav_bytes(reply, str(reaction))
            return {
                "heard": heard,
                "reply": reply,
                "reaction": str(reaction),
                "provider_error": provider_error,
                "route": route,
                "action": action,
                "mic_peak": peak,
                "mic_average": average,
                "now_playing": self._music.state() if self._music is not None else None,
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "audio_seconds": round(wav_duration_seconds(audio), 2),
            }
