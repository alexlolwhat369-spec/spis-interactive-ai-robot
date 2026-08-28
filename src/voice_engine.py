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

try:
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .robot_runtime import RobotDialogueSession
    from .robot_state import Reaction
    from .speech import PiperSpeaker, wav_duration_seconds
    from .speech_input import text_from_result
except ImportError:  # Supports direct execution / test import
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from robot_runtime import RobotDialogueSession
    from robot_state import Reaction
    from speech import PiperSpeaker, wav_duration_seconds
    from speech_input import text_from_result

ROOT = Path(__file__).resolve().parents[1]
OBJECT_CATALOG = ROOT / "data" / "object_catalog.json"
NO_SPEECH_REPLY = "I did not hear you. Please try again."


class VoiceEngine:
    """Load the speech + dialogue stack once and serve one voice turn at a time."""

    def __init__(
        self,
        vosk_model_dir: Path,
        piper_voice: Path,
        ollama_model: str = "spis-robot",
        use_ollama: bool = True,
    ) -> None:
        self.available = False
        self.reason = ""
        self._lock = threading.Lock()
        self._recognizer_type = None
        self._model = None
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
            heard = self.transcribe(pcm16_le_16k)
            if not heard:
                reply, reaction, provider_error = NO_SPEECH_REPLY, Reaction.CONFUSED, None
            else:
                result = self._session.respond(heard).conversation
                reply = result.command.reply
                reaction = result.command.reaction
                provider_error = result.provider_error
            audio = self._speaker.synthesize_wav_bytes(reply, str(reaction))
            return {
                "heard": heard,
                "reply": reply,
                "reaction": str(reaction),
                "provider_error": provider_error,
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "audio_seconds": round(wav_duration_seconds(audio), 2),
            }
