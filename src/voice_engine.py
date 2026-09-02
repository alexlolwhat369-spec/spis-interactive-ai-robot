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
    from .music import MusicPlayer, MusicSelector, Track
    from .robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from .robot_state import Action, Reaction
    from .speech import PiperSpeaker, wav_duration_seconds
    from .speech_input import transcribe_pcm16
except ImportError:  # Supports direct execution / test import
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from music import MusicPlayer, MusicSelector, Track
    from robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from robot_state import Action, Reaction
    from speech import PiperSpeaker, wav_duration_seconds
    from speech_input import transcribe_pcm16

ROOT = Path(__file__).resolve().parents[1]
OBJECT_CATALOG = ROOT / "data" / "object_catalog.json"
PLAYLIST = ROOT / "assets" / "music" / "playlist.json"
NO_SPEECH_REPLY = "I did not hear you. Please try again."


class VoiceEngine:
    """Load the speech + dialogue stack once and serve one voice turn at a time."""

    def __init__(
        self,
        vosk_model_dir: Path,
        piper_voice: Path,
        ollama_model: str = "spis-robot",
        use_ollama: bool = True,
        *,
        session: object | None = None,
        speaker: object | None = None,
        recognizer_model: object | None = None,
        recognizer_type: object | None = None,
        music: MusicSelector | None = None,
        music_player: MusicPlayer | None = None,
        project_root: Path = ROOT,
    ) -> None:
        self.available = False
        self.reason = ""
        self._lock = threading.Lock()
        self._recognizer_type = recognizer_type
        self._model = recognizer_model
        self._session = session
        self._speaker = speaker
        self._project_root = project_root
        self._music = music or (MusicSelector.from_file(PLAYLIST) if PLAYLIST.is_file() else None)
        self._music_player = music_player or MusicPlayer()
        self._music_category: str | None = None
        self._music_paused_for_turn = False
        self._resume_after_turn = False
        self._pending_track: Track | None = None

        if all(value is not None for value in (session, speaker, recognizer_model, recognizer_type)):
            self.available = True
            return
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

    def begin_turn(self) -> None:
        """Pause local music while the browser microphone is listening."""
        with self._lock:
            self._pending_track = None
            self._music_paused_for_turn = self._music_player.pause()
            self._resume_after_turn = self._music_paused_for_turn

    def finish_turn(self) -> bool:
        """Apply delayed playback only after the browser finishes speaking."""
        with self._lock:
            changed = False
            if self._pending_track is not None:
                changed = self._music_player.play(self._pending_track, self._project_root)
            elif self._resume_after_turn:
                changed = self._music_player.resume()
            self._pending_track = None
            self._music_paused_for_turn = False
            self._resume_after_turn = False
            return changed

    def stop(self) -> None:
        with self._lock:
            self._pending_track = None
            self._resume_after_turn = False
            self._music_player.stop()

    def _recognition_phrases(self) -> tuple[str, ...] | None:
        if bool(getattr(self._session, "expects_game_answer", False)):
            return GAME_ANSWER_PHRASES
        if bool(getattr(self._session, "expects_music_category", False)):
            return MUSIC_CATEGORY_PHRASES
        return None

    def transcribe(self, pcm16_le_16k: bytes, phrases: tuple[str, ...] | None = None) -> tuple[str, str, bool]:
        """Turn one browser utterance into text plus diagnostic metadata."""
        return transcribe_pcm16(
            self._model,
            self._recognizer_type,
            pcm16_le_16k,
            phrases=phrases,
        )

    def _track_path(self, track: Track) -> Path:
        path = Path(track.path)
        return path if path.is_absolute() else self._project_root / path

    def _prepare_action(
        self,
        action: Action,
        category: str | None,
        reply: str,
        reaction: Reaction,
    ) -> tuple[str, Reaction]:
        if action == Action.PLAY_MUSIC:
            if self._music is None:
                self._resume_after_turn = self._music_paused_for_turn
                return "The playlist is not configured yet.", Reaction.CONFUSED
            track = self._music.choose(requested_category=category)
            if not self._track_path(track).is_file():
                self._resume_after_turn = self._music_paused_for_turn
                return "I understood the request, but that music file is not installed.", Reaction.CONFUSED
            self._pending_track = track
            self._music_category = category or track.category
            self._resume_after_turn = False
            return f"Playing {track.title}.", Reaction.HAPPY
        if action == Action.STOP_MUSIC:
            stopped = self._music_player.stop()
            self._resume_after_turn = False
            return ("Music stopped.", Reaction.OK) if stopped else ("No music is playing.", Reaction.CONFUSED)
        if action == Action.PAUSE_MUSIC:
            paused = self._music_paused_for_turn or self._music_player.pause()
            self._resume_after_turn = False
            return ("Music paused.", Reaction.OK) if paused else ("No music is playing.", Reaction.CONFUSED)
        if action == Action.RESUME_MUSIC:
            if not self._music_player.is_active:
                self._resume_after_turn = False
                return "There is no paused music to resume.", Reaction.CONFUSED
            self._resume_after_turn = True
            return "Resuming the music.", Reaction.OK
        if action == Action.NEXT_MUSIC:
            if not self._music_player.is_active or self._music is None:
                self._resume_after_turn = False
                return "There is no active playlist to skip.", Reaction.CONFUSED
            track = self._music.choose(requested_category=self._music_category)
            if not self._track_path(track).is_file():
                self._resume_after_turn = self._music_paused_for_turn
                return "The next music file is not installed.", Reaction.CONFUSED
            self._pending_track = track
            self._resume_after_turn = False
            return f"Playing {track.title}.", Reaction.HAPPY
        return reply, reaction

    def handle(self, pcm16_le_16k: bytes) -> dict:
        """Run one full turn: transcribe -> respond -> synthesize. Serialized."""
        with self._lock:
            heard, transcript_source, guided_used = self.transcribe(
                pcm16_le_16k,
                self._recognition_phrases(),
            )
            route = "no_input"
            action = Action.NONE
            if not heard:
                reply, reaction, provider_error = NO_SPEECH_REPLY, Reaction.CONFUSED, None
            else:
                session_result = self._session.respond(heard)
                result = session_result.conversation
                route = str(session_result.route)
                action = result.command.action
                reply, reaction = self._prepare_action(
                    action,
                    result.music_category,
                    result.command.reply,
                    result.command.reaction,
                )
                provider_error = result.provider_error
            audio = self._speaker.synthesize_wav_bytes(reply, str(reaction))
            return {
                "heard": heard,
                "reply": reply,
                "reaction": str(reaction),
                "route": route,
                "action": str(action),
                "provider_error": provider_error,
                "transcript_source": transcript_source,
                "guided_used": guided_used,
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "audio_seconds": round(wav_duration_seconds(audio), 2),
            }
