"""Server-side web voice turn: Vosk STT -> dialogue -> Piper TTS."""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import numpy as np

try:
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .diagnostics import TurnDiagnostics
    from .music import MusicPlayer, MusicSelector, Track
    from .robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from .robot_state import Action, Reaction
    from .speech import PiperSpeaker, wav_duration_seconds
    from .speech_input import transcribe_pcm16
except ImportError:  # Supports direct execution / test import
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from diagnostics import TurnDiagnostics
    from music import MusicPlayer, MusicSelector, Track
    from robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from robot_state import Action, Reaction
    from speech import PiperSpeaker, wav_duration_seconds
    from speech_input import transcribe_pcm16

ROOT = Path(__file__).resolve().parents[1]
OBJECT_CATALOG = ROOT / "data" / "object_catalog.json"
PLAYLIST = ROOT / "assets" / "music" / "playlist.json"
NO_SPEECH_REPLY = "I did not hear you. Please try again."
MUSIC_ACTIONS = frozenset(
    {Action.PLAY_MUSIC, Action.PAUSE_MUSIC, Action.RESUME_MUSIC, Action.NEXT_MUSIC, Action.STOP_MUSIC}
)


def mic_metrics(pcm16_le: bytes) -> tuple[float, float]:
    """Return peak and RMS levels (0..1) for little-endian PCM16 audio."""
    if not pcm16_le:
        return 0.0, 0.0
    usable = len(pcm16_le) - (len(pcm16_le) % np.dtype(np.int16).itemsize)
    if usable == 0:
        return 0.0, 0.0
    samples = np.frombuffer(pcm16_le[:usable], dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return round(peak, 3), round(rms, 3)


class VoiceEngine:
    """Load the speech and dialogue stack once and serialize voice turns.

    ``music`` accepts either a ``MusicSelector`` for desktop playback or a
    browser music controller exposing ``apply_action`` and ``state``. This
    keeps the shared speech behavior identical in both interfaces.
    """

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
        music: object | None = None,
        music_player: MusicPlayer | None = None,
        project_root: Path = ROOT,
        diagnostics: TurnDiagnostics | None = None,
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
        self._diagnostics = diagnostics or TurnDiagnostics()
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

    @property
    def _browser_music(self) -> object | None:
        if self._music is not None and callable(getattr(self._music, "apply_action", None)):
            return self._music
        return None

    def begin_turn(self) -> None:
        """Pause local playback while listening; browser audio ducks in JavaScript."""
        with self._lock:
            self._pending_track = None
            if self._browser_music is not None:
                self._music_paused_for_turn = False
                self._resume_after_turn = False
                return
            self._music_paused_for_turn = self._music_player.pause()
            self._resume_after_turn = self._music_paused_for_turn

    def finish_turn(self) -> bool:
        """Apply local playback only after the browser finishes the spoken reply."""
        with self._lock:
            if self._browser_music is not None:
                return False
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
        return transcribe_pcm16(
            self._model,
            self._recognizer_type,
            pcm16_le_16k,
            phrases=phrases,
        )

    def _track_path(self, track: Track) -> Path:
        path = Path(track.path)
        return path if path.is_absolute() else self._project_root / path

    def _prepare_browser_action(
        self,
        action: Action,
        category: str | None,
        reply: str,
        reaction: Reaction,
    ) -> tuple[str, Reaction]:
        controller = self._browser_music
        if controller is None or action not in MUSIC_ACTIONS:
            return reply, reaction
        state = controller.apply_action(action, category)
        if state.get("ok") is False:
            return str(state.get("reason") or "Music is unavailable."), Reaction.CONFUSED
        if action in {Action.PLAY_MUSIC, Action.NEXT_MUSIC} and state.get("title"):
            return f"Playing {state['title']}.", Reaction.HAPPY
        if action == Action.STOP_MUSIC:
            return "Music stopped.", Reaction.OK
        if action == Action.PAUSE_MUSIC:
            return "Music paused.", Reaction.OK
        if action == Action.RESUME_MUSIC:
            return "Resuming the music.", Reaction.OK
        return reply, reaction

    def _prepare_local_action(
        self,
        action: Action,
        category: str | None,
        reply: str,
        reaction: Reaction,
    ) -> tuple[str, Reaction]:
        if action == Action.PLAY_MUSIC:
            if not isinstance(self._music, MusicSelector):
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
            if not self._music_player.is_active or not isinstance(self._music, MusicSelector):
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

    def _prepare_action(
        self,
        action: Action,
        category: str | None,
        reply: str,
        reaction: Reaction,
    ) -> tuple[str, Reaction]:
        if self._browser_music is not None:
            return self._prepare_browser_action(action, category, reply, reaction)
        return self._prepare_local_action(action, category, reply, reaction)

    def _music_state(self) -> dict | None:
        if self._browser_music is not None:
            return self._browser_music.state()
        return None

    def handle(self, pcm16_le_16k: bytes) -> dict:
        """Run one full turn: transcribe, decide, synthesize, and report diagnostics."""
        with self._lock:
            peak, average = mic_metrics(pcm16_le_16k)
            self._diagnostics.begin()
            heard, transcript_source, guided_used = self.transcribe(
                pcm16_le_16k,
                self._recognition_phrases(),
            )
            route = "no_input"
            action = Action.NONE
            if not heard:
                self._diagnostics.no_input(mic_peak=peak, mic_average=average)
                reply, reaction, provider_error = NO_SPEECH_REPLY, Reaction.CONFUSED, None
            else:
                self._diagnostics.heard(
                    heard,
                    mic_peak=peak,
                    mic_average=average,
                    transcript_source=transcript_source,
                )
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
                self._diagnostics.complete(
                    route=route,
                    action=str(action),
                    reaction=str(reaction),
                    reply=reply,
                    provider_error=provider_error,
                )
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
                "mic_peak": peak,
                "mic_average": average,
                "now_playing": self._music_state(),
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "audio_seconds": round(wav_duration_seconds(audio), 2),
            }
