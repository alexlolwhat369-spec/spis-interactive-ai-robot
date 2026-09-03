"""Full fair demo: camera gestures, animated face, local speech, and Ollama."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

try:
    from .camera_io import open_working_camera
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .diagnostics import TurnDiagnostics
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .music import MusicPlayer, MusicSelector, SoundEffectPlayer
    from .robot_face import render_face
    from .robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from .robot_state import Action, Reaction, RobotCommand, RobotController
    from .push_to_talk import SpaceKey
    from .speech import FallbackSpeaker, LocalSpeaker, PiperSpeaker
    from .speech_input import MicrophoneListener, WindowsMicrophoneListener
except ImportError:  # Supports direct execution: python src/interactive_robot.py
    from camera_io import open_working_camera
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from diagnostics import TurnDiagnostics
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from music import MusicPlayer, MusicSelector, SoundEffectPlayer
    from robot_face import render_face
    from robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from robot_state import Action, Reaction, RobotCommand, RobotController
    from push_to_talk import SpaceKey
    from speech import FallbackSpeaker, LocalSpeaker, PiperSpeaker
    from speech_input import MicrophoneListener, WindowsMicrophoneListener


ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DEFAULT_SPEECH_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
DEFAULT_PIPER_VOICE = ROOT / "models" / "voices" / "en_US-lessac-medium.onnx"
GESTURE_LOCKING_REACTIONS = frozenset({Reaction.LISTENING, Reaction.THINKING, Reaction.SPEAKING})


@dataclass(frozen=True)
class VoiceActivity:
    reaction: Reaction
    subtitle: str = ""
    speaking: bool = False


def gestures_locked(activity: VoiceActivity, voice_busy: bool = False) -> bool:
    """Conversation always wins over an accidental camera gesture."""
    return voice_busy or activity.speaking or activity.reaction in GESTURE_LOCKING_REACTIONS


class VoiceState:
    def __init__(self) -> None:
        self._activity = VoiceActivity(Reaction.IDLE)
        self._lock = threading.Lock()

    def set(self, reaction: Reaction, subtitle: str = "", *, speaking: bool = False) -> None:
        with self._lock:
            self._activity = VoiceActivity(reaction, subtitle, speaking)

    def current(self) -> VoiceActivity:
        with self._lock:
            return self._activity


class GestureFeedback:
    """Keep sound-backed reactions visible for exactly the effect duration."""

    def __init__(
        self,
        heart_hold_seconds: float = 1.5,
        mohan_hold_seconds: float = 2.0,
        hold_seconds: dict[str, float] | None = None,
    ) -> None:
        self.hold_seconds = (
            hold_seconds
            if hold_seconds is not None
            else {"heart": heart_hold_seconds, "mohan": mohan_hold_seconds}
        )
        self._held_until = 0.0
        self._held_activity: VoiceActivity | None = None
        self._held_label: str | None = None
        self._completed_label: str | None = None
        self._previous_label = "none"

    def choose(
        self,
        label: str,
        gesture_command: RobotCommand,
        voice_activity: VoiceActivity,
        now: float,
    ) -> VoiceActivity:
        if gestures_locked(voice_activity):
            self._held_until = 0.0
            self._held_activity = None
            self._held_label = None
            self._completed_label = None
            self._previous_label = "none"
            return voice_activity
        is_new_activation = label not in {"none", "unknown"} and label != self._previous_label
        self._previous_label = label
        if is_new_activation and label in self.hold_seconds:
            same_effect_is_active = self._held_label == label and now < self._held_until
            if not same_effect_is_active:
                self._held_until = now + self.hold_seconds[label]
                self._held_activity = VoiceActivity(gesture_command.reaction, gesture_command.reply)
                self._held_label = label
                self._completed_label = None
        if now < self._held_until and self._held_activity is not None:
            return self._held_activity
        if self._held_label is not None:
            self._completed_label = self._held_label
        self._held_activity = None
        self._held_label = None
        if label == self._completed_label and label in self.hold_seconds:
            return voice_activity
        if label != "none":
            return VoiceActivity(gesture_command.reaction, gesture_command.reply)
        return voice_activity


class GestureSoundFeedback:
    """Trigger a mapped effect once when a gesture becomes active."""

    def __init__(self, player: SoundEffectPlayer, sounds: dict[str, Path]) -> None:
        self.player = player
        self.sounds = sounds
        self._previous_label = "none"

    def update(self, label: str) -> bool:
        triggered = False
        path = self.sounds.get(label)
        if label != self._previous_label and path is not None:
            triggered = self.player.play(path)
            if not triggered:
                print(f"Sound effect unavailable: {path}")
        self._previous_label = label
        return triggered

    def stop(self) -> None:
        self.player.stop()
        self._previous_label = "none"


class VoiceWorker:
    def __init__(
        self,
        listener: object,
        speaker: object,
        session: RobotDialogueSession,
        listen_seconds: float,
        music: MusicSelector | None = None,
        project_root: Path = ROOT,
        music_player: MusicPlayer | None = None,
        diagnostics: TurnDiagnostics | None = None,
    ) -> None:
        self.listener = listener
        self.speaker = speaker
        self.session = session
        self.listen_seconds = listen_seconds
        self.music = music
        self.project_root = project_root
        self.music_player = music_player or MusicPlayer()
        self.diagnostics = diagnostics or TurnDiagnostics()
        self._music_category: str | None = None
        self._music_paused_for_turn = False
        self.state = VoiceState()
        self.stop_requested = threading.Event()
        self._release_listening = threading.Event()
        self._busy = threading.Event()

    def start(self) -> None:
        self._start_task(self._greet)

    def stop(self) -> None:
        self.stop_requested.set()
        self._release_listening.set()
        self.music_player.stop()

    def request_listening(self) -> bool:
        """Start one listener turn only when the visitor presses Space."""
        if self.stop_requested.is_set() or self._busy.is_set():
            return False
        # Pause instead of discarding the song. An unrelated conversation turn
        # resumes it after the robot finishes replying.
        self._music_paused_for_turn = self.music_player.pause()
        self._release_listening.clear()
        self.diagnostics.begin()
        self._start_task(self._listen_and_respond)
        return True

    def release_listening(self) -> None:
        self._release_listening.set()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def _start_task(self, action: Callable[[], None]) -> None:
        self._busy.set()

        def run() -> None:
            try:
                action()
            finally:
                self._busy.clear()

        threading.Thread(target=run, daemon=True).start()

    def _say(self, text: str, delivery_reaction: Reaction = Reaction.SPEAKING) -> None:
        self.state.set(delivery_reaction, text, speaking=True)
        print(f"Robot says: {text}")
        try:
            spoken = self.speaker.speak(text, delivery_reaction)
        except Exception as error:
            print(f"TTS error: {error}. The subtitle is still available.")
            spoken = False
        if not spoken:
            print("TTS unavailable: the reply remains visible as a subtitle.")

    def _greet(self) -> None:
        self._say("Hello! I am ready to talk and see your gestures.", Reaction.HAPPY)
        self.state.set(Reaction.IDLE, "Hold SPACE to talk")

    def _listen_and_respond(self) -> None:
        self.state.set(Reaction.LISTENING, "Hold SPACE and speak")
        if self.session.expects_game_answer:
            phrases = GAME_ANSWER_PHRASES
        elif getattr(self.session, "expects_music_category", False):
            phrases = MUSIC_CATEGORY_PHRASES
        else:
            phrases = None
        heard = self.listener.listen_once(self.listen_seconds, self._release_listening, phrases=phrases)
        metrics = getattr(self.listener, "last_metrics", None)
        mic_peak = float(getattr(metrics, "peak_level", 0.0))
        mic_average = float(getattr(metrics, "average_level", 0.0))
        transcript_source = str(getattr(metrics, "transcript_source", "unknown"))
        if self.stop_requested.is_set():
            return
        if not heard:
            print("Heard: (nothing)")
            self.diagnostics.no_input(mic_peak=mic_peak, mic_average=mic_average)
            if self._music_paused_for_turn:
                self.music_player.resume()
            self._music_paused_for_turn = False
            self.state.set(Reaction.IDLE, "Hold SPACE to talk")
            return
        print(f"Heard (not saved): {heard}")
        self.diagnostics.heard(
            heard,
            mic_peak=mic_peak,
            mic_average=mic_average,
            transcript_source=transcript_source,
        )
        if heard.lower() in {"quit", "exit"}:
            self.music_player.stop()
            self._say("Goodbye!", Reaction.HAPPY)
            self.stop_requested.set()
            return
        self.state.set(Reaction.THINKING, f"Heard: {heard[:60]}")
        routed_message = (
            "stop music"
            if self._music_paused_for_turn and heard.lower().strip() == "stop" and not self.session.game_active
            else heard
        )
        session_result = self.session.respond(routed_message)
        response = session_result.conversation
        music_actions = {
            Action.PLAY_MUSIC,
            Action.PAUSE_MUSIC,
            Action.RESUME_MUSIC,
            Action.NEXT_MUSIC,
            Action.STOP_MUSIC,
        }
        resume_music = self._music_paused_for_turn and response.command.action not in music_actions
        self.diagnostics.complete(
            route=session_result.route.value,
            action=response.command.action.value,
            reaction=response.command.reaction.value,
            reply=response.command.reply,
            provider_error=response.provider_error,
        )
        if response.command.action == Action.PLAY_MUSIC:
            self._handle_music(response.music_category)
        elif response.command.action in {
            Action.PAUSE_MUSIC,
            Action.RESUME_MUSIC,
            Action.NEXT_MUSIC,
            Action.STOP_MUSIC,
        }:
            self._handle_music_control(response.command.action)
        else:
            self._say(response.command.reply, response.command.reaction)
        if resume_music:
            self.music_player.resume()
        self._music_paused_for_turn = False
        self.state.set(Reaction.IDLE, "Hold SPACE to talk")

    def _handle_music(self, category: str | None) -> None:
        if self.music is None:
            self._say("I understood the music request, but the playlist is not configured.", Reaction.CONFUSED)
            return
        track = self.music.choose(requested_category=category)
        path = Path(track.path)
        path = path if path.is_absolute() else self.project_root / path
        if not path.is_file():
            self._say("I understood the music request, but no playable music file is installed yet.", Reaction.CONFUSED)
            return
        self._music_category = category or track.category
        self._say(f"Playing {track.title}.", Reaction.HAPPY)
        if not self.music_player.play(track, self.project_root):
            self._say("I found the track, but the audio player could not open it.", Reaction.CONFUSED)

    def _handle_music_control(self, action: Action) -> None:
        if action == Action.STOP_MUSIC:
            stopped = self.music_player.stop()
            self._say("Music stopped." if stopped else "No music is playing.", Reaction.OK if stopped else Reaction.CONFUSED)
            self._music_paused_for_turn = False
            return
        if action == Action.PAUSE_MUSIC:
            paused = self._music_paused_for_turn or self.music_player.pause()
            self._say("Music paused." if paused else "No music is playing.", Reaction.OK if paused else Reaction.CONFUSED)
            self._music_paused_for_turn = False
            return
        if action == Action.RESUME_MUSIC:
            self._say("Resuming the music.", Reaction.OK)
            resumed = self.music_player.resume()
            if not resumed:
                self._say("There is no paused music to resume.", Reaction.CONFUSED)
            self._music_paused_for_turn = False
            return
        if action == Action.NEXT_MUSIC:
            if not self.music_player.is_active or self.music is None:
                self._say("There is no active playlist to skip.", Reaction.CONFUSED)
            else:
                self._handle_music(self._music_category)
            self._music_paused_for_turn = False


def build_listener(args: argparse.Namespace) -> tuple[object, str]:
    if args.recognizer == "windows":
        try:
            return WindowsMicrophoneListener(), "Windows English recognition"
        except RuntimeError:
            raise
    listener = MicrophoneListener(args.speech_model, args.microphone)
    return listener, f"Vosk offline recognition ({listener.device_name})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full camera-and-voice SPIS robot demo.")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "gesture_knn.npz")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ollama-model", default="spis-robot")
    parser.add_argument("--speech-model", type=Path, default=DEFAULT_SPEECH_MODEL)
    parser.add_argument("--microphone", type=int, help="Optional Vosk microphone device number.")
    parser.add_argument("--recognizer", choices=("auto", "windows", "vosk"), default="vosk")
    parser.add_argument("--tts", choices=("piper", "windows"), default="piper")
    parser.add_argument("--piper-voice", type=Path, default=DEFAULT_PIPER_VOICE)
    parser.add_argument("--voice", default="auto", help="Windows fallback voice name; auto prefers a natural voice.")
    parser.add_argument("--voice-rate", type=int, default=4, help="Voice energy from -10 to 10.")
    parser.add_argument("--listen-seconds", type=float, default=12.0)
    parser.add_argument("--debug-camera", action="store_true", help="Show the camera diagnostics window at startup.")
    parser.add_argument(
        "--diagnostic-log",
        type=Path,
        help="Optional text-only JSONL turn log. Microphone audio and camera images are never written.",
    )
    parser.add_argument(
        "--peace-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "peace_reaction.wav",
        help="Local sound effect played once when the peace gesture activates.",
    )
    parser.add_argument(
        "--ok-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "ok_reaction.mp3",
        help="Local sound effect played once when the OK gesture activates.",
    )
    parser.add_argument(
        "--angry-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "angry_reaction.mp3",
        help="Local sound effect played once when the rude gesture activates.",
    )
    parser.add_argument(
        "--thumbs-up-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "thumbs_up_reaction.mp3",
        help="Local sound effect played once when the thumbs-up gesture activates.",
    )
    parser.add_argument(
        "--heart-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "heart_reaction.wav",
        help="Local sound effect played once when the heart gesture activates.",
    )
    parser.add_argument(
        "--mohan-sound",
        type=Path,
        default=ROOT / "assets" / "sounds" / "mohan_whistle.mp3",
        help="Local sound effect played once when the Mohan gesture activates.",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Show the robot face in a fullscreen laptop window.")
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Train the gesture model first: {args.model}")
    if args.recognizer == "windows":
        raise ValueError("Push-to-talk needs Vosk. Use --recognizer vosk --microphone 1 on this laptop.")

    gesture_model = GestureKNN.load(args.model)
    listener, recognizer_name = build_listener(args)
    provider = OllamaConversationProvider(args.ollama_model) if args.ollama_model else RuleConversationProvider()
    session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json")
    windows_speaker = LocalSpeaker(args.voice, args.voice_rate)
    if args.tts == "piper":
        try:
            speaker = FallbackSpeaker(PiperSpeaker(args.piper_voice), windows_speaker)
            tts_name = "Piper neural voice with Windows fallback"
        except (FileNotFoundError, RuntimeError) as error:
            print(f"Piper unavailable ({error}); using the Windows voice.")
            speaker = windows_speaker
            tts_name = "Windows voice"
    else:
        speaker = windows_speaker
        tts_name = "Windows voice"
    music = MusicSelector.from_file(ROOT / "assets" / "music" / "playlist.json")
    diagnostics = TurnDiagnostics(args.diagnostic_log)
    voice = VoiceWorker(listener, speaker, session, args.listen_seconds, music, diagnostics=diagnostics)

    gate = GestureGate(distance_limit=gesture_model.distance_limit)
    controller = RobotController()
    sound_player = SoundEffectPlayer()
    gesture_sound_paths = {
        "peace": args.peace_sound,
        "thumbs_up": args.thumbs_up_sound,
        "heart": args.heart_sound,
        "ok": args.ok_sound,
        "middle_finger": args.angry_sound,
        "mohan": args.mohan_sound,
    }
    gesture_feedback = GestureFeedback(
        hold_seconds={
            label: duration
            for label, path in gesture_sound_paths.items()
            if (duration := sound_player.duration_seconds(path)) > 0.0
        }
    )
    gesture_sounds = GestureSoundFeedback(
        sound_player,
        gesture_sound_paths,
    )
    space_key = SpaceKey()
    space_was_down = False
    previous_face: np.ndarray | None = None
    previous_key: tuple[Reaction, str] | None = None
    transition_started = time.monotonic()
    camera, first_frame, camera_backend = open_working_camera(args.camera)
    tracker = HandTracker(HAND_MODEL_PATH)
    print(f"Camera: {camera_backend}. Voice input: {recognizer_name}. Voice output: {tts_name}.")
    print("Hold SPACE to talk. Press D for camera diagnostics and Q to quit.")
    face_window = "SPIS Robot"
    camera_window = "SPIS Robot Camera"
    cv2.namedWindow(face_window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(face_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    voice.start()
    debug_camera_visible = args.debug_camera
    pending_frame: np.ndarray | None = first_frame
    failed_frames = 0
    gestures_were_locked = False

    try:
        while not voice.stop_requested.is_set():
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ok = True
            else:
                ok, frame = camera.read()
            if not ok:
                failed_frames += 1
                if failed_frames >= 20:
                    raise RuntimeError("The camera stopped delivering frames. Close other camera apps and restart the robot.")
                time.sleep(0.03)
                continue
            failed_frames = 0
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)

            space_is_down = space_key.is_down()
            if space_is_down and not space_was_down:
                voice.request_listening()
            elif space_was_down and not space_is_down:
                voice.release_listening()
            space_was_down = space_is_down

            voice_activity = voice.state.current()
            prediction = Prediction("none", 1.0, 0.0) if not hands else gesture_model.predict(landmarks_to_features(hands))
            if gestures_locked(voice_activity, voice.busy):
                if not gestures_were_locked:
                    gate.suspend()
                    gesture_sounds.stop()
                gestures_were_locked = True
                label = "none"
            else:
                if gestures_were_locked:
                    # Conversation cleared the previous candidate. A newly
                    # stable pose can now activate without leaving the frame.
                    gate.resume()
                    gestures_were_locked = False
                label = gate.update(prediction, len(hands))
            gesture_sounds.update(label)
            gesture_command = controller.from_gesture(label)
            now = time.monotonic()
            display_activity = gesture_feedback.choose(label, gesture_command, voice_activity, now)
            reaction, subtitle = display_activity.reaction, display_activity.subtitle

            draw_hands(frame, hands)
            cv2.putText(frame, f"Gesture: {label}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
            cv2.putText(frame, f"Robot: {reaction}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
            cv2.putText(frame, f"Voice: {voice_activity.reaction}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
            cv2.putText(frame, f"Hands: {len(hands)} | Hold SPACE to talk", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 0), 2)
            raw_status = "LOCKED" if gestures_were_locked else prediction.label
            cv2.putText(
                frame,
                f"Raw: {raw_status} | confidence {prediction.confidence:.0%} | distance {prediction.nearest_distance:.1f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
            )
            turn = diagnostics.current()
            cv2.putText(
                frame,
                f"Heard: {turn.heard[:58] or '-'}",
                (20, 202),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (160, 255, 190),
                2,
            )
            cv2.putText(
                frame,
                f"Route: {turn.route} | action: {turn.action} | mic peak: {turn.mic_peak:.0%}",
                (20, 232),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (160, 255, 190),
                2,
            )

            key = (reaction, subtitle)
            if key != previous_key:
                previous_key = key
                transition_started = now
            input_level = float(getattr(listener, "input_level", 0.0))
            status = "SPEAKING" if display_activity.speaking else reaction.value.upper()
            target = render_face(
                reaction,
                subtitle,
                time_seconds=now,
                speaking=display_activity.speaking,
                input_level=input_level,
                status=status,
                music_title=voice.music_player.current_title,
            )
            progress = min(1.0, (now - transition_started) / 0.28)
            face = target if previous_face is None else cv2.addWeighted(previous_face, 1.0 - progress, target, progress, 0)
            if progress >= 1.0:
                previous_face = target

            cv2.imshow(face_window, face)
            if debug_camera_visible:
                cv2.imshow(camera_window, frame)
            pressed = cv2.waitKey(1) & 0xFF
            if pressed == ord("d"):
                debug_camera_visible = not debug_camera_visible
                if not debug_camera_visible:
                    try:
                        cv2.destroyWindow(camera_window)
                    except cv2.error:
                        pass
            elif pressed == ord("q"):
                break
    finally:
        voice.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
