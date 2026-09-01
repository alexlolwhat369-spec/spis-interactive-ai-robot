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
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .music import MusicPlayer, MusicSelector
    from .robot_face import render_face
    from .robot_runtime import GAME_ANSWER_PHRASES, MUSIC_CATEGORY_PHRASES, RobotDialogueSession
    from .robot_state import Action, Reaction, RobotCommand, RobotController
    from .push_to_talk import SpaceKey
    from .speech import FallbackSpeaker, LocalSpeaker, PiperSpeaker
    from .speech_input import MicrophoneListener, WindowsMicrophoneListener
except ImportError:  # Supports direct execution: python src/interactive_robot.py
    from camera_io import open_working_camera
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from music import MusicPlayer, MusicSelector
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
    """Keep special visual reactions visible long enough to be noticed."""

    def __init__(self, heart_hold_seconds: float = 1.5, mohan_hold_seconds: float = 2.0) -> None:
        self.hold_seconds = {"heart": heart_hold_seconds, "mohan": mohan_hold_seconds}
        self._held_until = 0.0
        self._held_activity: VoiceActivity | None = None

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
            return voice_activity
        if label in self.hold_seconds:
            self._held_until = now + self.hold_seconds[label]
            self._held_activity = VoiceActivity(gesture_command.reaction, gesture_command.reply)
            return self._held_activity
        if now < self._held_until and self._held_activity is not None:
            return self._held_activity
        self._held_activity = None
        if label != "none":
            return VoiceActivity(gesture_command.reaction, gesture_command.reply)
        return voice_activity


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
    ) -> None:
        self.listener = listener
        self.speaker = speaker
        self.session = session
        self.listen_seconds = listen_seconds
        self.music = music
        self.project_root = project_root
        self.music_player = music_player or MusicPlayer()
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
        if self.session.expects_game_answer and not getattr(self.session, "supports_semantic_game_input", False):
            phrases = GAME_ANSWER_PHRASES
        elif getattr(self.session, "expects_music_category", False):
            phrases = MUSIC_CATEGORY_PHRASES
        else:
            phrases = None
        heard = self.listener.listen_once(self.listen_seconds, self._release_listening, phrases=phrases)
        if self.stop_requested.is_set():
            return
        if not heard:
            print("Heard: (nothing)")
            if self._music_paused_for_turn:
                self.music_player.resume()
            self._music_paused_for_turn = False
            self.state.set(Reaction.IDLE, "Hold SPACE to talk")
            return
        print(f"Heard (not saved): {heard}")
        if heard.lower() in {"quit", "exit"}:
            self.music_player.stop()
            self._say("Goodbye!", Reaction.HAPPY)
            self.stop_requested.set()
            return
        self.state.set(Reaction.THINKING, f"Heard: {heard[:60]}")
        response = self.session.respond(heard).conversation
        resume_music = self._music_paused_for_turn and response.command.action not in {Action.PLAY_MUSIC, Action.STOP}
        if response.command.action == Action.PLAY_MUSIC:
            self._handle_music(response.music_category)
        elif response.command.action == Action.STOP and self._music_paused_for_turn:
            self.music_player.stop()
            self._say("Music stopped.", Reaction.OK)
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
        self._say(f"Playing {track.title}.", Reaction.HAPPY)
        if not self.music_player.play(track, self.project_root):
            self._say("I found the track, but the audio player could not open it.", Reaction.CONFUSED)


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
    voice = VoiceWorker(listener, speaker, session, args.listen_seconds, music)

    gate = GestureGate(distance_limit=gesture_model.distance_limit)
    controller = RobotController()
    gesture_feedback = GestureFeedback()
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
                gestures_were_locked = True
                label = "none"
            else:
                if gestures_were_locked:
                    # Conversation cleared the previous candidate. A newly
                    # stable pose can now activate without leaving the frame.
                    gate.resume()
                    gestures_were_locked = False
                label = gate.update(prediction, len(hands))
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
