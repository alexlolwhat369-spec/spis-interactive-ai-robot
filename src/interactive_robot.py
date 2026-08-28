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
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .robot_face import render_face
    from .robot_runtime import RobotDialogueSession
    from .robot_state import Reaction, RobotCommand, RobotController
    from .push_to_talk import SpaceKey
    from .speech import LocalSpeaker, PiperSpeaker
    from .speech_input import MicrophoneListener, WindowsMicrophoneListener
except ImportError:  # Supports direct execution: python src/interactive_robot.py
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from robot_face import render_face
    from robot_runtime import RobotDialogueSession
    from robot_state import Reaction, RobotCommand, RobotController
    from push_to_talk import SpaceKey
    from speech import LocalSpeaker, PiperSpeaker
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


def gestures_locked(activity: VoiceActivity, voice_busy: bool = False) -> bool:
    """Conversation always wins over an accidental camera gesture."""
    return voice_busy or activity.reaction in GESTURE_LOCKING_REACTIONS


class VoiceState:
    def __init__(self) -> None:
        self._activity = VoiceActivity(Reaction.IDLE)
        self._lock = threading.Lock()

    def set(self, reaction: Reaction, subtitle: str = "") -> None:
        with self._lock:
            self._activity = VoiceActivity(reaction, subtitle)

    def current(self) -> VoiceActivity:
        with self._lock:
            return self._activity


class GestureFeedback:
    """Keep heart visible long enough for a visitor to notice the response."""

    def __init__(self, heart_hold_seconds: float = 1.5) -> None:
        self.heart_hold_seconds = heart_hold_seconds
        self._heart_until = 0.0
        self._heart_subtitle = ""

    def choose(
        self,
        label: str,
        gesture_command: RobotCommand,
        voice_activity: VoiceActivity,
        now: float,
    ) -> VoiceActivity:
        if gestures_locked(voice_activity):
            self._heart_until = 0.0
            self._heart_subtitle = ""
            return voice_activity
        if label == "heart":
            self._heart_until = now + self.heart_hold_seconds
            self._heart_subtitle = gesture_command.reply
            return VoiceActivity(Reaction.HEART, self._heart_subtitle)
        if now < self._heart_until:
            return VoiceActivity(Reaction.HEART, self._heart_subtitle)
        if label != "none":
            return VoiceActivity(gesture_command.reaction, gesture_command.reply)
        return voice_activity


class VoiceWorker:
    def __init__(self, listener: object, speaker: object, session: RobotDialogueSession, listen_seconds: float) -> None:
        self.listener = listener
        self.speaker = speaker
        self.session = session
        self.listen_seconds = listen_seconds
        self.state = VoiceState()
        self.stop_requested = threading.Event()
        self._release_listening = threading.Event()
        self._busy = threading.Event()

    def start(self) -> None:
        self._start_task(self._greet)

    def stop(self) -> None:
        self.stop_requested.set()
        self._release_listening.set()

    def request_listening(self) -> bool:
        """Start one listener turn only when the visitor presses Space."""
        if self.stop_requested.is_set() or self._busy.is_set():
            return False
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
        self.state.set(Reaction.SPEAKING, text)
        self.speaker.speak(text, delivery_reaction)

    def _greet(self) -> None:
        self._say("Hello! I am ready to talk and see your gestures.", Reaction.HAPPY)
        self.state.set(Reaction.IDLE, "Hold SPACE to talk")

    def _listen_and_respond(self) -> None:
        self.state.set(Reaction.LISTENING, "Hold SPACE and speak")
        heard = self.listener.listen_once(self.listen_seconds, self._release_listening)
        if self.stop_requested.is_set():
            return
        if not heard:
            self.state.set(Reaction.IDLE, "Hold SPACE to talk")
            return
        if heard.lower() in {"quit", "exit"}:
            self._say("Goodbye!", Reaction.HAPPY)
            self.stop_requested.set()
            return
        self.state.set(Reaction.THINKING, "Thinking...")
        response = self.session.respond(heard).conversation
        self._say(response.command.reply, response.command.reaction)
        self.state.set(Reaction.IDLE, "Hold SPACE to talk")


def build_listener(args: argparse.Namespace) -> tuple[object, str]:
    if args.recognizer == "windows":
        try:
            return WindowsMicrophoneListener(), "Windows English recognition"
        except RuntimeError:
            raise
    return MicrophoneListener(args.speech_model, args.microphone), "Vosk offline recognition"


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
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Train the gesture model first: {args.model}")
    if args.recognizer == "windows":
        raise ValueError("Push-to-talk needs Vosk. Use --recognizer vosk --microphone 1 on this laptop.")

    gesture_model = GestureKNN.load(args.model)
    listener, recognizer_name = build_listener(args)
    provider = OllamaConversationProvider(args.ollama_model) if args.ollama_model else RuleConversationProvider()
    session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json")
    speaker = PiperSpeaker(args.piper_voice) if args.tts == "piper" else LocalSpeaker(args.voice, args.voice_rate)
    voice = VoiceWorker(listener, speaker, session, args.listen_seconds)

    gate = GestureGate(distance_limit=gesture_model.distance_limit)
    controller = RobotController()
    gesture_feedback = GestureFeedback()
    space_key = SpaceKey()
    space_was_down = False
    previous_face: np.ndarray | None = None
    previous_key: tuple[Reaction, str] | None = None
    transition_started = time.monotonic()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")
    tracker = HandTracker(HAND_MODEL_PATH)
    print(f"Voice input: {recognizer_name}. Hold SPACE to talk. Press Q in either window to quit.")
    voice.start()

    try:
        while not voice.stop_requested.is_set():
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)

            space_is_down = space_key.is_down()
            if space_is_down and not space_was_down:
                voice.request_listening()
            elif space_was_down and not space_is_down:
                voice.release_listening()
            space_was_down = space_is_down

            voice_activity = voice.state.current()
            if gestures_locked(voice_activity, voice.busy):
                # A held gesture must be released after a conversation before
                # it can affect the robot again.
                gate.suspend()
                label = "none"
            else:
                prediction = Prediction("none", 1.0, 0.0) if not hands else gesture_model.predict(landmarks_to_features(hands))
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

            key = (reaction, subtitle)
            if key != previous_key:
                previous_key = key
                transition_started = now
            target = render_face(reaction, subtitle, time_seconds=now)
            progress = min(1.0, (now - transition_started) / 0.28)
            face = target if previous_face is None else cv2.addWeighted(previous_face, 1.0 - progress, target, progress, 0)
            if progress >= 1.0:
                previous_face = target

            cv2.imshow("SPIS Robot Face", face)
            cv2.imshow("SPIS Robot Camera - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        voice.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
