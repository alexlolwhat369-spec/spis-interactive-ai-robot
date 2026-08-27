"""Talk with the robot using a local microphone, Vosk, Ollama, and local speech."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2

try:
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .robot_runtime import RobotDialogueSession
    from .robot_state import Reaction
    from .speech import LocalSpeaker, PiperSpeaker
    from .speech_input import MicrophoneListener, WindowsMicrophoneListener
    from .robot_face import render_face
except ImportError:  # Supports direct execution: python src/voice_demo.py
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from robot_runtime import RobotDialogueSession
    from robot_state import Reaction
    from speech import LocalSpeaker, PiperSpeaker
    from speech_input import MicrophoneListener, WindowsMicrophoneListener
    from robot_face import render_face


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEECH_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
DEFAULT_PIPER_VOICE = ROOT / "models" / "voices" / "en_US-lessac-medium.onnx"


class FaceWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.open = enabled
        if enabled:
            cv2.namedWindow("SPIS Robot", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SPIS Robot", 800, 480)

    def show(self, reaction: Reaction, subtitle: str) -> bool:
        if not self.open:
            return True
        frame = render_face(reaction, subtitle, time_seconds=time.monotonic())
        cv2.imshow("SPIS Robot", frame)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            self.open = False
            return False
        return True

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow("SPIS Robot")


def speak_with_face(face: FaceWindow, speaker: object, text: str, reaction: Reaction = Reaction.SPEAKING) -> bool:
    worker = threading.Thread(target=speaker.speak, args=(text, reaction), daemon=True)
    worker.start()
    while worker.is_alive():
        if not face.show(Reaction.SPEAKING, text):
            return False
    worker.join()
    return face.show(Reaction.IDLE, "")


def listen_with_face(face: FaceWindow, listener: object, seconds: float) -> tuple[str, bool]:
    heard = ""

    def listen() -> None:
        nonlocal heard
        heard = listener.listen_once(seconds)

    worker = threading.Thread(target=listen, daemon=True)
    worker.start()
    while worker.is_alive():
        if not face.show(Reaction.LISTENING, "Listening..."):
            return "", False
    worker.join()
    return heard, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk with the robot without typing.")
    parser.add_argument("--ollama-model", default="spis-robot", help="Installed Ollama model name.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SPEECH_MODEL, help="Folder containing the Vosk speech model.")
    parser.add_argument("--microphone", type=int, help="Optional input-device number; omit for the Windows default.")
    parser.add_argument("--tts", choices=("piper", "windows"), default="piper", help="Speech engine; Piper is the natural local voice.")
    parser.add_argument("--piper-voice", type=Path, default=DEFAULT_PIPER_VOICE)
    parser.add_argument("--voice", default="Microsoft Zira Desktop", help="Windows fallback voice name.")
    parser.add_argument("--voice-rate", type=int, default=4, help="Voice energy from -10 to 10; 4 is upbeat.")
    parser.add_argument("--recognizer", choices=("auto", "windows", "vosk"), default="auto", help="Speech recognizer; auto uses Windows on a laptop and Vosk elsewhere.")
    parser.add_argument("--listen-seconds", type=float, default=12.0, help="Maximum wait for one spoken turn.")
    parser.add_argument("--list-microphones", action="store_true", help="Print input devices and exit.")
    parser.add_argument("--no-face", action="store_true", help="Run speech without opening the robot face window.")
    args = parser.parse_args()

    if args.list_microphones:
        for index, name in MicrophoneListener.input_devices():
            print(f"{index}: {name}")
        return

    provider = OllamaConversationProvider(args.ollama_model) if args.ollama_model else RuleConversationProvider()
    session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json")
    speaker = (
        PiperSpeaker(args.piper_voice)
        if args.tts == "piper"
        else LocalSpeaker(args.voice, args.voice_rate)
    )
    face = FaceWindow(enabled=not args.no_face)
    use_windows_recognizer = args.recognizer == "windows" or (
        args.recognizer == "auto" and args.microphone is None
    )
    if use_windows_recognizer:
        try:
            listener = WindowsMicrophoneListener()
            recognizer_name = "Windows English recognition"
        except RuntimeError:
            if args.recognizer == "windows":
                raise
            listener = MicrophoneListener(args.model_path, args.microphone)
            recognizer_name = "Vosk offline recognition"
    else:
        listener = MicrophoneListener(args.model_path, args.microphone)
        recognizer_name = "Vosk offline recognition"

    greeting = "Hello! I am ready to talk. Say stop when you want me to pause."
    print(f"Robot: {greeting} ({recognizer_name})")
    if not speak_with_face(face, speaker, greeting, Reaction.HAPPY):
        face.close()
        return
    while face.open:
        print("Listening...")
        heard, keep_running = listen_with_face(face, listener, args.listen_seconds)
        if not keep_running:
            break
        if not heard:
            reply = "I did not hear you. Please try again."
        elif heard.lower() in {"quit", "exit"}:
            reply = "Goodbye!"
            print(f"You: {heard}\nRobot: {reply}")
            speak_with_face(face, speaker, reply, Reaction.HAPPY)
            face.close()
            return
        else:
            print(f"You: {heard}")
            result = session.respond(heard).conversation
            reply = result.command.reply
            if result.provider_error:
                print("Ollama was unavailable, so I used the local fallback.")
        print(f"Robot: {reply}")
        reaction = Reaction.CONFUSED if not heard else result.command.reaction
        if not speak_with_face(face, speaker, reply, reaction):
            break
    face.close()


if __name__ == "__main__":
    main()
