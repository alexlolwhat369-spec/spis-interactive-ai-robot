"""Serve the camera feed and the animated robot face in a browser.

This is the same pipeline as ``live_demo.py`` (camera -> hand tracking ->
gesture KNN -> robot reaction -> animated face), but instead of two OpenCV
windows it JPEG-encodes both frames into shared buffers and streams them as
MJPEG to any browser tabs. A single background thread owns the camera, so many
viewers share one capture without fighting over the device.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from .camera_io import open_working_camera
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .music import SoundEffectPlayer
    from .robot_face import render_face
    from .robot_state import Reaction, RobotController
    from .voice_engine import VoiceEngine
except ImportError:  # Supports direct execution: python src/web_app.py
    from camera_io import open_working_camera
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from music import SoundEffectPlayer
    from robot_face import render_face
    from robot_state import Reaction, RobotController
    from voice_engine import VoiceEngine

ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
GESTURE_MODEL_PATH = ROOT / "model" / "gesture_knn.npz"
DEFAULT_VOSK_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
DEFAULT_PIPER_VOICE = ROOT / "models" / "voices" / "en_US-lessac-medium.onnx"
WEB_DIR = ROOT / "web"
DEFAULT_MOHAN_SOUND = ROOT / "assets" / "sounds" / "mohan_whistle.mp3"

JPEG_QUALITY = 80
FACE_WIDTH, FACE_HEIGHT = 800, 480
BOUNDARY = "frame"


class RobotWebRuntime:
    """Own the camera and drive one capture/inference/render loop in a thread.

    Latest encoded frames and state are published behind a condition variable so
    the MJPEG generators can block until a fresh frame exists instead of busy
    looping.
    """

    def __init__(
        self,
        camera_index: int,
        mohan_sound: Path = DEFAULT_MOHAN_SOUND,
        sound_player: SoundEffectPlayer | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._model = GestureKNN.load(GESTURE_MODEL_PATH)
        self._gate = GestureGate(distance_limit=self._model.distance_limit)
        self._controller = RobotController()
        self._tracker: HandTracker | None = None

        self._condition = threading.Condition()
        self._camera_jpeg: bytes | None = None
        self._face_jpeg: bytes | None = None
        self._state = {
            "gesture": "none",
            "reaction": "idle",
            "confidence": 0.0,
            "camera_backend": "starting",
        }
        self._sequence = 0
        self._camera_backend = "starting"
        self._mohan_sound = mohan_sound
        self._sound_player = sound_player or SoundEffectPlayer()
        self._previous_sound_label = "none"

        # Voice temporarily drives the face (listening/thinking/speaking) even
        # when no gesture is present. Stored as (reaction, subtitle, expiry).
        self._override: tuple[Reaction, str, float] | None = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="robot-runtime", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def set_override(self, reaction: Reaction, subtitle: str, ttl: float) -> None:
        with self._condition:
            self._override = (reaction, subtitle, time.monotonic() + ttl)

    def clear_override(self) -> None:
        with self._condition:
            self._override = None

    def current_override(self) -> tuple[Reaction, str] | None:
        with self._condition:
            if self._override is None:
                return None
            reaction, subtitle, expiry = self._override
            if time.monotonic() >= expiry:
                self._override = None
                return None
            return reaction, subtitle

    def _open_camera(self) -> tuple[cv2.VideoCapture, np.ndarray, str]:
        return open_working_camera(self._camera_index)

    def _update_gesture_sound(self, label: str) -> None:
        if label == "mohan" and label != self._previous_sound_label:
            if not self._sound_player.play(self._mohan_sound):
                print(f"Mohan sound unavailable: {self._mohan_sound}")
        self._previous_sound_label = label

    def _publish(self, camera_jpeg: bytes, face_jpeg: bytes, state: dict) -> None:
        with self._condition:
            self._camera_jpeg = camera_jpeg
            self._face_jpeg = face_jpeg
            self._state = state
            self._sequence += 1
            self._condition.notify_all()

    def _run(self) -> None:
        camera, pending_frame, self._camera_backend = self._open_camera()
        self._tracker = HandTracker(HAND_MODEL_PATH)
        try:
            while not self._stop.is_set():
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, frame = camera.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                hand_samples = self._tracker.detect(frame)
                if not hand_samples:
                    prediction = Prediction("none", 1.0, 0.0)
                else:
                    prediction = self._model.predict(landmarks_to_features(hand_samples))
                label = self._gate.update(prediction, len(hand_samples))
                command = self._controller.from_gesture(label)
                reaction = command.reaction
                self._update_gesture_sound(label)

                draw_hands(frame, hand_samples)
                cv2.putText(frame, f"Gesture: {label}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
                cv2.putText(frame, f"Robot: {reaction}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
                if label not in {"none", "unknown"}:
                    cv2.putText(
                        frame,
                        f"Confidence: {prediction.confidence:.0%}",
                        (20, 106),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 220, 0),
                        2,
                    )

                # Voice (listening/thinking/speaking) takes over the face when active;
                # otherwise the gesture reaction drives it.
                override = self.current_override()
                if override is not None:
                    face_reaction, face_subtitle = override
                else:
                    face_reaction, face_subtitle = reaction, command.reply
                face = render_face(face_reaction, face_subtitle, FACE_WIDTH, FACE_HEIGHT, time_seconds=time.monotonic())

                camera_jpeg = _encode(frame)
                face_jpeg = _encode(face)
                if camera_jpeg is None or face_jpeg is None:
                    continue
                self._publish(
                    camera_jpeg,
                    face_jpeg,
                    {
                        "gesture": label,
                        "reaction": str(face_reaction),
                        "confidence": round(float(prediction.confidence), 2) if label not in {"none", "unknown"} else 0.0,
                        "camera_backend": self._camera_backend,
                    },
                )
        finally:
            if self._tracker is not None:
                self._tracker.close()
            camera.release()

    def stream(self, kind: str):
        """Yield an MJPEG part each time a new frame is published."""
        last_seen = -1
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._sequence != last_seen or self._stop.is_set(), timeout=5.0)
                if self._stop.is_set():
                    return
                last_seen = self._sequence
                payload = self._camera_jpeg if kind == "camera" else self._face_jpeg
            if payload is None:
                continue
            yield (
                b"--" + BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                + payload + b"\r\n"
            )

    def snapshot_state(self) -> dict:
        with self._condition:
            return dict(self._state)


def _encode(frame: np.ndarray) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def create_app(runtime: RobotWebRuntime, engine: VoiceEngine) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    @app.route("/")
    def index() -> Response:
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/camera.mjpg")
    def camera_feed() -> Response:
        return Response(runtime.stream("camera"), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/face.mjpg")
    def face_feed() -> Response:
        return Response(runtime.stream("face"), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/state")
    def state() -> Response:
        return jsonify(runtime.snapshot_state())

    @app.route("/voice/available")
    def voice_available() -> Response:
        return jsonify({"available": engine.available, "reason": engine.reason})

    @app.route("/voice/listening", methods=["POST"])
    def voice_listening() -> Response:
        engine.begin_turn()
        runtime.set_override(Reaction.LISTENING, "Listening...", ttl=20.0)
        return ("", 204)

    @app.route("/voice", methods=["POST"])
    def voice() -> Response:
        if not engine.available:
            return jsonify({"error": engine.reason or "Voice is unavailable."}), 503
        runtime.set_override(Reaction.THINKING, "", ttl=30.0)
        result = engine.handle(request.get_data())
        runtime.set_override(Reaction(result["reaction"]), result["reply"], ttl=result["audio_seconds"] + 0.5)
        return jsonify(result)

    @app.route("/voice/done", methods=["POST"])
    def voice_done() -> Response:
        engine.finish_turn()
        runtime.clear_override()
        return ("", 204)

    return app


def _require_models() -> None:
    if not HAND_MODEL_PATH.exists():
        sys.exit(f"Hand model missing: {HAND_MODEL_PATH}\nRun ./setup first.")
    if not GESTURE_MODEL_PATH.exists():
        sys.exit(f"Gesture model missing: {GESTURE_MODEL_PATH}\nRun ./train-gestures first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SPIS robot camera + face in a browser.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ollama-model", default="spis-robot", help="Ollama model for voice replies (falls back to rules if offline).")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama and always use the built-in rule replies.")
    parser.add_argument("--vosk-model", type=Path, default=DEFAULT_VOSK_MODEL)
    parser.add_argument("--piper-voice", type=Path, default=DEFAULT_PIPER_VOICE)
    parser.add_argument("--mohan-sound", type=Path, default=DEFAULT_MOHAN_SOUND)
    args = parser.parse_args()

    _require_models()

    engine = VoiceEngine(
        args.vosk_model,
        args.piper_voice,
        ollama_model=args.ollama_model,
        use_ollama=not args.no_ollama,
    )
    if engine.available:
        print("Voice enabled (hold-to-talk).")
    else:
        print(f"Voice disabled: {engine.reason}")

    runtime = RobotWebRuntime(args.camera, args.mohan_sound)
    runtime.start()
    app = create_app(runtime, engine)
    print(f"SPIS robot web UI on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        engine.stop()
        runtime.stop()


if __name__ == "__main__":
    main()
