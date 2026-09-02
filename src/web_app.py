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
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

try:
    from .diagnostics import TurnDiagnostics
    from .gesture_features import landmarks_to_features
    from .gesture_gate import GestureGate
    from .gesture_model import GestureKNN, Prediction
    from .hand_tracker import HandTracker, draw_hands
    from .music import MusicSelector, WebMusicController
    from .robot_face import render_face
    from .robot_state import Reaction, RobotController
    from .voice_engine import VoiceEngine
except ImportError:  # Supports direct execution: python src/web_app.py
    from diagnostics import TurnDiagnostics
    from gesture_features import landmarks_to_features
    from gesture_gate import GestureGate
    from gesture_model import GestureKNN, Prediction
    from hand_tracker import HandTracker, draw_hands
    from music import MusicSelector, WebMusicController
    from robot_face import render_face
    from robot_state import Reaction, RobotController
    from voice_engine import VoiceEngine

ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
GESTURE_MODEL_PATH = ROOT / "model" / "gesture_knn.npz"
DEFAULT_VOSK_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
DEFAULT_PIPER_VOICE = ROOT / "models" / "voices" / "en_US-lessac-medium.onnx"
PLAYLIST_PATH = ROOT / "assets" / "music" / "playlist.json"
DEFAULT_MOHAN_SOUND = ROOT / "assets" / "sounds" / "mohan_whistle.mp3"
WEB_DIR = ROOT / "web"

# Gestures the reference card shows, in legend order. Replies/reactions are read
# live from RobotController so the card can never drift from the backend mapping.
# (The model still recognizes "wave"; it is just omitted from the UI legend.)
GESTURE_LABELS = ("thumbs_up", "peace", "stop", "heart", "ok", "middle_finger", "mohan")

JPEG_QUALITY = 80
FACE_WIDTH, FACE_HEIGHT = 800, 480
BOUNDARY = "frame"


class RobotWebRuntime:
    """Own the camera and drive one capture/inference/render loop in a thread.

    Latest encoded frames and state are published behind a condition variable so
    the MJPEG generators can block until a fresh frame exists instead of busy
    looping.
    """

    def __init__(self, camera_index: int) -> None:
        self._camera_index = camera_index
        self._model = GestureKNN.load(GESTURE_MODEL_PATH)
        self._gate = GestureGate(distance_limit=self._model.distance_limit)
        self._controller = RobotController()
        self._tracker: HandTracker | None = None

        self._condition = threading.Condition()
        self._camera_jpeg: bytes | None = None
        self._face_jpeg: bytes | None = None
        self._state = {"gesture": "none", "reaction": "idle", "confidence": 0.0}
        self._sequence = 0

        # Voice temporarily drives the face (listening/thinking/speaking) even
        # when no gesture is present. Stored as (reaction, subtitle, expiry).
        self._override: tuple[Reaction, str, float] | None = None

        # Lets the face show the now-playing title without the runtime owning
        # the music player. Wired to the shared controller after construction.
        self._music_title: Callable[[], str | None] = lambda: None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="robot-runtime", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def set_music_title_source(self, source: Callable[[], str | None]) -> None:
        self._music_title = source

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

    def _open_camera(self) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(self._camera_index)
        # isOpened() is not enough on macOS: a virtual/Continuity device can open
        # yet never deliver a frame. Require a real frame before trusting it.
        if camera.isOpened():
            for _ in range(30):
                ok, frame = camera.read()
                if ok and frame is not None and frame.size:
                    return camera
        camera.release()
        raise RuntimeError(
            f"Camera {self._camera_index} did not deliver video. Try another index "
            "(e.g. ./start --camera 1), close other apps using the webcam, and grant "
            "your terminal Camera access in System Settings > Privacy & Security > Camera."
        )

    def _publish(self, camera_jpeg: bytes, face_jpeg: bytes, state: dict) -> None:
        with self._condition:
            self._camera_jpeg = camera_jpeg
            self._face_jpeg = face_jpeg
            self._state = state
            self._sequence += 1
            self._condition.notify_all()

    def _run(self) -> None:
        camera = self._open_camera()
        self._tracker = HandTracker(HAND_MODEL_PATH)
        try:
            while not self._stop.is_set():
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
                face = render_face(
                    face_reaction,
                    face_subtitle,
                    FACE_WIDTH,
                    FACE_HEIGHT,
                    time_seconds=time.monotonic(),
                    music_title=self._music_title(),
                )

                camera_jpeg = _encode(frame)
                face_jpeg = _encode(face)
                if camera_jpeg is None or face_jpeg is None:
                    continue
                self._publish(
                    camera_jpeg,
                    face_jpeg,
                    {
                        "gesture": label,
                        "reaction": str(reaction),
                        "confidence": round(float(prediction.confidence), 2) if label not in {"none", "unknown"} else 0.0,
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


def gesture_catalog() -> list[dict]:
    """The 8 recognized gestures with their live reply + reaction mapping."""
    controller = RobotController()
    catalog = []
    for label in GESTURE_LABELS:
        command = controller.from_gesture(label)
        catalog.append(
            {"label": label, "reply": command.reply, "reaction": str(command.reaction)}
        )
    return catalog


def create_app(
    runtime: RobotWebRuntime,
    engine: VoiceEngine,
    music: WebMusicController,
    diagnostics: TurnDiagnostics,
    sounds: dict[str, Path] | None = None,
) -> Flask:
    sounds = sounds or {}
    app = Flask(__name__)

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
        return jsonify(
            {
                **runtime.snapshot_state(),
                "music": music.state(),
                "diagnostics": asdict(diagnostics.current()),
            }
        )

    @app.route("/gestures")
    def gestures() -> Response:
        catalog = gesture_catalog()
        for entry in catalog:
            path = sounds.get(entry["label"])
            entry["sound"] = f"/sound/{entry['label']}" if path is not None and path.is_file() else None
        return jsonify({"gestures": catalog})

    @app.route("/sound/<label>")
    def sound(label: str) -> Response:
        # Serve only sounds mapped to a gesture label; the browser plays them on
        # a separate <audio> so a gesture effect layers over any music.
        path = sounds.get(label)
        if path is None or not path.is_file():
            abort(404)
        return send_file(path, mimetype="audio/mpeg", conditional=True)

    @app.route("/music/state")
    def music_state() -> Response:
        return jsonify(music.state())

    @app.route("/music/play", methods=["POST"])
    def music_play() -> Response:
        payload = request.get_json(silent=True) or {}
        category = payload.get("category") or None
        return jsonify(music.select(category))

    @app.route("/music/next", methods=["POST"])
    def music_next() -> Response:
        return jsonify(music.skip())

    @app.route("/music/stop", methods=["POST"])
    def music_stop() -> Response:
        return jsonify(music.clear())

    @app.route("/music/track/<int:index>")
    def music_track(index: int) -> Response:
        # Serve only files named in the playlist (by index) so the browser can
        # stream them; never an arbitrary path. Range requests enable seeking.
        track = music.track_by_index(index)
        path = music.resolve_path(track) if track is not None else None
        if path is None:
            abort(404)
        return send_file(path, mimetype="audio/mpeg", conditional=True)

    @app.route("/voice/available")
    def voice_available() -> Response:
        return jsonify({"available": engine.available, "reason": engine.reason})

    @app.route("/voice/listening", methods=["POST"])
    def voice_listening() -> Response:
        # The browser ducks its own <audio> for the turn; the server only drives
        # the face here.
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
    parser.add_argument(
        "--mohan-sound",
        type=Path,
        default=DEFAULT_MOHAN_SOUND,
        help="Sound streamed to the browser when the Mohan gesture activates.",
    )
    args = parser.parse_args()

    _require_models()

    try:
        selector = MusicSelector.from_file(PLAYLIST_PATH) if PLAYLIST_PATH.exists() else None
    except (ValueError, OSError) as error:
        print(f"Playlist unavailable: {error}")
        selector = None
    music = WebMusicController(selector, ROOT)
    diagnostics = TurnDiagnostics()
    if music.available:
        print(f"Music enabled ({', '.join(music.categories())}), streamed to the browser.")
    else:
        print("Music disabled: playlist not configured.")

    engine = VoiceEngine(
        args.vosk_model,
        args.piper_voice,
        ollama_model=args.ollama_model,
        use_ollama=not args.no_ollama,
        music=music,
        diagnostics=diagnostics,
    )
    if engine.available:
        print("Voice enabled (hold-to-talk).")
    else:
        print(f"Voice disabled: {engine.reason}")

    sounds = {"mohan": args.mohan_sound}
    if args.mohan_sound.is_file():
        print(f"Mohan gesture sound: {args.mohan_sound.name} (browser).")
    else:
        print(f"Mohan gesture sound: not installed ({args.mohan_sound}).")

    runtime = RobotWebRuntime(args.camera)
    runtime.set_music_title_source(lambda: music.state()["title"])
    runtime.start()
    app = create_app(runtime, engine, music, diagnostics, sounds)
    print(f"SPIS robot web UI on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
