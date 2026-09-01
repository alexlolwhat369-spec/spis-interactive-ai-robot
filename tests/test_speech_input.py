from __future__ import annotations

import unittest
import queue
import threading

from src.speech_input import (
    MicrophoneListener,
    WindowsMicrophoneListener,
    partial_text_from_result,
    recognizer_vocabulary,
    text_from_result,
)


class SpeechInputTests(unittest.TestCase):
    def test_extracts_transcript_from_vosk_result(self) -> None:
        self.assertEqual(text_from_result('{"text": "hello robot"}'), "hello robot")

    def test_invalid_or_missing_text_returns_empty_string(self) -> None:
        self.assertEqual(text_from_result("not json"), "")
        self.assertEqual(text_from_result('{"partial": "hello"}'), "")

    def test_extracts_an_unfinished_partial_phrase(self) -> None:
        self.assertEqual(partial_text_from_result('{"partial": "play energetic"}'), "play energetic")

    def test_windows_listener_can_be_constructed_on_windows(self) -> None:
        listener = WindowsMicrophoneListener()
        self.assertIsNotNone(listener)

    def test_game_vocabulary_is_normalized_and_keeps_unknown_words(self) -> None:
        self.assertEqual(
            recognizer_vocabulary((" Probably ", "maybe", "probably")),
            '["probably", "maybe", "[unk]"]',
        )

    def test_release_still_processes_the_last_queued_audio_block(self) -> None:
        class FakeRecognizer:
            def __init__(self, *args: object) -> None:
                self.arguments = args
                self.accepted: list[bytes] = []

            def AcceptWaveform(self, audio: bytes) -> bool:
                self.accepted.append(audio)
                return False

            def FinalResult(self) -> str:
                return '{"text": "probably not"}' if self.accepted else '{"text": ""}'

        class FakeStream:
            def __init__(self, callback: object, options: dict[str, object]) -> None:
                self.callback = callback
                self.options = options

            def __enter__(self) -> "FakeStream":
                self.callback(b"final audio", 0, None, None)
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeSoundDevice:
            def __init__(self) -> None:
                self.options: dict[str, object] = {}

            def RawInputStream(self, **options: object) -> FakeStream:
                self.options = options
                return FakeStream(options["callback"], options)

        listener = MicrophoneListener.__new__(MicrophoneListener)
        listener._sounddevice = FakeSoundDevice()
        listener._recognizer_type = FakeRecognizer
        listener.model = object()
        listener.sample_rate = 16000
        listener.device = 1
        listener._audio = queue.Queue()
        listener.release_grace_seconds = 0.0
        released = threading.Event()
        released.set()

        transcript = listener.listen_once(1.0, released, phrases=("probably not", "no"))

        self.assertEqual(transcript, "probably not")
        self.assertEqual(listener._sounddevice.options["blocksize"], 800)

    def test_best_partial_is_used_when_vosk_has_no_final_text(self) -> None:
        class FakeRecognizer:
            def __init__(self, *args: object) -> None:
                pass

            def AcceptWaveform(self, audio: bytes) -> bool:
                del audio
                return False

            def PartialResult(self) -> str:
                return '{"partial": "stop the music"}'

            def FinalResult(self) -> str:
                return '{"text": ""}'

        class FakeStream:
            def __init__(self, callback: object) -> None:
                self.callback = callback

            def __enter__(self) -> "FakeStream":
                self.callback(bytes((0, 0)), 1, None, None)
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeSoundDevice:
            def RawInputStream(self, **options: object) -> FakeStream:
                return FakeStream(options["callback"])

        listener = MicrophoneListener.__new__(MicrophoneListener)
        listener._sounddevice = FakeSoundDevice()
        listener._recognizer_type = FakeRecognizer
        listener.model = object()
        listener.sample_rate = 16000
        listener.device = 1
        listener._audio = queue.Queue()
        listener.release_grace_seconds = 0.0
        listener.input_level = 0.0
        released = threading.Event()
        released.set()

        self.assertEqual(listener.listen_once(1.0, released), "stop the music")
