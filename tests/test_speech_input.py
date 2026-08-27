from __future__ import annotations

import unittest

from src.speech_input import WindowsMicrophoneListener, text_from_result


class SpeechInputTests(unittest.TestCase):
    def test_extracts_transcript_from_vosk_result(self) -> None:
        self.assertEqual(text_from_result('{"text": "hello robot"}'), "hello robot")

    def test_invalid_or_missing_text_returns_empty_string(self) -> None:
        self.assertEqual(text_from_result("not json"), "")
        self.assertEqual(text_from_result('{"partial": "hello"}'), "")

    def test_windows_listener_can_be_constructed_on_windows(self) -> None:
        listener = WindowsMicrophoneListener()
        self.assertIsNotNone(listener)
