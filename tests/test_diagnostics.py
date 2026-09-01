from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.diagnostics import TurnDiagnostics


class TurnDiagnosticsTests(unittest.TestCase):
    def test_snapshot_explains_one_completed_turn(self) -> None:
        diagnostics = TurnDiagnostics()
        diagnostics.begin()
        diagnostics.heard("play music", mic_peak=0.7, mic_average=0.3, transcript_source="final")
        diagnostics.complete(
            route="music_request",
            action="none",
            reaction="listening",
            reply="Which category?",
        )

        snapshot = diagnostics.current()

        self.assertEqual(snapshot.heard, "play music")
        self.assertEqual(snapshot.route, "music_request")
        self.assertEqual(snapshot.mic_peak, 0.7)
        self.assertEqual(snapshot.transcript_source, "final")

    def test_optional_log_contains_text_but_no_audio_or_camera_data(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "turns.jsonl"
            diagnostics = TurnDiagnostics(path)
            diagnostics.begin()
            diagnostics.heard("hello")
            diagnostics.complete(
                route="conversation",
                action="none",
                reaction="happy",
                reply="Hello!",
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["heard"], "hello")
        self.assertNotIn("audio", record)
        self.assertNotIn("frame", record)
