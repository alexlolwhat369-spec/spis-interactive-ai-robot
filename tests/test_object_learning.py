from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.object_learning import record_suggestion


class ObjectLearningTests(unittest.TestCase):
    def test_records_a_small_reviewable_json_line(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "suggestions.jsonl"

            record_suggestion(path, "phone", "calculator", "Is it mainly used for calculations?")

            suggestion = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(suggestion["guessed_object"], "phone")
        self.assertEqual(suggestion["new_object"], "calculator")
        self.assertIn("calculations", suggestion["distinguishing_question"])

    def test_rejects_an_empty_correction(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "suggestions.jsonl"

            with self.assertRaises(ValueError):
                record_suggestion(path, "phone", "", "Is it mainly used for calculations?")
