from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.game_trials import record_trial, summarize_trials


class GameTrialTests(unittest.TestCase):
    def test_recorded_trial_contains_normalized_turns_without_audio(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trials.jsonl"
            record_trial(
                path,
                target="laptop",
                turns=[{"question_id": "electronic", "question": "Is it electronic?", "answer": "yes"}],
                guess="laptop",
                confidence=0.92,
                outcome="guessed",
                trial_id="round-01",
                notes="The question was clear.",
                valid_for_training=True,
            )

            entry = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_trials(path)

        self.assertEqual(entry["turns"][0]["answer"], "yes")
        self.assertNotIn("audio", entry)
        self.assertNotIn("participant_name", entry)
        self.assertTrue(entry["valid_for_training"])
        self.assertEqual(summary, {"total": 1, "completed": 1, "correct": 1, "low_confidence": 0, "cancelled": 0})

    def test_rejects_an_unnormalized_trial_answer(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trials.jsonl"
            with self.assertRaises(ValueError):
                record_trial(
                    path,
                    target="laptop",
                    turns=[{"question_id": "electronic", "question": "Is it electronic?", "answer": "sure"}],
                    guess="laptop",
                    confidence=0.92,
                    outcome="guessed",
                )
