from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.game_trials import record_trial
from src.object_game import ObjectGuessingGame
from src.object_game_training import train_and_write


ROOT = Path(__file__).resolve().parents[1]


class ObjectGameTrainingTests(unittest.TestCase):
    def test_training_calibrates_only_questions_with_examples_on_both_sides(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            trials = directory / "trials.jsonl"
            calibration = directory / "calibration.json"
            report = directory / "report.json"
            for number in range(4):
                record_trial(
                    trials,
                    target="laptop",
                    turns=[{"question_id": "electronic", "question": "Is it electronic?", "answer": "yes"}],
                    guess="laptop",
                    confidence=0.9,
                    outcome="guessed",
                    trial_id=f"present-{number}",
                    valid_for_training=True,
                )
                record_trial(
                    trials,
                    target="book",
                    turns=[{"question_id": "electronic", "question": "Is it electronic?", "answer": "no"}],
                    guess="book",
                    confidence=0.9,
                    outcome="guessed",
                    trial_id=f"absent-{number}",
                    valid_for_training=True,
                )

            document, training_report = train_and_write(
                trials,
                ROOT / "data" / "object_catalog.json",
                calibration,
                report,
                min_per_side=2,
            )
            calibrated_game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json", calibration)
            self.assertTrue(calibration.exists())
            self.assertTrue(report.exists())
            self.assertIn("electronic", document["questions"])
            self.assertIn("electronic", calibrated_game.likelihoods_by_question)
            self.assertEqual(training_report["source_trials"], 8)

    def test_training_ignores_exploratory_trials(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            trials = directory / "trials.jsonl"
            calibration = directory / "calibration.json"
            report = directory / "report.json"
            record_trial(
                trials,
                target="laptop",
                turns=[{"question_id": "electronic", "question": "Is it electronic?", "answer": "no"}],
                guess="pen",
                confidence=0.9,
                outcome="guessed",
                trial_id="explore-01",
                valid_for_training=False,
            )

            document, training_report = train_and_write(
                trials,
                ROOT / "data" / "object_catalog.json",
                calibration,
                report,
            )

        self.assertEqual(document["source_trials"], 0)
        self.assertEqual(training_report["ignored_trials"], 1)

    def test_dry_run_does_not_write_training_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            trials = directory / "trials.jsonl"
            calibration = directory / "calibration.json"
            report = directory / "report.json"

            document, _ = train_and_write(
                trials,
                ROOT / "data" / "object_catalog.json",
                calibration,
                report,
                dry_run=True,
            )
            self.assertEqual(document["source_trials"], 0)
            self.assertFalse(calibration.exists())
            self.assertFalse(report.exists())
