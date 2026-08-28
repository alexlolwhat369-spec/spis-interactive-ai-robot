from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.conversation import ConversationResult, RuleConversationProvider
from src.object_game import ObjectGuessingGame, ObjectProfile, default_likelihoods
from src.robot_runtime import RobotDialogueSession, answer_from_text
from src.robot_state import Action, Reaction, RobotCommand


ROOT = Path(__file__).resolve().parents[1]


class RobotRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.trial_log = Path(self.temporary_directory.name) / "trials.jsonl"
        self.session = RobotDialogueSession(
            RuleConversationProvider(), ROOT / "data" / "object_catalog.json", trial_log_path=self.trial_log
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_game_request_opens_a_question_in_the_same_session(self) -> None:
        result = self.session.respond("Can we play a guessing game?")

        self.assertTrue(result.game_active)
        self.assertEqual(result.conversation.command.action, Action.START_GAME)
        self.assertEqual(result.conversation.command.reaction, Reaction.LISTENING)
        self.assertIn("yes, probably, maybe, probably not, or no", result.conversation.command.reply)

    def test_game_always_keeps_the_robot_as_the_guesser(self) -> None:
        class RoleReversingProvider:
            def respond(self, message: str) -> ConversationResult:
                return ConversationResult(
                    RobotCommand("Try to guess the object I am thinking of.", Reaction.SPEAKING, Action.START_GAME)
                )

        session = RobotDialogueSession(
            RoleReversingProvider(), ROOT / "data" / "object_catalog.json", trial_log_path=self.trial_log
        )
        result = session.respond("Start a game")

        self.assertIn("I will ask questions and try to guess your object", result.conversation.command.reply)
        self.assertNotIn("guess the object I am thinking", result.conversation.command.reply)

    def test_game_can_be_cancelled(self) -> None:
        self.session.respond("Play a game")
        result = self.session.respond("stop")

        self.assertFalse(result.game_active)
        self.assertEqual(result.conversation.command.action, Action.STOP)

    def test_short_answers_are_normalized(self) -> None:
        self.assertEqual(answer_from_text("Yep"), "yes")
        self.assertEqual(answer_from_text("Nope"), "no")
        self.assertEqual(answer_from_text("Correct!"), "yes")
        self.assertEqual(answer_from_text("Sure, that is right"), "yes")
        self.assertEqual(answer_from_text("Nah"), "no")
        self.assertEqual(answer_from_text("I guess no"), "probably_not")
        self.assertEqual(answer_from_text("I think not"), "probably_not")
        self.assertEqual(answer_from_text("I am not sure"), "maybe")
        self.assertEqual(answer_from_text("No idea"), "maybe")
        self.assertEqual(answer_from_text("probably"), "probably")
        self.assertEqual(answer_from_text("probably not"), "probably_not")

    def test_correct_guess_finishes_the_game(self) -> None:
        self.session.game = ObjectGuessingGame(
            [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
            {"electronic": "Is it electronic?"},
        )
        self.session.pending_guess = "phone"

        result = self.session.respond("yes")

        self.assertFalse(result.game_active)
        self.assertEqual(result.conversation.command.reaction, Reaction.PROUD)
        entry = json.loads(self.trial_log.read_text(encoding="utf-8"))
        self.assertEqual(entry["target"], "phone")
        self.assertTrue(entry["valid_for_training"])

    def test_uncertain_guess_confirmation_keeps_the_game_open(self) -> None:
        self.session.game = ObjectGuessingGame(
            [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
            {"electronic": "Is it electronic?"},
        )
        self.session.pending_guess = "phone"

        result = self.session.respond("probably")

        self.assertTrue(result.game_active)
        self.assertEqual(self.session.pending_guess, "phone")
        self.assertIn("yes or no", result.conversation.command.reply)

    def test_game_loads_an_optional_human_calibration(self) -> None:
        likelihoods = default_likelihoods()
        with TemporaryDirectory() as temporary_directory:
            calibration = Path(temporary_directory) / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "questions": {
                            "electronic": {
                                "if_present": dict(likelihoods.if_present),
                                "if_absent": dict(likelihoods.if_absent),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            session = RobotDialogueSession(
                RuleConversationProvider(),
                ROOT / "data" / "object_catalog.json",
                calibration_path=calibration,
            )
            session.respond("Can we play a guessing game?")

            self.assertIn("electronic", session.game.likelihoods_by_question)

    def test_wrong_guess_against_a_known_object_becomes_training_evidence(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            queue = directory / "suggestions.jsonl"
            trials = directory / "trials.jsonl"
            session = RobotDialogueSession(
                RuleConversationProvider(), ROOT / "data" / "object_catalog.json", queue, trial_log_path=trials
            )
            session.game = ObjectGuessingGame(
                [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
                {"electronic": "Is it electronic?"},
            )
            session.pending_guess = "phone"
            session.pending_confidence = 0.9
            session.game_turns = [{"question_id": "electronic", "question": "Is it electronic?", "answer": "yes"}]

            missed = session.respond("no")
            saved = session.respond("calculator")

            self.assertTrue(missed.game_active)
            self.assertFalse(saved.game_active)
            entry = json.loads(trials.read_text(encoding="utf-8"))
        self.assertEqual(entry["target"], "calculator")
        self.assertEqual(entry["guess"], "phone")
        self.assertTrue(entry["valid_for_training"])

    def test_wrong_guess_for_an_unknown_object_becomes_a_reviewable_suggestion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            queue = Path(temporary_directory) / "suggestions.jsonl"
            session = RobotDialogueSession(RuleConversationProvider(), ROOT / "data" / "object_catalog.json", queue)
            session.game = ObjectGuessingGame(
                [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
                {"electronic": "Is it electronic?"},
            )
            session.pending_guess = "phone"

            missed = session.respond("no")
            question = session.respond("spaceship")
            saved = session.respond("Can it travel in space?")

            self.assertTrue(missed.game_active)
            self.assertTrue(question.game_active)
            self.assertFalse(saved.game_active)
            contents = queue.read_text(encoding="utf-8")
        self.assertIn('"new_object": "spaceship"', contents)
        self.assertIn('"guessed_object": "phone"', contents)

    def test_low_confidence_after_question_budget_requests_learning_not_a_guess(self) -> None:
        self.session.game = ObjectGuessingGame(
            [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
            {"electronic": "Is it electronic?"},
        )
        self.session.game.asked = ["electronic"] * 12

        result = self.session._ask_next_question()

        self.assertTrue(result.game_active)
        self.assertIn("will not make a weak guess", result.conversation.command.reply)
        self.assertIn("phone", result.conversation.command.reply)
