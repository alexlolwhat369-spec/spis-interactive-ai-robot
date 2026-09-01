from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.conversation import ConversationResult, RuleConversationProvider
from src.object_game import ObjectGuessingGame, ObjectProfile, default_likelihoods
from src.robot_runtime import RobotDialogueSession, answer_from_text, object_name_from_text
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
        self.assertTrue(self.session.expects_game_answer)

    def test_electronic_yes_announces_and_keeps_the_technology_branch(self) -> None:
        self.session.respond("Play the object game")

        second = self.session.respond("yes")

        self.assertEqual(self.session.game.focused_category(), "technology")
        self.assertIn("I understood yes", second.conversation.command.reply)
        self.assertIn("point toward electronic objects", second.conversation.command.reply)
        self.assertIn(
            self.session.game.questions[self.session.pending_attribute].category,
            {"technology", "general"},
        )

        third = self.session.respond("maybe")
        self.assertIn("Still focusing on electronic objects", third.conversation.command.reply)
        self.assertNotEqual(self.session.game.questions[self.session.pending_attribute].category, "school")

    def test_free_conversation_and_learning_names_do_not_limit_speech_vocabulary(self) -> None:
        self.assertFalse(self.session.expects_game_answer)
        self.session.learning_missed_guess = "phone"
        self.assertFalse(self.session.expects_game_answer)

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
        self.assertEqual(answer_from_text("maybe"), "maybe")
        self.assertEqual(answer_from_text("could be"), "maybe")
        self.assertEqual(answer_from_text("probably"), "probably")
        self.assertEqual(answer_from_text("probably not"), "probably_not")
        self.assertEqual(answer_from_text("maybe not"), "probably_not")
        self.assertEqual(answer_from_text("Well, probably"), "probably")
        self.assertEqual(answer_from_text("Okay, maybe"), "maybe")
        self.assertEqual(answer_from_text("I would say probably not"), "probably_not")
        self.assertEqual(answer_from_text("It has one most of the time"), "probably")
        self.assertEqual(answer_from_text("I am not completely sure"), "maybe")
        self.assertEqual(answer_from_text("It does not have one"), "no")
        self.assertEqual(answer_from_text("That is exactly what it does"), "yes")
        self.assertIsNone(answer_from_text("I like pizza"))

    def test_natural_correction_extracts_only_the_object_name(self) -> None:
        self.assertEqual(object_name_from_text("It was actually a toaster."), "toaster")
        self.assertEqual(object_name_from_text("I was thinking of an electric fan"), "electric fan")

    def test_unrecognized_game_answer_repeats_without_consuming_question(self) -> None:
        opening = self.session.respond("Play Alkinator")
        pending_attribute = self.session.pending_attribute
        asked = list(self.session.game.asked)

        result = self.session.respond("I like pizza")

        self.assertTrue(opening.game_active)
        self.assertTrue(result.game_active)
        self.assertEqual(self.session.pending_attribute, pending_attribute)
        self.assertEqual(self.session.game.asked, asked)
        self.assertIn("did not catch a game answer", result.conversation.command.reply)

    def test_repeating_game_request_restarts_a_clean_round(self) -> None:
        self.session.respond("Play the object game")
        self.session.respond("yes")
        self.assertGreater(len(self.session.game.asked), 1)

        restarted = self.session.respond("Start Akinator again")

        self.assertTrue(restarted.game_active)
        self.assertEqual(len(self.session.game.asked), 1)
        self.assertIn("Starting a new round", restarted.conversation.command.reply)

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

    def test_probably_accepts_a_close_final_guess(self) -> None:
        self.session.game = ObjectGuessingGame(
            [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
            {"electronic": "Is it electronic?"},
        )
        self.session.pending_guess = "phone"

        result = self.session.respond("probably")

        self.assertFalse(result.game_active)
        self.assertIsNone(self.session.pending_guess)
        self.assertIn("close match", result.conversation.command.reply)

    def test_maybe_after_a_guess_asks_one_more_question(self) -> None:
        self.session.game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"electronic", "portable"})),
                ObjectProfile("desktop", frozenset({"electronic"})),
                ObjectProfile("book", frozenset({"portable"})),
            ],
            {
                "electronic": "Is it electronic?",
                "portable": "Is it portable?",
            },
        )
        self.session.game.asked = ["electronic"]
        self.session.pending_guess = "phone"

        result = self.session.respond("maybe")

        self.assertTrue(result.game_active)
        self.assertIsNone(self.session.pending_guess)
        self.assertEqual(self.session.pending_attribute, "portable")
        self.assertIn("one more question", result.conversation.command.reply)

    def test_probably_not_rejects_the_guess_without_being_called_invalid(self) -> None:
        self.session.game = ObjectGuessingGame(
            [ObjectProfile("phone", frozenset({"electronic"})), ObjectProfile("book", frozenset())],
            {"electronic": "Is it electronic?"},
        )
        self.session.pending_guess = "phone"

        result = self.session.respond("probably not")

        self.assertTrue(result.game_active)
        self.assertIsNone(self.session.pending_guess)
        self.assertEqual(self.session.learning_missed_guess, "phone")
        self.assertIn("probably missed", result.conversation.command.reply)

    def test_unique_pointer_answer_immediately_guesses_mouse(self) -> None:
        self.session.game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        self.session.game.asked.append("computer_pointer")
        self.session.pending_attribute = "computer_pointer"

        result = self.session.respond("yes")

        self.assertTrue(result.game_active)
        self.assertEqual(self.session.pending_guess, "mouse")
        self.assertIsNone(self.session.pending_attribute)
        self.assertIn("uniquely matches mouse", result.conversation.command.reply)

    def test_semantic_advisor_can_interpret_a_natural_game_answer(self) -> None:
        class HybridProvider:
            def respond(self, message: str) -> ConversationResult:
                del message
                return ConversationResult(RobotCommand("Okay.", Reaction.SPEAKING))

            def interpret_game_answer(self, message: str, question: str, turns: list[dict[str, str]]) -> str:
                self.seen = (message, question, turns)
                return "yes"

        provider = HybridProvider()
        session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json", trial_log_path=self.trial_log)
        session.game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        session.game.asked.append("computer_pointer")
        session.pending_attribute = "computer_pointer"
        session.pending_question_text = "Would you use it to control the cursor?"

        result = session.respond("I use it to control the cursor")

        self.assertEqual(session.pending_guess, "mouse")
        self.assertIn("uniquely matches mouse", result.conversation.command.reply)
        self.assertEqual(provider.seen[1], "Would you use it to control the cursor?")

    def test_hybrid_provider_can_rephrase_but_not_choose_game_state(self) -> None:
        class HybridProvider:
            def respond(self, message: str) -> ConversationResult:
                del message
                return ConversationResult(RobotCommand("Okay.", Reaction.SPEAKING))

            def rephrase_game_question(self, question: str, turns: list[dict[str, str]]) -> str:
                del question, turns
                return "Would you describe it as mainly electronic?"

        session = RobotDialogueSession(HybridProvider(), ROOT / "data" / "object_catalog.json")

        result = session.respond("Play the object game")

        self.assertEqual(session.pending_attribute, "electronic")
        self.assertIn("Would you describe it as mainly electronic?", result.conversation.command.reply)

    def test_broken_semantic_advisor_falls_back_without_consuming_the_answer(self) -> None:
        class BrokenHybridProvider:
            def respond(self, message: str) -> ConversationResult:
                del message
                return ConversationResult(RobotCommand("Okay.", Reaction.SPEAKING))

            def rephrase_game_question(self, question: str, turns: list[dict[str, str]]) -> str:
                del question, turns
                raise RuntimeError("Ollama unavailable")

            def interpret_game_answer(self, message: str, question: str, turns: list[dict[str, str]]) -> str:
                del message, question, turns
                raise RuntimeError("Ollama unavailable")

        session = RobotDialogueSession(BrokenHybridProvider(), ROOT / "data" / "object_catalog.json")
        opening = session.respond("Play the object game")
        pending = session.pending_attribute
        asked = list(session.game.asked)

        repeated = session.respond("I like pizza")

        self.assertIn("Is it mainly an electronic device?", opening.conversation.command.reply)
        self.assertEqual(session.pending_attribute, pending)
        self.assertEqual(session.game.asked, asked)
        self.assertIn("did not catch a game answer", repeated.conversation.command.reply)

    def test_llm_learning_question_needs_visitor_approval_before_it_is_saved(self) -> None:
        class HybridProvider:
            def respond(self, message: str) -> ConversationResult:
                del message
                return ConversationResult(RobotCommand("Okay.", Reaction.SPEAKING))

            def suggest_distinguishing_question(
                self, new_object: str, missed_guess: str, turns: list[dict[str, str]]
            ) -> str:
                self.seen = (new_object, missed_guess, turns)
                return "Can it toast slices of bread?"

        with TemporaryDirectory() as temporary_directory:
            queue = Path(temporary_directory) / "suggestions.jsonl"
            provider = HybridProvider()
            session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json", queue)
            session.learning_missed_guess = "mouse"
            session.learning_outcome = "low_confidence"

            proposed = session.respond("It was actually a toaster.")
            self.assertFalse(queue.exists())
            self.assertIn("Can it toast slices of bread?", proposed.conversation.command.reply)

            saved = session.respond("yes")
            contents = queue.read_text(encoding="utf-8")

        self.assertFalse(saved.game_active)
        self.assertIn('"new_object": "toaster"', contents)
        self.assertIn('"distinguishing_question": "Can it toast slices of bread?"', contents)

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
