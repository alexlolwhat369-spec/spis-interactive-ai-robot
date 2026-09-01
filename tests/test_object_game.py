from __future__ import annotations

from pathlib import Path
import unittest

from src.object_game import ObjectGuessingGame, ObjectProfile, QuestionDefinition


ROOT = Path(__file__).resolve().parents[1]


class ObjectGameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"electronic", "screen"})),
                ObjectProfile("pencil", frozenset({"writing"})),
                ObjectProfile("speaker", frozenset({"electronic", "sound"})),
            ],
            {"electronic": "Is it electronic?", "screen": "Does it have a screen?", "writing": "Is it used for writing?"},
        )

    def test_answering_questions_updates_best_guess(self) -> None:
        self.game.answer("electronic", "yes")
        self.game.answer("screen", "yes")
        guess, confidence = self.game.best_guess()
        self.assertEqual(guess, "phone")
        self.assertGreater(confidence, 0.8)

    def test_maybe_does_not_remove_candidates(self) -> None:
        before = self.game.candidates()
        self.game.answer("electronic", "maybe")
        self.assertEqual(before, self.game.candidates())

    def test_probably_is_softer_than_yes(self) -> None:
        likely_game = ObjectGuessingGame(
            [ObjectProfile("yes_item", frozenset({"feature"})), ObjectProfile("no_item", frozenset())],
            {"feature": "Does it have the feature?"},
        )
        certain_game = ObjectGuessingGame(
            [ObjectProfile("yes_item", frozenset({"feature"})), ObjectProfile("no_item", frozenset())],
            {"feature": "Does it have the feature?"},
        )

        likely_game.answer("feature", "probably")
        certain_game.answer("feature", "yes")

        self.assertGreater(likely_game.scores["yes_item"], 0.5)
        self.assertLess(likely_game.scores["yes_item"], certain_game.scores["yes_item"])

    def test_question_with_no_split_has_no_information_gain(self) -> None:
        game = ObjectGuessingGame(
            [ObjectProfile("one", frozenset({"shared"})), ObjectProfile("two", frozenset({"shared"}))],
            {"shared": "Is it shared?"},
        )

        self.assertEqual(game.expected_information_gain("shared"), 0.0)

    def test_question_limit_never_forces_a_low_confidence_guess(self) -> None:
        self.game.asked = ["electronic"] * 12

        self.assertTrue(self.game.question_budget_reached())
        self.assertFalse(self.game.should_guess())

    def test_school_supply_category_does_not_include_technology_used_for_schoolwork(self) -> None:
        catalog = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        profiles = {item.name: item for item in catalog.objects}

        self.assertEqual(catalog.questions["school"].text, "Is it a school supply?")
        self.assertIn("school_supply", profiles["pen"].attributes)
        self.assertNotIn("school_supply", profiles["laptop"].attributes)
        self.assertIn("school_use", profiles["laptop"].attributes)

    def test_catalog_starts_with_a_deliberate_category_first_sequence(self) -> None:
        catalog = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        asked: list[str] = []
        for _ in range(4):
            question_id, _ = catalog.next_question()
            asked.append(question_id)
            catalog.answer(question_id, "no")

        self.assertEqual(asked, ["electronic", "school", "toy", "food"])

    def test_contrast_question_interprets_no_as_the_defined_opposite(self) -> None:
        game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"portable"})),
                ObjectProfile("desktop computer", frozenset({"stationary"})),
            ],
            {
                "portable_or_stationary": QuestionDefinition(
                    "Is it portable rather than stationary?", "portable", "stationary"
                )
            },
        )

        game.answer("portable_or_stationary", "no")
        guess, confidence = game.best_guess()
        self.assertEqual(guess, "desktop computer")
        self.assertGreater(confidence, 0.8)

    def test_contrast_question_rejects_a_missing_third_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one side"):
            ObjectGuessingGame(
                [
                    ObjectProfile("phone", frozenset({"portable"})),
                    ObjectProfile("mystery", frozenset()),
                ],
                {
                    "portable_or_stationary": QuestionDefinition(
                        "Is it portable rather than stationary?", "portable", "stationary"
                    )
                },
            )

    def test_category_probe_is_followed_by_questions_from_that_category(self) -> None:
        game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"screen"}), "technology"),
                ObjectProfile("camera", frozenset({"camera"}), "technology"),
                ObjectProfile("pen", frozenset({"ink"}), "school"),
                ObjectProfile("apple", frozenset({"fruit"}), "food"),
            ],
            {
                "technology_category": QuestionDefinition(
                    "Is it an electronic device?", "technology_category", category="technology", category_probe=True
                ),
                "school_category": QuestionDefinition(
                    "Is it mainly a school item?", "school_category", category="school", category_probe=True
                ),
                "food_category": QuestionDefinition(
                    "Is it food?", "food_category", category="food", category_probe=True
                ),
                "screen": QuestionDefinition("Does it have a screen?", "screen", category="technology"),
                "camera": QuestionDefinition("Does it take pictures?", "camera", category="technology"),
                "ink": QuestionDefinition("Does it use ink?", "ink", category="school"),
                "fruit": QuestionDefinition("Is it a fruit?", "fruit", category="food"),
            },
        )

        first, _ = game.next_question()
        self.assertEqual(first, "technology_category")
        game.answer(first, "yes")
        second, _ = game.next_question()
        self.assertIn(second, {"screen", "camera"})

    def test_dominant_technology_branch_excludes_other_category_details(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        first, _ = game.next_question()
        self.assertEqual(first, "electronic")
        game.answer(first, "yes")

        self.assertEqual(game.focused_category(), "technology")
        for _ in range(6):
            key, _ = game.next_question()
            self.assertIn(game.questions[key].category, {"technology", "general"}, key)
            game.answer(key, "maybe")

    def test_category_probe_no_moves_to_another_category_probe(self) -> None:
        game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"screen"}), "technology"),
                ObjectProfile("pen", frozenset({"ink"}), "school"),
                ObjectProfile("apple", frozenset({"fruit"}), "food"),
            ],
            {
                "technology_category": QuestionDefinition(
                    "Is it an electronic device?", "technology_category", category="technology", category_probe=True
                ),
                "school_category": QuestionDefinition(
                    "Is it mainly a school item?", "school_category", category="school", category_probe=True
                ),
                "food_category": QuestionDefinition(
                    "Is it food?", "food_category", category="food", category_probe=True
                ),
                "screen": QuestionDefinition("Does it have a screen?", "screen", category="technology"),
            },
        )

        first, _ = game.next_question()
        self.assertEqual(first, "technology_category")
        game.answer(first, "no")
        second, _ = game.next_question()
        self.assertIn(second, {"school_category", "food_category"})

    def test_dominant_category_is_confirmed_before_detail_questions(self) -> None:
        game = ObjectGuessingGame(
            [
                ObjectProfile("phone", frozenset({"screen"}), "technology"),
                ObjectProfile("pen", frozenset({"ink"}), "school"),
                ObjectProfile("apple", frozenset({"fruit"}), "food"),
            ],
            {
                "technology_category": QuestionDefinition(
                    "Is it an electronic device?", "technology_category", category="technology", category_probe=True
                ),
                "school_category": QuestionDefinition(
                    "Is it mainly a school item?", "school_category", category="school", category_probe=True
                ),
                "food_category": QuestionDefinition(
                    "Is it food?", "food_category", category="food", category_probe=True
                ),
                "fruit": QuestionDefinition("Is it a fruit?", "fruit", category="food"),
            },
        )

        game.answer("technology_category", "no")
        game.asked.append("technology_category")
        game.answer("school_category", "no")
        game.asked.append("school_category")

        next_key, _ = game.next_question()
        self.assertEqual(next_key, "food_category")

    def test_catalog_objects_have_unique_attribute_signatures(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        signatures = [item.attributes for item in game.objects]

        self.assertEqual(len(signatures), len(set(signatures)))

    def test_exact_answer_to_unique_attribute_becomes_decisive(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")

        game.answer("computer_pointer", "yes")

        self.assertEqual(game.decisive_guess(), "mouse")

    def test_uncertain_answer_to_unique_attribute_is_not_decisive(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")

        game.answer("computer_pointer", "probably")

        self.assertIsNone(game.decisive_guess())

    def test_catalog_semantics_cover_time_voice_and_soft_materials(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        by_name = {item.name: item for item in game.objects}

        self.assertIn("timekeeping", by_name["watch"].attributes)
        self.assertIn("records_audio", by_name["microphone"].attributes)
        self.assertNotIn("sound", by_name["microphone"].attributes)
        self.assertNotIn("soft", by_name["soccer ball"].attributes)
        self.assertNotIn("soft", by_name["basketball"].attributes)
        self.assertNotIn("writing", by_name["ruler"].attributes)
        self.assertNotIn("pages", by_name["paper"].attributes)

    def test_every_catalog_question_splits_at_least_two_candidates(self) -> None:
        game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        for key, definition in game.questions.items():
            yes_count = sum(
                item.category == definition.category
                if definition.category_probe
                else definition.yes_attribute in item.attributes
                for item in game.objects
            )
            self.assertGreater(yes_count, 0, key)
            self.assertLess(yes_count, len(game.objects), key)

    def test_catalog_guesses_every_object_from_truthful_answers(self) -> None:
        catalog = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
        for target in catalog.objects:
            game = ObjectGuessingGame.from_file(ROOT / "data" / "object_catalog.json")
            while not game.should_guess():
                question = game.next_question()
                self.assertIsNotNone(question)
                attribute, _ = question
                definition = game.questions[attribute]
                is_yes = (
                    target.category == definition.category
                    if definition.category_probe
                    else definition.yes_attribute in target.attributes
                )
                game.answer(attribute, "yes" if is_yes else "no")

            guess, _ = game.best_guess()
            self.assertEqual(guess, target.name, target.name)
            self.assertLessEqual(len(game.asked), 12, target.name)
