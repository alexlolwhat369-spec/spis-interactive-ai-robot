"""A small, explainable Akinator-style object guessing game."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

Answer = Literal["yes", "probably", "maybe", "probably_not", "no"]

# P(answer | an object really has the asked attribute). The values sum to one,
# which keeps expected-information-gain calculations mathematically valid.
ANSWER_IF_PRESENT: dict[Answer, float] = {
    "yes": 0.86,
    "probably": 0.06,
    "maybe": 0.03,
    "probably_not": 0.01,
    "no": 0.04,
}
MAX_QUESTIONS = 12
CATEGORY_FOCUS_THRESHOLD = 0.65
CATEGORY_QUESTION_BONUS = 0.10


@dataclass(frozen=True)
class ObjectProfile:
    name: str
    attributes: frozenset[str]
    category: str = "general"


@dataclass(frozen=True)
class QuestionDefinition:
    """A question with an optional, explicit opposite class.

    For ordinary questions, a ``no`` means the object lacks ``yes_attribute``.
    For contrast questions, ``no_attribute`` documents the intended other side
    and prevents using a misleading "A or B" question with a third possibility.
    """

    text: str
    yes_attribute: str
    no_attribute: str | None = None
    category: str = "general"
    category_probe: bool = False


@dataclass(frozen=True)
class QuestionLikelihoods:
    """Observed answer distributions for one question after human QA."""

    if_present: Mapping[Answer, float]
    if_absent: Mapping[Answer, float]


class ObjectGuessingGame:
    def __init__(
        self,
        objects: list[ObjectProfile],
        questions: Mapping[str, str | QuestionDefinition],
        likelihoods_by_question: Mapping[str, QuestionLikelihoods] | None = None,
    ) -> None:
        if len(objects) < 2:
            raise ValueError("The game needs at least two objects.")
        self.objects = tuple(objects)
        self.questions = {
            key: value if isinstance(value, QuestionDefinition) else QuestionDefinition(value, key)
            for key, value in questions.items()
        }
        self._validate_contrast_questions()
        self._validate_question_categories()
        self.likelihoods_by_question = dict(likelihoods_by_question or {})
        self.scores = {item.name: 1.0 / len(self.objects) for item in self.objects}
        self.asked: list[str] = []

    @classmethod
    def from_file(cls, path: Path, calibration_path: Path | None = None) -> "ObjectGuessingGame":
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = [
            ObjectProfile(item["name"], frozenset(item["attributes"]), item.get("category", "general"))
            for item in data["objects"]
        ]
        questions = {key: _question_from_data(key, value) for key, value in data["questions"].items()}
        likelihoods = load_calibration(calibration_path) if calibration_path else {}
        return cls(objects, questions, likelihoods)

    def candidates(self) -> list[str]:
        return [name for name, _ in sorted(self.scores.items(), key=lambda item: item[1], reverse=True)]

    def next_question(self) -> tuple[str, str] | None:
        available = [key for key in self.questions if key not in self.asked]
        if not available:
            return None
        attribute = self._choose_category_aware_question(available)
        self.asked.append(attribute)
        return attribute, self.questions[attribute].text

    def category_probabilities(self) -> dict[str, float]:
        """Current probability mass for each primary object category."""
        categories = {item.category for item in self.objects}
        return {
            category: sum(self.scores[item.name] for item in self.objects if item.category == category)
            for category in categories
        }

    def answer(self, attribute: str, answer: Answer) -> None:
        if attribute not in self.questions:
            raise ValueError(f"Unknown question: {attribute}")
        if answer not in ANSWER_IF_PRESENT:
            raise ValueError(f"Invalid answer: {answer}")
        self.scores = self._normalized_posterior(attribute, answer)

    def expected_information_gain(self, attribute: str) -> float:
        """Expected Shannon entropy reduction for one candidate question."""
        if attribute not in self.questions:
            raise ValueError(f"Unknown question: {attribute}")
        expected_entropy = 0.0
        for answer in ANSWER_IF_PRESENT:
            unnormalized = self._posterior(attribute, answer)
            evidence = sum(unnormalized.values())
            if evidence > 0.0:
                posterior = (score / evidence for score in unnormalized.values())
                expected_entropy += evidence * self._entropy(posterior)
        return max(0.0, self.entropy() - expected_entropy)

    def entropy(self) -> float:
        return self._entropy(self.scores.values())

    def best_guess(self) -> tuple[str, float]:
        name = max(self.scores, key=self.scores.get)
        return name, self.scores[name]

    def should_guess(self) -> bool:
        ordered = sorted(self.scores.values(), reverse=True)
        confidence = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        # A question budget protects the visitor's time, but it must never turn
        # a weak 30% candidate into a pretend confident guess.
        return confidence >= 0.82 and confidence - runner_up >= 0.45

    def question_budget_reached(self) -> bool:
        return len(self.asked) >= MAX_QUESTIONS

    def _posterior(self, attribute: str, answer: Answer) -> dict[str, float]:
        return {
            item.name: self.scores[item.name]
            * self._answer_likelihood(
                self._item_answers_yes(item, self.questions[attribute]),
                answer,
                self.likelihoods_by_question.get(attribute),
            )
            for item in self.objects
        }

    def _choose_category_aware_question(self, available: list[str]) -> str:
        """Ask category first, then investigate the currently likely branch."""
        probabilities = self.category_probabilities()
        dominant_category, dominant_probability = max(probabilities.items(), key=lambda item: item[1])
        if dominant_probability >= CATEGORY_FOCUS_THRESHOLD:
            confirmation = [
                key
                for key in available
                if self.questions[key].category_probe and self.questions[key].category == dominant_category
            ]
            if confirmation:
                return max(confirmation, key=self.expected_information_gain)
            general_or_other_details = [key for key in available if not self.questions[key].category_probe]
            if general_or_other_details:
                # Prefer the active category when two questions are similarly
                # useful, but let a much stronger general discriminator win.
                return max(
                    general_or_other_details,
                    key=lambda key: self.expected_information_gain(key)
                    + (CATEGORY_QUESTION_BONUS if self.questions[key].category == dominant_category else 0.0),
                )

        probes = [key for key in available if self.questions[key].category_probe]
        if probes:
            return max(probes, key=self.expected_information_gain)

        # Once category probes are exhausted, use the same expected
        # information-gain rule across the remaining general questions.
        return max(available, key=self.expected_information_gain)

    @staticmethod
    def _item_answers_yes(item: ObjectProfile, question: QuestionDefinition) -> bool:
        if question.category_probe:
            return item.category == question.category
        return question.yes_attribute in item.attributes

    def _validate_contrast_questions(self) -> None:
        for key, question in self.questions.items():
            if question.no_attribute is None:
                continue
            for item in self.objects:
                is_yes_side = question.yes_attribute in item.attributes
                is_no_side = question.no_attribute in item.attributes
                if is_yes_side == is_no_side:
                    raise ValueError(
                        f"Contrast question '{key}' must put every object in exactly one side: "
                        f"{question.yes_attribute} or {question.no_attribute}."
                    )

    def _validate_question_categories(self) -> None:
        known_categories = {item.category for item in self.objects}
        for key, question in self.questions.items():
            if question.category_probe and question.category not in known_categories:
                raise ValueError(f"Category question '{key}' refers to unknown category '{question.category}'.")

    def _normalized_posterior(self, attribute: str, answer: Answer) -> dict[str, float]:
        posterior = self._posterior(attribute, answer)
        evidence = sum(posterior.values())
        if evidence <= 0.0:
            return self.scores.copy()
        return {name: score / evidence for name, score in posterior.items()}

    @staticmethod
    def _entropy(scores: Iterable[float]) -> float:
        return -sum(score * math.log2(score) for score in scores if score > 0.0)

    @staticmethod
    def _answer_likelihood(
        has_attribute: bool,
        answer: Answer,
        calibration: QuestionLikelihoods | None = None,
    ) -> float:
        if calibration is not None:
            likelihoods = calibration.if_present if has_attribute else calibration.if_absent
            return likelihoods[answer]
        likelihood = ANSWER_IF_PRESENT[answer]
        return likelihood if has_attribute else ANSWER_IF_PRESENT[_opposite_answer(answer)]


def _opposite_answer(answer: Answer) -> Answer:
    opposites: dict[Answer, Answer] = {
        "yes": "no",
        "probably": "probably_not",
        "maybe": "maybe",
        "probably_not": "probably",
        "no": "yes",
    }
    return opposites[answer]


def _question_from_data(key: str, value: object) -> QuestionDefinition:
    if isinstance(value, str):
        return QuestionDefinition(value, key)
    if not isinstance(value, dict):
        raise ValueError(f"Question '{key}' must be text or an object definition.")
    text = value.get("text")
    yes_attribute = value.get("yes_attribute", key)
    no_attribute = value.get("no_attribute")
    category = value.get("category", "general")
    category_probe = value.get("category_probe", False)
    if not isinstance(text, str) or not isinstance(yes_attribute, str):
        raise ValueError(f"Question '{key}' needs text and yes_attribute strings.")
    if no_attribute is not None and not isinstance(no_attribute, str):
        raise ValueError(f"Question '{key}' has an invalid no_attribute.")
    if not isinstance(category, str) or not isinstance(category_probe, bool):
        raise ValueError(f"Question '{key}' has invalid category metadata.")
    return QuestionDefinition(text, yes_attribute, no_attribute, category, category_probe)


def default_likelihoods() -> QuestionLikelihoods:
    """Return the hand-authored fallback used before enough human trials exist."""
    return QuestionLikelihoods(
        if_present=ANSWER_IF_PRESENT.copy(),
        if_absent={answer: ANSWER_IF_PRESENT[_opposite_answer(answer)] for answer in ANSWER_IF_PRESENT},
    )


def load_calibration(path: Path) -> dict[str, QuestionLikelihoods]:
    """Load only valid question calibration; an absent file means baseline behavior."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Object-game calibration is not valid JSON: {path}") from error
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, dict):
        raise ValueError("Object-game calibration needs a questions object.")
    calibrated: dict[str, QuestionLikelihoods] = {}
    for key, value in questions.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("Object-game calibration has an invalid question entry.")
        if_present = _validated_distribution(value.get("if_present"), key, "if_present")
        if_absent = _validated_distribution(value.get("if_absent"), key, "if_absent")
        calibrated[key] = QuestionLikelihoods(if_present, if_absent)
    return calibrated


def _validated_distribution(value: object, question_key: str, side: str) -> dict[Answer, float]:
    if not isinstance(value, dict) or set(value) != set(ANSWER_IF_PRESENT):
        raise ValueError(f"Calibration for {question_key} needs every answer on {side}.")
    try:
        distribution = {answer: float(value[answer]) for answer in ANSWER_IF_PRESENT}
    except (TypeError, ValueError) as error:
        raise ValueError(f"Calibration for {question_key} has invalid probabilities.") from error
    if any(probability <= 0.0 for probability in distribution.values()) or not math.isclose(
        sum(distribution.values()), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"Calibration for {question_key} must be positive and sum to one.")
    return distribution
