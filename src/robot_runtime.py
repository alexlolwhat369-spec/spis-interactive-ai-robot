"""Connect conversation commands to the explainable object guessing game."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:
    from .conversation import ConversationProvider, ConversationResult
    from .game_trials import record_trial
    from .object_game import ObjectGuessingGame
    from .object_learning import record_suggestion
    from .robot_state import Action, Reaction, RobotCommand
except ImportError:  # Supports direct execution: python src/chat_console.py
    from conversation import ConversationProvider, ConversationResult
    from game_trials import record_trial
    from object_game import ObjectGuessingGame
    from object_learning import record_suggestion
    from robot_state import Action, Reaction, RobotCommand


GAME_INTRODUCTION = "Think of one object. I will ask questions and try to guess your object."


def answer_from_text(text: str) -> str:
    """Convert a short visitor response into an answer the object game understands."""
    normalized = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    if _starts_with(normalized, "no idea", "not sure", "i do not know", "i don't know"):
        return "maybe"
    if _starts_with(
        normalized,
        "probably not",
        "maybe not",
        "i do not think",
        "i don't think",
        "i think not",
        "i guess no",
        "not really",
        "likely not",
    ):
        return "probably_not"
    if _starts_with(normalized, "probably", "maybe yes", "i think so", "i guess so", "likely yes"):
        return "probably"
    if _starts_with(
        normalized,
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "sure",
        "absolutely",
        "definitely",
        "exactly",
        "of course",
    ):
        return "yes"
    if _starts_with(normalized, "no", "nope", "nah", "negative", "not at all"):
        return "no"
    if _starts_with(normalized, "i guess"):
        return "maybe"
    return "maybe"


def _starts_with(text: str, *phrases: str) -> bool:
    return any(text == phrase or text.startswith(f"{phrase} ") for phrase in phrases)


@dataclass(frozen=True)
class SessionResult:
    conversation: ConversationResult
    game_active: bool


class RobotDialogueSession:
    """Keeps normal conversation and the explicit game in one visible session."""

    def __init__(
        self,
        provider: ConversationProvider,
        object_catalog: Path,
        learning_queue: Path | None = None,
        calibration_path: Path | None = None,
        trial_log_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.object_catalog = object_catalog
        self.learning_queue = learning_queue or object_catalog.parent / "pending_object_suggestions.jsonl"
        self.calibration_path = calibration_path or object_catalog.parent / "object_game_calibration.json"
        self.trial_log_path = trial_log_path or object_catalog.parent / "game_trials.jsonl"
        catalog = ObjectGuessingGame.from_file(object_catalog)
        self.catalog_names = {item.name.casefold(): item.name for item in catalog.objects}
        self.game: ObjectGuessingGame | None = None
        self.pending_attribute: str | None = None
        self.pending_guess: str | None = None
        self.pending_confidence: float | None = None
        self.learning_missed_guess: str | None = None
        self.learning_object_name: str | None = None
        self.learning_outcome: str | None = None
        self.learning_confidence: float | None = None
        self.game_turns: list[dict[str, str]] = []

    @property
    def game_active(self) -> bool:
        return self.game is not None or self.learning_missed_guess is not None

    def respond(self, message: str) -> SessionResult:
        if self.game_active:
            return self._answer_game(message)

        result = self.provider.respond(message)
        if result.command.action != Action.START_GAME:
            return SessionResult(result, False)

        self.game = ObjectGuessingGame.from_file(self.object_catalog, self.calibration_path)
        self.game_turns = []
        # The LLM may choose the start action, but it never controls the game's
        # role assignment or opening wording.
        return self._ask_next_question(introduction=GAME_INTRODUCTION)

    def _answer_game(self, message: str) -> SessionResult:
        if message.lower().strip() in {"stop", "quit", "exit", "cancel"}:
            self._clear_game()
            return SessionResult(
                ConversationResult(RobotCommand("Okay, we can play another time.", Reaction.IDLE, Action.STOP)),
                False,
            )

        if self.pending_guess is not None:
            return self._confirm_guess(message)
        if self.learning_missed_guess is not None:
            return self._collect_correction(message)

        if self.pending_attribute is None:
            return self._ask_next_question()
        answer = answer_from_text(message)
        self.game_turns.append(
            {
                "question_id": self.pending_attribute,
                "question": self.game.questions[self.pending_attribute].text,
                "answer": answer,
            }
        )
        self.game.answer(self.pending_attribute, answer)
        return self._ask_next_question()

    def _confirm_guess(self, message: str) -> SessionResult:
        answer = answer_from_text(message)
        if answer == "yes":
            self._record_game_trial(self.pending_guess, "guessed", valid_for_training=True)
            self._clear_game()
            return SessionResult(
                ConversationResult(RobotCommand("Yes! Thanks for playing with me.", Reaction.PROUD)),
                False,
            )
        if answer != "no":
            return SessionResult(
                ConversationResult(RobotCommand("Please tell me clearly: was my guess right, yes or no?", Reaction.CONFUSED)),
                True,
            )
        self.learning_missed_guess = self.pending_guess
        self.learning_outcome = "guessed"
        self.learning_confidence = self.pending_confidence
        self.pending_guess = None
        self.pending_confidence = None
        self.game = None
        return SessionResult(
            ConversationResult(RobotCommand("I missed it. What object were you thinking of?", Reaction.CONFUSED)),
            True,
        )

    def _collect_correction(self, message: str) -> SessionResult:
        if self.learning_object_name is None:
            object_name = " ".join(message.split())
            if not object_name:
                return SessionResult(
                    ConversationResult(RobotCommand("Please tell me the object's name.", Reaction.CONFUSED)),
                    True,
                )
            self.learning_object_name = object_name[:160]
            known_object = self.catalog_names.get(self.learning_object_name.casefold())
            if known_object is not None:
                self._record_game_trial(known_object, self.learning_outcome or "low_confidence", valid_for_training=True)
                self._clear_game()
                return SessionResult(
                    ConversationResult(
                        RobotCommand("Thanks. I logged that round so I can improve my questions.", Reaction.PROUD)
                    ),
                    False,
                )
            reply = (
                f"Thanks. Give me one yes or no question that is true for {self.learning_object_name} "
                f"and false for {self.learning_missed_guess}."
            )
            return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING)), True)

        try:
            record_suggestion(
                self.learning_queue,
                self.learning_missed_guess,
                self.learning_object_name,
                message,
            )
        except ValueError as error:
            return SessionResult(ConversationResult(RobotCommand(str(error), Reaction.CONFUSED)), True)
        self._clear_game()
        return SessionResult(
            ConversationResult(
                RobotCommand("Thank you. I saved that suggestion for my teacher to review before I learn it.", Reaction.PROUD)
            ),
            False,
        )

    def _ask_next_question(self, introduction: str = "") -> SessionResult:
        if self.game is None:
            raise RuntimeError("No active game.")
        # Scores are intentionally unnormalized before the first answer, so a
        # guess is meaningful only after at least one question was answered.
        if self.game.asked and self.game.should_guess():
            guess, confidence = self.game.best_guess()
            self.pending_attribute = None
            self.pending_guess = guess
            self.pending_confidence = confidence
            reply = f"My guess is a {guess}. I am {confidence:.0%} confident. Was I right?"
            return SessionResult(ConversationResult(RobotCommand(reply, Reaction.PROUD)), True)

        if self.game.question_budget_reached():
            return self._request_learning_for_low_confidence()
        question = self.game.next_question()
        if question is None:
            return self._request_learning_for_low_confidence()

        self.pending_attribute, question_text = question
        prefix = f"{introduction} " if introduction else ""
        reply = f"{prefix}{question_text} You can say yes, probably, maybe, probably not, or no."
        return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING, Action.START_GAME)), True)

    def _clear_game(self) -> None:
        self.game = None
        self.pending_attribute = None
        self.pending_guess = None
        self.pending_confidence = None
        self.learning_missed_guess = None
        self.learning_object_name = None
        self.learning_outcome = None
        self.learning_confidence = None
        self.game_turns = []

    def _record_game_trial(self, target: str | None, outcome: str, *, valid_for_training: bool) -> None:
        """Record a finished live round after the visitor confirms its target."""
        if not target:
            return
        guess = self.pending_guess or self.learning_missed_guess or "unknown"
        confidence = self.pending_confidence if self.pending_confidence is not None else self.learning_confidence
        record_trial(
            self.trial_log_path,
            target=target,
            turns=self.game_turns,
            guess=guess,
            confidence=confidence if confidence is not None else 0.0,
            outcome=outcome,
            trial_id="live-game",
            valid_for_training=valid_for_training,
        )

    def _request_learning_for_low_confidence(self) -> SessionResult:
        if self.game is None:
            raise RuntimeError("No active game.")
        guess, confidence = self.game.best_guess()
        self.pending_attribute = None
        self.learning_missed_guess = guess
        self.learning_outcome = "low_confidence"
        self.learning_confidence = confidence
        self.game = None
        reply = (
            f"I am only {confidence:.0%} confident about {guess}, so I will not make a weak guess. "
            "What object were you thinking of?"
        )
        return SessionResult(ConversationResult(RobotCommand(reply, Reaction.CONFUSED)), True)
