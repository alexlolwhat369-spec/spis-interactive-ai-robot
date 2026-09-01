"""Connect conversation commands to the explainable object guessing game."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

try:
    from .conversation import (
        MUSIC_CATEGORY_QUESTION,
        ConversationProvider,
        ConversationResult,
        _preserves_question_terms,
        _valid_game_question,
        explicit_action_result,
        explicit_music_category,
        is_game_request,
        is_music_request,
        music_control_action,
    )
    from .game_trials import record_trial
    from .object_game import ObjectGuessingGame
    from .object_learning import record_suggestion
    from .robot_state import Action, Reaction, RobotCommand
except ImportError:  # Supports direct execution: python src/chat_console.py
    from conversation import (
        MUSIC_CATEGORY_QUESTION,
        ConversationProvider,
        ConversationResult,
        _preserves_question_terms,
        _valid_game_question,
        explicit_action_result,
        explicit_music_category,
        is_game_request,
        is_music_request,
        music_control_action,
    )
    from game_trials import record_trial
    from object_game import ObjectGuessingGame
    from object_learning import record_suggestion
    from robot_state import Action, Reaction, RobotCommand


GAME_INTRODUCTION = "Think of one object. I will ask questions and try to guess your object."
GAME_ANSWER_PHRASES = (
    "yes",
    "yeah",
    "yep",
    "correct",
    "sure",
    "probably",
    "maybe yes",
    "i think so",
    "maybe",
    "not sure",
    "i do not know",
    "probably not",
    "maybe not",
    "i think not",
    "no",
    "nope",
    "nah",
    "stop",
    "quit",
    "cancel",
)
MUSIC_CATEGORY_PHRASES = (
    "calm",
    "relaxing",
    "peaceful",
    "warm",
    "romantic",
    "love",
    "happy",
    "cheerful",
    "fun",
    "energetic",
    "energy",
    "dance",
    "celebration",
    "birthday",
    "victory",
    "stop",
    "cancel",
)
CATEGORY_FOCUS_LABELS = {
    "technology": "electronic objects",
    "school": "school supplies",
    "food_drink": "food and drink objects",
    "play_mobility": "play, sports, and mobility objects",
}


class TurnRoute(StrEnum):
    CONVERSATION = "conversation"
    GAME_START = "game_start"
    GAME_ANSWER = "game_answer"
    MUSIC_REQUEST = "music_request"
    MUSIC_CATEGORY = "music_category"
    MUSIC_CONTROL = "music_control"


def answer_from_text(text: str) -> str | None:
    """Convert a short visitor response into an answer the object game understands."""
    normalized = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    prefix = r"^(?:(?:um+|uh+|well|ok|okay|my answer is|i would say|i'd say|it is|it's)\s+)+"
    normalized = re.sub(prefix, "", normalized).strip()
    if re.search(r"\b(?:not completely sure|hard to say|cannot tell|can't tell|it depends)\b", normalized):
        return "maybe"
    if re.search(r"\b(?:not usually|rarely|most likely not|probably does not|probably doesn't)\b", normalized):
        return "probably_not"
    if re.search(r"\b(?:most of the time|usually|generally|typically|often)\b", normalized):
        return "probably"
    if _starts_with(
        normalized,
        "probably not",
        "probably no",
        "maybe not",
        "i do not think",
        "i don't think",
        "i think not",
        "i guess no",
        "not really",
        "likely not",
        "not likely",
    ):
        return "probably_not"
    if _starts_with(
        normalized,
        "probably",
        "maybe yes",
        "i think so",
        "i guess so",
        "likely",
        "likely yes",
        "most likely",
    ):
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
    if _starts_with(
        normalized,
        "maybe",
        "may be",
        "perhaps",
        "no idea",
        "not sure",
        "i am not sure",
        "i'm not sure",
        "i do not know",
        "i don't know",
        "unsure",
        "could be",
    ):
        return "maybe"
    if _starts_with(normalized, "no", "nope", "nah", "negative", "not at all"):
        return "no"
    if re.search(r"\b(?:does not|doesn't|is not|isn't|has no|never|absolutely not|no way)\b", normalized):
        return "no"
    if re.search(r"\b(?:it does|it is|it has|that is exactly|that's exactly|exactly what|for sure)\b", normalized):
        return "yes"
    if _starts_with(normalized, "i guess"):
        return "maybe"
    return None


def _starts_with(text: str, *phrases: str) -> bool:
    return any(text == phrase or text.startswith(f"{phrase} ") for phrase in phrases)


def object_name_from_text(text: str) -> str:
    """Extract a short object name from a natural correction sentence."""
    cleaned = " ".join(text.strip().strip(".,!?\"'").split())
    cleaned = re.sub(
        r"^(?:actually\s+)?(?:it (?:was|is)|i was thinking (?:of|about)|my object (?:was|is)|"
        r"the object (?:was|is)|i (?:picked|chose))\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:actually\s+)?(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:160].strip()


@dataclass(frozen=True)
class SessionResult:
    conversation: ConversationResult
    game_active: bool
    route: TurnRoute = TurnRoute.CONVERSATION


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
        self.learning_suggested_question: str | None = None
        self.game_turns: list[dict[str, str]] = []
        self.pending_question_text: str | None = None
        self.game_focus: str | None = None
        self.pending_music_category = False

    @property
    def game_active(self) -> bool:
        return self.game is not None or self.learning_missed_guess is not None

    @property
    def expects_game_answer(self) -> bool:
        """True only while one of the game's five short answers is expected."""
        return (
            self.pending_attribute is not None
            or self.pending_guess is not None
            or self.learning_suggested_question is not None
        )

    @property
    def supports_semantic_game_input(self) -> bool:
        return callable(getattr(self.provider, "interpret_game_answer", None))

    @property
    def expects_music_category(self) -> bool:
        return self.pending_music_category

    def route_message(self, message: str) -> TurnRoute:
        """Choose one owner for the turn before any provider can reinterpret it."""
        if music_control_action(message) is not None:
            return TurnRoute.MUSIC_CONTROL
        if is_game_request(message):
            return TurnRoute.GAME_START
        if self.game_active:
            return TurnRoute.GAME_ANSWER
        if self.pending_music_category:
            return TurnRoute.MUSIC_CATEGORY
        if is_music_request(message):
            return TurnRoute.MUSIC_REQUEST
        return TurnRoute.CONVERSATION

    @staticmethod
    def _with_route(result: SessionResult, route: TurnRoute) -> SessionResult:
        return SessionResult(result.conversation, result.game_active, route)

    def respond(self, message: str) -> SessionResult:
        route = self.route_message(message)
        if route == TurnRoute.MUSIC_CONTROL:
            result = explicit_action_result(message)
            if result is None:
                raise RuntimeError("A routed music control did not produce an action.")
            return SessionResult(result, self.game_active, route)

        if self.game_active:
            return self._with_route(self._answer_game(message), route)

        normalized = re.sub(r"[^a-z0-9' ]+", " ", message.lower()).strip()
        if self.pending_music_category:
            if route == TurnRoute.GAME_START:
                self.pending_music_category = False
            else:
                if _starts_with(normalized, "stop", "cancel", "never mind", "nevermind"):
                    self.pending_music_category = False
                    return SessionResult(
                        ConversationResult(RobotCommand("Okay, I will not play music.", Reaction.IDLE, Action.STOP)),
                        False,
                        route,
                    )
                category = explicit_music_category(message)
                if category is None:
                    return SessionResult(
                        ConversationResult(RobotCommand("I did not catch the category. " + MUSIC_CATEGORY_QUESTION, Reaction.CONFUSED)),
                        False,
                        route,
                    )
                self.pending_music_category = False
                return SessionResult(
                    ConversationResult(
                        RobotCommand(f"I will play something {category}.", Reaction.HAPPY, Action.PLAY_MUSIC),
                        category,
                    ),
                    False,
                    route,
                )

        if route == TurnRoute.MUSIC_REQUEST and explicit_music_category(message) is None:
            self.pending_music_category = True
            return SessionResult(
                ConversationResult(RobotCommand(MUSIC_CATEGORY_QUESTION, Reaction.LISTENING)),
                False,
                route,
            )

        if route != TurnRoute.GAME_START:
            return SessionResult(self.provider.respond(message), False, route)

        self.game = ObjectGuessingGame.from_file(self.object_catalog, self.calibration_path)
        self.game_turns = []
        # The LLM may choose the start action, but it never controls the game's
        # role assignment or opening wording.
        return self._with_route(self._ask_next_question(introduction=GAME_INTRODUCTION), route)

    def _answer_game(self, message: str) -> SessionResult:
        normalized = re.sub(r"[^a-z0-9' ]+", " ", message.lower()).strip()
        if _starts_with(normalized, "stop", "quit", "exit", "cancel", "end the game"):
            self._clear_game()
            return SessionResult(
                ConversationResult(RobotCommand("Okay, we can play another time.", Reaction.IDLE, Action.STOP)),
                False,
            )

        if is_game_request(message):
            self._clear_game()
            self.game = ObjectGuessingGame.from_file(self.object_catalog, self.calibration_path)
            return self._ask_next_question(introduction="Starting a new round. " + GAME_INTRODUCTION)

        if self.pending_guess is not None:
            return self._confirm_guess(message)
        if self.learning_missed_guess is not None:
            return self._collect_correction(message)

        if self.pending_attribute is None:
            return self._ask_next_question()
        canonical_question = self.game.questions[self.pending_attribute].text
        asked_question = self.pending_question_text or canonical_question
        answer = self._interpret_game_answer(message, asked_question)
        if answer is None:
            return SessionResult(
                ConversationResult(
                    RobotCommand(
                        "I did not catch a game answer. Please say yes, probably, maybe, probably not, or no.",
                        Reaction.CONFUSED,
                        Action.START_GAME,
                    )
                ),
                True,
            )
        self.game_turns.append(
            {
                "question_id": self.pending_attribute,
                "question": asked_question,
                "canonical_question": canonical_question,
                "answer": answer,
            }
        )
        self.game.answer(self.pending_attribute, answer)
        understood = answer.replace("_", " ")
        return self._ask_next_question(introduction=f"I understood {understood}.")

    def _confirm_guess(self, message: str) -> SessionResult:
        answer = self._interpret_game_answer(message, f"Is {self.pending_guess} your object?")
        if answer in {"yes", "probably"}:
            uncertain = answer == "probably"
            self._record_game_trial(
                self.pending_guess,
                "guessed",
                valid_for_training=not uncertain,
            )
            self._clear_game()
            return SessionResult(
                ConversationResult(
                    RobotCommand(
                        "I will count that as a close match. Thanks for playing with me!"
                        if uncertain
                        else "Yes! Thanks for playing with me.",
                        Reaction.PROUD,
                    )
                ),
                False,
            )
        if answer == "maybe":
            self.pending_guess = None
            self.pending_confidence = None
            return self._ask_question_after_uncertain_guess()
        if answer not in {"no", "probably_not"}:
            return SessionResult(
                ConversationResult(
                    RobotCommand(
                        "I did not catch that. Please say yes, probably, maybe, probably not, or no.",
                        Reaction.CONFUSED,
                    )
                ),
                True,
            )
        self.learning_missed_guess = self.pending_guess
        self.learning_outcome = "guessed"
        self.learning_confidence = self.pending_confidence
        self.pending_guess = None
        self.pending_confidence = None
        self.game = None
        return SessionResult(
            ConversationResult(
                RobotCommand(
                    "That sounds like I probably missed it. What object were you thinking of?"
                    if answer == "probably_not"
                    else "I missed it. What object were you thinking of?",
                    Reaction.CONFUSED,
                )
            ),
            True,
        )

    def _ask_question_after_uncertain_guess(self) -> SessionResult:
        """Use 'maybe' as useful uncertainty instead of rejecting it at a guess."""
        if self.game is None:
            raise RuntimeError("No active game.")
        if self.game.question_budget_reached():
            return self._request_learning_for_low_confidence()
        question = self.game.next_question()
        if question is None:
            return self._request_learning_for_low_confidence()
        self.pending_attribute, canonical_question = question
        question_text = self._question_for_visitor(canonical_question)
        self.pending_question_text = question_text
        focus_context = self._focus_context(self.game.questions[self.pending_attribute].category)
        focus_context = f"{focus_context} " if focus_context else ""
        reply = f"No problem, I will ask one more question. {focus_context}{question_text} You can use all five answers."
        return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING, Action.START_GAME)), True)

    def _collect_correction(self, message: str) -> SessionResult:
        if self.learning_suggested_question is not None:
            if message.strip().endswith("?"):
                return self._save_learning_question(message)
            answer = self._interpret_game_answer(
                message,
                f"Is this a good distinguishing question: {self.learning_suggested_question}",
            )
            if answer in {"yes", "probably"}:
                return self._save_learning_question(self.learning_suggested_question)
            if answer in {"no", "probably_not"}:
                self.learning_suggested_question = None
                return SessionResult(
                    ConversationResult(
                        RobotCommand("Okay. Please give me a better yes or no question.", Reaction.LISTENING)
                    ),
                    True,
                )
            return SessionResult(
                ConversationResult(
                    RobotCommand(
                        "Please say yes or no to my suggested question, or tell me a better question.",
                        Reaction.CONFUSED,
                    )
                ),
                True,
            )

        if self.learning_object_name is None:
            object_name = object_name_from_text(message)
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
            suggestion = self._suggest_distinguishing_question(self.learning_object_name, self.learning_missed_guess)
            if suggestion is not None:
                self.learning_suggested_question = suggestion
                reply = f'I suggest: "{suggestion}" Is that a good distinguishing question?'
                return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING)), True)
            reply = (
                f"Thanks. Give me one yes or no question that is true for {self.learning_object_name} "
                f"and false for {self.learning_missed_guess}."
            )
            return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING)), True)

        return self._save_learning_question(message)

    def _save_learning_question(self, question: str) -> SessionResult:
        try:
            record_suggestion(self.learning_queue, self.learning_missed_guess, self.learning_object_name, question)
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
        decisive_guess = self.game.decisive_guess()
        if decisive_guess is not None:
            self.pending_attribute = None
            self.pending_guess = decisive_guess
            self.pending_confidence = self.game.scores[decisive_guess]
            prefix = f"{introduction} " if introduction else ""
            reply = prefix + (
                f"That answer uniquely matches {decisive_guess} in my object list. "
                "Is that your object?"
            )
            return SessionResult(ConversationResult(RobotCommand(reply, Reaction.PROUD)), True)
        # Scores are intentionally unnormalized before the first answer, so a
        # guess is meaningful only after at least one question was answered.
        if self.game.asked and self.game.should_guess():
            guess, confidence = self.game.best_guess()
            self.pending_attribute = None
            self.pending_guess = guess
            self.pending_confidence = confidence
            prefix = f"{introduction} " if introduction else ""
            reply = f"{prefix}My guess is a {guess}. I am {confidence:.0%} confident. Was I right?"
            return SessionResult(ConversationResult(RobotCommand(reply, Reaction.PROUD)), True)

        if self.game.question_budget_reached():
            return self._request_learning_for_low_confidence()
        question = self.game.next_question()
        if question is None:
            return self._request_learning_for_low_confidence()

        self.pending_attribute, canonical_question = question
        question_text = self._question_for_visitor(canonical_question)
        self.pending_question_text = question_text
        focus_context = self._focus_context(self.game.questions[self.pending_attribute].category)
        prefix = " ".join(part for part in (introduction, focus_context) if part).strip()
        prefix = f"{prefix} " if prefix else ""
        reply = f"{prefix}{question_text} You can say yes, probably, maybe, probably not, or no."
        return SessionResult(ConversationResult(RobotCommand(reply, Reaction.LISTENING, Action.START_GAME)), True)

    def _clear_game(self) -> None:
        self.game = None
        self.pending_attribute = None
        self.pending_question_text = None
        self.pending_guess = None
        self.pending_confidence = None
        self.learning_missed_guess = None
        self.learning_object_name = None
        self.learning_outcome = None
        self.learning_confidence = None
        self.learning_suggested_question = None
        self.game_turns = []
        self.game_focus = None

    def _interpret_game_answer(self, message: str, question: str) -> str | None:
        answer = answer_from_text(message)
        if answer is not None:
            return answer
        interpreter = getattr(self.provider, "interpret_game_answer", None)
        if not callable(interpreter):
            return None
        try:
            answer = interpreter(message, question, self.game_turns)
        except Exception:
            return None
        return answer if answer in {"yes", "probably", "maybe", "probably_not", "no"} else None

    def _question_for_visitor(self, canonical_question: str) -> str:
        rephrase = getattr(self.provider, "rephrase_game_question", None)
        if not callable(rephrase):
            return canonical_question
        try:
            question = rephrase(canonical_question, self.game_turns)
        except Exception:
            return canonical_question
        if not isinstance(question, str):
            return canonical_question
        question = question.strip()
        return (
            question
            if _valid_game_question(question) and _preserves_question_terms(canonical_question, question)
            else canonical_question
        )

    def _suggest_distinguishing_question(self, new_object: str, missed_guess: str | None) -> str | None:
        suggest = getattr(self.provider, "suggest_distinguishing_question", None)
        if not callable(suggest) or not missed_guess:
            return None
        try:
            question = suggest(new_object, missed_guess, self.game_turns)
        except Exception:
            return None
        if not isinstance(question, str):
            return None
        question = question.strip()
        return question if _valid_game_question(question) else None

    def _focus_context(self, question_category: str) -> str:
        if self.game is None:
            return ""
        focus = self.game.focused_category()
        if focus is None:
            return ""
        label = CATEGORY_FOCUS_LABELS.get(focus, focus.replace("_", " "))
        if focus != self.game_focus:
            self.game_focus = focus
            return f"Your answers point toward {label}, so I will focus there."
        if question_category == "general":
            return f"Still focusing on {label}."
        return ""

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
