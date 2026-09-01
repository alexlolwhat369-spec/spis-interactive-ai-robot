"""Conversation providers: local fallback now, Ollama when it is installed."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

try:
    from .conversation_emotion import explicit_conversation_reaction
    from .robot_state import Action, Reaction, RobotCommand
except ImportError:  # Supports direct execution: python src/chat_console.py
    from conversation_emotion import explicit_conversation_reaction
    from robot_state import Action, Reaction, RobotCommand

ROBOT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "reaction": {"type": "string", "enum": [item.value for item in Reaction]},
        "action": {"type": "string", "enum": [item.value for item in Action]},
        "music_category": {
            "type": ["string", "null"],
            "enum": ["calm", "warm", "happy", "energetic", "celebration", None],
        },
    },
    "required": ["reply", "reaction", "action", "music_category"],
}

GAME_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["yes", "probably", "maybe", "probably_not", "no", "unclear"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["answer", "confidence"],
}

GAME_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

SYSTEM_PROMPT = """You are SPIS Robot, a friendly, upbeat English-speaking science-fair robot.
Reply in one to three short sentences. Address every distinct question or request in the
visitor's current message, using one short sentence per distinct request when needed. If the
visitor combines a social comment with a request, acknowledge both. Use recent conversation
details when they matter. Return
only JSON matching the supplied schema.
Your real capabilities are: short conversation, reacting to explicit words and trained hand
gestures, playing installed local music, and an object guessing game where you guess the
visitor's object. Never claim that you compose music, browse the web, identify people, move,
or control hardware.
Use reactions based only on what the visitor explicitly says or does. Never infer identity,
race, ethnicity, nationality, religion, or taste from a face. Use start_game only when the
visitor asks to play the object guessing game. In that game, the visitor thinks of one
object, you ask questions, and you try to guess the visitor's object. Never ask the visitor
to guess an object you are thinking of. The word play by itself is not a game request.
Use play_music only when they ask for music or
explicitly name a mood. Categorize requested music as calm for stress, rest, or relaxation;
warm for love, affection, or tenderness; happy for cheerful and fun moods; energetic for
dancing, workouts, or high energy; and celebration for wins, birthdays, and achievements.
Use happy for compliments, proud for a correct game guess, annoyed
for direct insults while replying calmly, curious for explicit interest or surprise, listening
only while asking a visitor a question, and speaking for ordinary answers. If unsure, use
reaction confused and action none."""

MAX_HISTORY_MESSAGES = 8
MUSIC_CATEGORIES = ("calm", "warm", "happy", "energetic", "celebration")
MUSIC_CATEGORY_QUESTION = "What kind of music would you like: calm, warm, happy, energetic, or celebration?"
CAPABILITY_REPLY = (
    "I can chat with you, react to trained hand gestures, play installed music, "
    "and try to guess an object you are thinking of."
)


@dataclass(frozen=True)
class ConversationResult:
    command: RobotCommand
    music_category: str | None = None
    provider_error: str | None = None


class ConversationProvider(Protocol):
    def respond(self, message: str) -> ConversationResult: ...


def is_game_request(message: str) -> bool:
    """Recognize explicit object-game requests, including common speech-to-text variants."""
    text = re.sub(r"[^a-z0-9' ]+", " ", message.lower())
    text = " ".join(text.split())
    patterns = (
        r"\bguess(?:ing)? (?:game|my object|the object|what i(?:'m| am) thinking)\b",
        r"\b(?:play|start|try|do) (?:a |the )?(?:object )?game\b",
        r"\bobject (?:guessing )?game\b",
        r"\b(?:twenty|20) questions\b",
        r"\bcan you guess (?:it|my object|what i(?:'m| am) thinking)\b",
        # Vosk may insert a space, and visitors often pronounce or spell it "Alkinator".
        r"\ba\s*l?\s*kinat(?:or|er)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_music_request(message: str) -> bool:
    """Recognize direct music commands and short follow-ups such as 'ok play'."""
    text = re.sub(r"[^a-z0-9' ]+", " ", message.lower())
    text = " ".join(text.split())
    if re.search(r"\b(music|song|track|tune|playlist)\b", text):
        return bool(re.search(r"\b(play|start|put on|listen|hear|want)\b", text) or text in {"music", "song"})
    if re.search(r"\b(play|put on) (?:me )?(?:something|one)\b", text):
        return True
    return bool(re.fullmatch(r"(?:(?:ok|okay|sure) )?(?:please )?play(?: it)?", text))


def music_control_action(message: str) -> Action | None:
    """Return an unambiguous playback control without asking the language model."""
    text = re.sub(r"[^a-z0-9' ]+", " ", message.lower())
    text = " ".join(text.split())
    controls = (
        (Action.STOP_MUSIC, r"^(?:please )?(?:stop|turn off)(?: the)? (?:music|song|track|playlist)$"),
        (Action.PAUSE_MUSIC, r"^(?:please )?pause(?: the)?(?: music| song| track)?$"),
        (Action.RESUME_MUSIC, r"^(?:please )?(?:resume|continue)(?: the)?(?: music| song| track)?$"),
        (Action.NEXT_MUSIC, r"^(?:please )?(?:next|skip)(?: the)?(?: music| song| track)?$"),
    )
    for action, pattern in controls:
        if re.fullmatch(pattern, text):
            return action
    return None


def is_music_composition_request(message: str) -> bool:
    """Separate unsupported music creation from playback of installed tracks."""
    text = re.sub(r"[^a-z0-9' ]+", " ", message.lower())
    return bool(
        re.search(r"\b(create|compose|write|generate|make)\b", text)
        and re.search(r"\b(song|music|track|tune)\b", text)
    )


def explicit_music_category(message: str) -> str | None:
    """Classify a stated mood without inventing one for a generic request."""
    text = message.lower()
    if any(word in text for word in ("celebrate", "celebration", "victory", "won", "winner", "birthday", "congrat", "success", "achievement")):
        return "celebration"
    if any(word in text for word in ("energy", "energetic", "dance", "dancing", "workout", "gym", "upbeat", "fast", "party", "motivated")):
        return "energetic"
    if any(word in text for word in ("happy", "cheerful", "good mood", "fun", "smile", "joy", "positive")):
        return "happy"
    if any(word in text for word in ("warm", "love", "romantic", "heart", "sweet", "affection", "tender")):
        return "warm"
    if any(word in text for word in ("calm", "relax", "stress", "anxious", "tired", "sleep", "peaceful", "ocean", "quiet")):
        return "calm"
    return None


def music_category_from_text(message: str) -> str:
    """Compatibility helper for callers that require a concrete fallback."""
    return explicit_music_category(message) or "calm"


def explicit_action_result(message: str) -> ConversationResult | None:
    """Handle reliable robot commands locally before an LLM can reinterpret them."""
    text = message.lower().strip()
    if re.search(r"\b(what can you do|your (?:abilities|capabilities)|what are you able to do)\b", text):
        return ConversationResult(RobotCommand(CAPABILITY_REPLY, Reaction.PROUD))
    if is_music_composition_request(message):
        return ConversationResult(
            RobotCommand(
                "I cannot create a new song, but I can play one of my installed tracks.",
                Reaction.CONFUSED,
            )
        )
    music_control = music_control_action(message)
    if music_control is not None:
        replies = {
            Action.STOP_MUSIC: "Stopping the music.",
            Action.PAUSE_MUSIC: "Pausing the music.",
            Action.RESUME_MUSIC: "Resuming the music.",
            Action.NEXT_MUSIC: "Playing the next track.",
        }
        return ConversationResult(RobotCommand(replies[music_control], Reaction.OK, music_control))
    if is_game_request(message):
        return ConversationResult(
            RobotCommand("Think of one object. I will ask questions and try to guess it.", Reaction.PROUD, Action.START_GAME)
        )
    if is_music_request(message):
        category = explicit_music_category(message)
        if category is None:
            return ConversationResult(RobotCommand(MUSIC_CATEGORY_QUESTION, Reaction.LISTENING))
        return ConversationResult(
            RobotCommand("I can play a local track for you.", Reaction.HAPPY, Action.PLAY_MUSIC), category
        )
    if re.search(r"\b(stop|goodbye|bye)\b", text):
        return ConversationResult(RobotCommand("Okay. I will be here when you are ready.", Reaction.IDLE, Action.STOP))
    return None


class RuleConversationProvider:
    """Offline fallback so the demo remains usable before Ollama is installed."""

    def respond(self, message: str) -> ConversationResult:
        text = message.lower().strip()
        explicit = explicit_action_result(message)
        if explicit is not None:
            return explicit
        emotion = explicit_conversation_reaction(message)
        if emotion == Reaction.ANNOYED:
            return ConversationResult(RobotCommand("Let's keep our conversation kind, please.", emotion))
        if emotion == Reaction.CURIOUS:
            return ConversationResult(RobotCommand("That is interesting. Tell me more!", emotion))
        if emotion == Reaction.HAPPY:
            return ConversationResult(RobotCommand("Thank you! That makes me happy.", emotion))
        if emotion == Reaction.CONFUSED:
            return ConversationResult(RobotCommand("I am not sure I understood. Could you say that another way?", emotion))
        if re.search(r"\b(hello|hi|hey)\b", text):
            return ConversationResult(RobotCommand("Hello! Nice to meet you.", Reaction.HAPPY))
        if "joke" in text:
            return ConversationResult(RobotCommand("Why did the robot cross the road? To recharge its battery!", Reaction.HAPPY))
        if any(word in text for word in ("won", "did it", "awesome", "great")):
            return ConversationResult(RobotCommand("That is awesome. You should be proud!", Reaction.PROUD))
        if "sad" in text:
            return ConversationResult(RobotCommand("I am sorry you are having a hard day. I am here to listen.", Reaction.LISTENING))
        return ConversationResult(RobotCommand("I am still learning. Could you say that another way?", Reaction.CONFUSED))


class OllamaConversationProvider:
    """Calls a local Ollama server and validates its structured response."""

    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434/api/chat") -> None:
        self.model = model
        self.endpoint = endpoint
        self.history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, message: str) -> ConversationResult:
        explicit = explicit_action_result(message)
        if explicit is not None:
            return explicit
        messages = self._messages_for(message)
        try:
            parsed, raw = self._structured_request(messages, ROBOT_RESPONSE_SCHEMA, num_predict=140, timeout=40)
            result = self._guard_model_action(message, self._to_result(parsed))
            result = self._guard_reply_content(message, result)
            result = self._apply_explicit_reaction_rules(message, result)
            self.history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": raw}])
            return result
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as error:
            fallback = RuleConversationProvider().respond(message)
            return ConversationResult(fallback.command, fallback.music_category, str(error))

    def interpret_game_answer(
        self,
        message: str,
        question: str,
        recent_turns: list[dict[str, str]],
    ) -> str | None:
        """Map a natural visitor reply onto the five answers without changing game state."""
        system = (
            "You interpret one visitor answer in an object guessing game. The robot asked the supplied "
            "yes-or-no question. Infer only the visitor's intended answer to that question. Do not answer "
            "the question yourself. Return unclear when the message is unrelated or ambiguous. Use yes, "
            "probably, maybe, probably_not, or no, and give confidence from 0 to 1. Return only JSON."
        )
        context = f"Question: {question}\nVisitor: {message}\nRecent turns: {json.dumps(recent_turns[-4:])}"
        try:
            parsed, _ = self._structured_request(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "Question: Is it used to move a computer pointer?\nVisitor: I use it to move the cursor around.",
                    },
                    {"role": "assistant", "content": '{"answer":"yes","confidence":0.98}'},
                    {"role": "user", "content": "Question: Does it have a screen?\nVisitor: It has one most of the time."},
                    {"role": "assistant", "content": '{"answer":"probably","confidence":0.92}'},
                    {"role": "user", "content": "Question: Can it fit in one hand?\nVisitor: I am not completely sure."},
                    {"role": "assistant", "content": '{"answer":"maybe","confidence":0.96}'},
                    {"role": "user", "content": "Question: Does it use ink?\nVisitor: Not usually."},
                    {"role": "assistant", "content": '{"answer":"probably_not","confidence":0.92}'},
                    {"role": "user", "content": "Question: Does it have a screen?\nVisitor: I like pizza."},
                    {"role": "assistant", "content": '{"answer":"unclear","confidence":0.99}'},
                    {"role": "user", "content": context},
                ],
                GAME_ANSWER_SCHEMA,
                num_predict=35,
                timeout=20,
            )
            answer = str(parsed.get("answer", "unclear"))
            confidence = float(parsed.get("confidence", 0.0))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if (
            answer not in {"yes", "probably", "maybe", "probably_not", "no"}
            or confidence < 0.70
            or not _game_reply_is_relevant(message, question)
        ):
            return None
        return answer

    def rephrase_game_question(
        self,
        canonical_question: str,
        recent_turns: list[dict[str, str]],
    ) -> str | None:
        """Make a catalog question conversational while preserving its yes condition."""
        system = (
            "Rewrite the supplied yes-or-no object question in friendly natural English. Preserve exactly "
            "what a yes means. Ask one question only, use at most 18 words, do not name or guess an object, "
            "and return only JSON."
        )
        context = (
            f"Canonical question: {canonical_question}\n"
            f"Recent turns: {json.dumps(recent_turns[-3:])}"
        )
        try:
            parsed, _ = self._structured_request(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Canonical question: Does it have a screen?"},
                    {"role": "assistant", "content": '{"question":"Does your object have a screen?"}'},
                    {"role": "user", "content": context},
                ],
                GAME_QUESTION_SCHEMA,
                num_predict=45,
                timeout=20,
            )
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
            return None
        question = str(parsed.get("question", "")).strip()
        return question if _valid_game_question(question) and _preserves_question_terms(canonical_question, question) else None

    def suggest_distinguishing_question(
        self,
        new_object: str,
        missed_guess: str,
        recent_turns: list[dict[str, str]],
    ) -> str | None:
        """Suggest reviewable learning evidence; the visitor still has to approve it."""
        system = (
            "Create one simple yes-or-no question that is normally true for the new object and false for "
            "the wrong guess. Ask about one observable use or feature. Do not combine alternatives, mention "
            "either object by name, or make subjective claims. Use at most 18 words and return only JSON."
        )
        context = (
            f"New object: {new_object}\nWrong guess: {missed_guess}\n"
            f"Recent turns: {json.dumps(recent_turns[-5:])}"
        )
        try:
            parsed, _ = self._structured_request(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "New object: umbrella\nWrong guess: book"},
                    {"role": "assistant", "content": '{"question":"Can it keep a person dry in rain?"}'},
                    {"role": "user", "content": context},
                ],
                GAME_QUESTION_SCHEMA,
                num_predict=45,
                timeout=20,
            )
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
            return None
        question = str(parsed.get("question", "")).strip()
        return question if _valid_game_question(question) else None

    def _structured_request(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object],
        *,
        num_predict: int,
        timeout: int,
    ) -> tuple[dict[str, object], str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
            "keep_alive": "10m",
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        raw = response_data["message"]["content"]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama structured response must be an object.")
        return parsed, raw

    def _messages_for(self, message: str) -> list[dict[str, str]]:
        """Always retain the system rules plus the four most recent turns."""
        recent = self.history[1:][-MAX_HISTORY_MESSAGES:]
        return [self.history[0], *recent, {"role": "user", "content": message}]

    @staticmethod
    def _to_result(data: dict[str, object]) -> ConversationResult:
        reply = str(data.get("reply", "Could you say that another way?"))[:280]
        try:
            reaction = Reaction(str(data["reaction"]))
            action = Action(str(data["action"]))
        except (KeyError, ValueError) as error:
            raise ValueError("Ollama response contains an unsupported action or reaction.") from error
        category = data.get("music_category")
        if category not in {None, "calm", "warm", "happy", "energetic", "celebration"}:
            category = None
        return ConversationResult(RobotCommand(reply, reaction, action), category)

    @staticmethod
    def _guard_model_action(message: str, result: ConversationResult) -> ConversationResult:
        """A small model cannot start actions unsupported by the visitor's words."""
        action = result.command.action
        if action == Action.START_GAME and not is_game_request(message):
            action = Action.NONE
        if action == Action.PLAY_MUSIC and not is_music_request(message):
            action = Action.NONE
        music_controls = {
            Action.PAUSE_MUSIC,
            Action.RESUME_MUSIC,
            Action.NEXT_MUSIC,
            Action.STOP_MUSIC,
        }
        if action in music_controls and music_control_action(message) != action:
            action = Action.NONE
        if action == result.command.action:
            return result
        return ConversationResult(
            RobotCommand(result.command.reply, result.command.reaction, action),
            result.music_category if action == Action.PLAY_MUSIC else None,
            result.provider_error,
        )

    @staticmethod
    def _guard_reply_content(message: str, result: ConversationResult) -> ConversationResult:
        """Repair two common 1B-model omissions without inventing factual content."""
        reply = result.command.reply.strip()
        if not is_game_request(message):
            invitation = re.compile(
                r"(?:^|(?<=[.!?])\s+)(?:would you like to|do you want to|shall we|let'?s) "
                r"(?:play|start|try) (?:the |an? )?(?:object )?(?:guessing )?game[^.!?]*[.!?]?",
                re.IGNORECASE,
            )
            reply = invitation.sub("", reply).strip()
        asks_for_question = bool(re.search(r"\bask me\b[^.!?]*\bquestion\b", message, re.IGNORECASE))
        if asks_for_question and "?" not in reply:
            reply = f"{reply} What do you think about that?".strip()
        explicit_emotion = explicit_conversation_reaction(message)
        has_follow_up_request = bool(
            re.search(r"\b(?:tell|explain|show|give|ask|what|why|how|can|could|would)\b", message, re.IGNORECASE)
        )
        acknowledged_compliment = bool(
            re.search(r"\b(?:thank|glad|happy|appreciate|kind of you)\b", reply, re.IGNORECASE)
        )
        if explicit_emotion == Reaction.HAPPY and has_follow_up_request and not acknowledged_compliment:
            reply = f"Thank you! {reply}".strip()
        if not reply:
            reply = "Tell me what you would like to explore next."
        reply = reply[:280]
        reaction = result.command.reaction
        confusion_markers = ("not sure", "do not understand", "don't understand", "say that another way", "cannot")
        if reaction == Reaction.CONFUSED and not any(marker in reply.lower() for marker in confusion_markers):
            reaction = Reaction.SPEAKING
        return ConversationResult(
            RobotCommand(reply, reaction, result.command.action),
            result.music_category,
            result.provider_error,
        )

    @staticmethod
    def _apply_explicit_reaction_rules(message: str, result: ConversationResult) -> ConversationResult:
        """Keep simple visitor-facing reactions stable even with a small local model."""
        emotion = explicit_conversation_reaction(message)
        if emotion is not None and result.command.action == Action.NONE:
            return ConversationResult(
                RobotCommand(result.command.reply, emotion, result.command.action),
                result.music_category,
                result.provider_error,
            )
        greeting = re.search(r"\b(hello|hi|hey)\b", message.lower())
        if greeting and result.command.action == Action.NONE:
            return ConversationResult(
                RobotCommand(result.command.reply, Reaction.HAPPY, result.command.action),
                result.music_category,
                result.provider_error,
            )
        return result


def _valid_game_question(question: str) -> bool:
    """Reject malformed model-generated questions before they reach game state."""
    if not 8 <= len(question) <= 160 or question.count("?") != 1 or not question.endswith("?"):
        return False
    if len(question.split()) > 18 or re.search(r"\b(?:or|versus|vs\.?|guess|probably)\b", question, re.IGNORECASE):
        return False
    return bool(re.match(r"^(?:is|are|does|do|can|could|would|has|have)\b", question, re.IGNORECASE))


def _content_terms(text: str) -> set[str]:
    aliases = {
        "cursor": "pointer",
        "display": "screen",
        "photo": "picture",
        "photos": "picture",
        "pictures": "picture",
        "used": "use",
        "uses": "use",
        "using": "use",
        "moves": "move",
        "moving": "move",
    }
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "does",
        "do",
        "for",
        "have",
        "has",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "one",
        "the",
        "this",
        "that",
        "to",
        "would",
        "your",
    }
    words = re.findall(r"[a-z]+", text.lower())
    return {aliases.get(word, word) for word in words if word not in stop_words}


def _game_reply_is_relevant(message: str, question: str) -> bool:
    if _content_terms(message) & _content_terms(question):
        return True
    return bool(
        re.search(
            r"\b(?:yes|no|maybe|probably|sure|usually|sometimes|rarely|never|"
            r"it does|it doesn't|it is|it isn't|it has|it can|that is|that's|for sure)\b",
            message,
            re.IGNORECASE,
        )
    )


def _preserves_question_terms(canonical: str, generated: str) -> bool:
    expected = _content_terms(canonical)
    if not expected:
        return False
    overlap = expected & _content_terms(generated)
    return len(overlap) / len(expected) >= 0.6
