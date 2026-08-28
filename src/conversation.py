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
        "music_category": {"type": ["string", "null"], "enum": ["calm", "warm", "energetic", "celebration", None]},
    },
    "required": ["reply", "reaction", "action", "music_category"],
}

SYSTEM_PROMPT = """You are SPIS Robot, a friendly, upbeat English-speaking science-fair robot.
Reply in one or two short sentences. Return only JSON matching the supplied schema.
Use reactions based only on what the visitor explicitly says or does. Never infer identity,
race, ethnicity, nationality, religion, or taste from a face. Use start_game only when the
visitor asks to play the object guessing game. In that game, the visitor thinks of one
object, you ask questions, and you try to guess the visitor's object. Never ask the visitor
to guess an object you are thinking of. Use play_music only when they ask for music or
explicitly name a mood. Use happy for compliments, proud for a correct game guess, annoyed
for direct insults while replying calmly, curious for explicit interest or surprise, listening
only while asking a visitor a question, and speaking for ordinary answers. If unsure, use
reaction confused and action none."""


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


def explicit_action_result(message: str) -> ConversationResult | None:
    """Handle reliable robot commands locally before an LLM can reinterpret them."""
    text = message.lower().strip()
    if is_game_request(message):
        return ConversationResult(
            RobotCommand("Think of one object. I will ask questions and try to guess it.", Reaction.PROUD, Action.START_GAME)
        )
    if any(word in text for word in ("play music", "some music", "song", "music")):
        category = "energetic" if any(word in text for word in ("energy", "energetic", "dance")) else "calm"
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
        messages = [*self.history[-7:], {"role": "user", "content": message}]
        payload = {
            "model": self.model,
            "messages": messages,
            "format": ROBOT_RESPONSE_SCHEMA,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 80},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            raw = response_data["message"]["content"]
            parsed = json.loads(raw)
            result = self._apply_explicit_reaction_rules(message, self._to_result(parsed))
            self.history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": raw}])
            return result
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as error:
            fallback = RuleConversationProvider().respond(message)
            return ConversationResult(fallback.command, fallback.music_category, str(error))

    @staticmethod
    def _to_result(data: dict[str, object]) -> ConversationResult:
        reply = str(data.get("reply", "Could you say that another way?"))[:280]
        try:
            reaction = Reaction(str(data["reaction"]))
            action = Action(str(data["action"]))
        except (KeyError, ValueError) as error:
            raise ValueError("Ollama response contains an unsupported action or reaction.") from error
        category = data.get("music_category")
        if category not in {None, "calm", "warm", "energetic", "celebration"}:
            category = None
        return ConversationResult(RobotCommand(reply, reaction, action), category)

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
