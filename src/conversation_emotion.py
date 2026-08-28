"""Small, explicit conversational-emotion rules for reliable robot reactions."""

from __future__ import annotations

import re

try:
    from .robot_state import Reaction
except ImportError:  # Supports direct execution: python src/chat_console.py
    from robot_state import Reaction


_INSULT = re.compile(r"\b(stupid|dumb|idiot|ugly|useless|annoying|hate you|bad robot|shut up)\b", re.IGNORECASE)
_CONFUSION = re.compile(
    r"\b(i (do not|don't) know|what do you mean|i (do not|don't) understand|can you repeat|huh|i am confused)\b",
    re.IGNORECASE,
)
_INTEREST = re.compile(r"\b(interesting|tell me more|that is wild|that's wild|wow|really)\b", re.IGNORECASE)
_COMPLIMENT = re.compile(r"\b(cute|pretty|beautiful|adorable|nice robot|good job|love you|amazing)\b", re.IGNORECASE)


def explicit_conversation_reaction(message: str) -> Reaction | None:
    """Return a visible reaction only for unambiguous visitor phrasing."""
    if _INSULT.search(message):
        return Reaction.ANNOYED
    if _CONFUSION.search(message):
        return Reaction.CONFUSED
    if _INTEREST.search(message):
        return Reaction.CURIOUS
    if _COMPLIMENT.search(message):
        return Reaction.HAPPY
    return None
