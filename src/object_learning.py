"""Store human-provided object-game corrections for review before promotion."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def record_suggestion(path: Path, guessed_object: str, new_object: str, distinguishing_question: str) -> None:
    """Append a small correction without silently editing the trusted catalog."""
    suggestion = {
        "created_at": datetime.now(UTC).isoformat(),
        "guessed_object": _clean(guessed_object, "guessed object"),
        "new_object": _clean(new_object, "new object"),
        "distinguishing_question": _clean(distinguishing_question, "question"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(suggestion, ensure_ascii=True) + "\n")


def _clean(value: str, field_name: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > 160:
        raise ValueError(f"The {field_name} must contain 1 to 160 visible characters.")
    return cleaned
