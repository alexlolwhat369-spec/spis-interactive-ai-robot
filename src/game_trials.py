"""Privacy-conscious helpers for human object-game quality trials."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


VALID_ANSWERS = frozenset({"yes", "probably", "maybe", "probably_not", "no"})


def record_trial(
    path: Path,
    *,
    target: str,
    turns: Iterable[dict[str, str]],
    guess: str,
    confidence: float,
    outcome: str,
    trial_id: str = "",
    notes: str = "",
    valid_for_training: bool = False,
) -> None:
    """Append a trial with normalized answers only, never audio or a name."""
    clean_turns = [_clean_turn(turn) for turn in turns]
    if outcome not in {"guessed", "low_confidence", "cancelled"}:
        raise ValueError("Trial outcome must be guessed, low_confidence, or cancelled.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Trial confidence must be between 0 and 1.")
    if not isinstance(valid_for_training, bool):
        raise ValueError("valid_for_training must be true or false.")
    entry = {
        "created_at": datetime.now(UTC).isoformat(),
        "trial_id": _clean_text(trial_id, "trial id", 80),
        "target": _clean_text(target, "target", 160),
        "turns": clean_turns,
        "guess": _clean_text(guess, "guess", 160),
        "confidence": round(confidence, 4),
        "outcome": outcome,
        "correct": outcome == "guessed" and guess.casefold() == target.casefold(),
        "valid_for_training": valid_for_training,
        "notes": _clean_text(notes, "notes", 400),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(entry, ensure_ascii=True) + "\n")


def summarize_trials(path: Path) -> dict[str, int]:
    """Return a compact local QA summary without exposing trial contents."""
    entries = load_trials(path)
    completed = [entry for entry in entries if entry.get("outcome") == "guessed"]
    return {
        "total": len(entries),
        "completed": len(completed),
        "correct": sum(bool(entry.get("correct")) for entry in completed),
        "low_confidence": sum(entry.get("outcome") == "low_confidence" for entry in entries),
        "cancelled": sum(entry.get("outcome") == "cancelled" for entry in entries),
    }


def load_trials(path: Path) -> list[dict[str, object]]:
    """Load local QA records for training or reporting."""
    return _read_entries(path)


def _clean_turn(turn: dict[str, str]) -> dict[str, str]:
    question_id = _clean_text(turn.get("question_id", ""), "question id", 100)
    question = _clean_text(turn.get("question", ""), "question", 300)
    answer = turn.get("answer", "")
    if answer not in VALID_ANSWERS:
        raise ValueError("Trial answers must be normalized object-game answers.")
    return {"question_id": question_id, "question": question, "answer": answer}


def _clean_text(value: str, field_name: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) > limit:
        raise ValueError(f"The {field_name} must contain at most {limit} visible characters.")
    return cleaned


def _read_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid trial JSON on line {line_number}.") from error
        if not isinstance(entry, dict):
            raise ValueError(f"Trial line {line_number} must be an object.")
        entries.append(entry)
    return entries
