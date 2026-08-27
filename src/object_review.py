"""Review and explicitly promote object-game suggestions into the catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path


ATTRIBUTE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def load_suggestions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    suggestions: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            suggestion = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid suggestion JSON on line {line_number}.") from error
        if not isinstance(suggestion, dict) or not all(
            isinstance(suggestion.get(field), str)
            for field in ("guessed_object", "new_object", "distinguishing_question")
        ):
            raise ValueError(f"Suggestion line {line_number} is missing required text fields.")
        suggestions.append(suggestion)
    return suggestions


def reject_suggestion(queue_path: Path, index: int) -> dict[str, str]:
    suggestions = load_suggestions(queue_path)
    suggestion = _pop_suggestion(suggestions, index)
    _write_json_lines(queue_path, suggestions)
    return suggestion


def approve_suggestion(
    queue_path: Path,
    catalog_path: Path,
    index: int,
    *,
    category: str,
    attributes: list[str],
    distinguishing_attribute: str,
    question_text: str = "",
) -> dict[str, str]:
    """Add one reviewed suggestion only with an explicit category and schema."""
    suggestions = load_suggestions(queue_path)
    suggestion = _pop_suggestion(suggestions, index)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    objects = catalog.get("objects")
    questions = catalog.get("questions")
    if not isinstance(objects, list) or not isinstance(questions, dict):
        raise ValueError("Catalog must contain objects and questions.")

    known_categories = {item.get("category") for item in objects if isinstance(item, dict)}
    if category not in known_categories:
        raise ValueError("Choose an existing catalog category.")
    name = _clean_name(suggestion["new_object"])
    if any(_clean_name(str(item.get("name", ""))).casefold() == name.casefold() for item in objects if isinstance(item, dict)):
        raise ValueError("That object already exists in the catalog.")
    clean_attributes = _clean_attributes(attributes)
    if distinguishing_attribute not in clean_attributes:
        raise ValueError("The distinguishing attribute must be included in --attributes.")
    if any(
        distinguishing_attribute in item.get("attributes", [])
        for item in objects
        if isinstance(item, dict) and item.get("name") == suggestion["guessed_object"]
    ):
        raise ValueError("The distinguishing attribute must be false for the object the robot guessed.")
    if distinguishing_attribute not in questions:
        cleaned_question = _clean_question(question_text)
        questions[distinguishing_attribute] = {"text": cleaned_question, "category": category}

    signature = frozenset(clean_attributes)
    if any(signature == frozenset(item.get("attributes", [])) for item in objects if isinstance(item, dict)):
        raise ValueError("The proposed attributes duplicate an existing object signature.")
    objects.append({"name": name, "category": category, "attributes": clean_attributes})
    _write_catalog(catalog_path, catalog)
    _write_json_lines(queue_path, suggestions)
    return suggestion


def _pop_suggestion(suggestions: list[dict[str, str]], index: int) -> dict[str, str]:
    if index < 1 or index > len(suggestions):
        raise ValueError(f"Choose a suggestion number from 1 to {len(suggestions)}.")
    return suggestions.pop(index - 1)


def _clean_attributes(attributes: list[str]) -> list[str]:
    cleaned = sorted({attribute.strip() for attribute in attributes if attribute.strip()})
    if not cleaned or any(not ATTRIBUTE_PATTERN.fullmatch(attribute) for attribute in cleaned):
        raise ValueError("Attributes must be comma-separated lowercase snake_case names.")
    return cleaned


def _clean_name(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > 160:
        raise ValueError("Object name must contain 1 to 160 visible characters.")
    return cleaned


def _clean_question(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > 300 or not cleaned.endswith("?"):
        raise ValueError("A new attribute needs a short yes-or-no question ending in '?'.")
    return cleaned


def _write_catalog(path: Path, catalog: dict[str, object]) -> None:
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, entries: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(entry, ensure_ascii=True) + "\n" for entry in entries)
    path.write_text(content, encoding="utf-8")
