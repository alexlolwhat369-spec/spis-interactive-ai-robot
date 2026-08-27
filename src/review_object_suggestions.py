"""List, reject, or explicitly approve reviewed object-game suggestions."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .object_review import approve_suggestion, load_suggestions, reject_suggestion
except ImportError:  # Supports direct execution: python src/review_object_suggestions.py
    from object_review import approve_suggestion, load_suggestions, reject_suggestion


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pending object-game learning suggestions.")
    parser.add_argument("--queue", type=Path, default=ROOT / "data" / "pending_object_suggestions.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "object_catalog.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--approve", type=int, metavar="NUMBER")
    action.add_argument("--reject", type=int, metavar="NUMBER")
    parser.add_argument("--category", help="Existing category required for --approve.")
    parser.add_argument("--attributes", help="Comma-separated attributes required for --approve.")
    parser.add_argument("--attribute", help="New or existing distinguishing attribute required for --approve.")
    parser.add_argument("--question", help="Question for a new --attribute; it must end in '?'.")
    args = parser.parse_args()

    if args.approve is not None:
        if not args.category or not args.attributes or not args.attribute:
            parser.error("--approve needs --category, --attributes, and --attribute.")
        suggestion = approve_suggestion(
            args.queue,
            args.catalog,
            args.approve,
            category=args.category,
            attributes=args.attributes.split(","),
            distinguishing_attribute=args.attribute,
            question_text=args.question or "",
        )
        print(f"Approved and added: {suggestion['new_object']}")
        return
    if args.reject is not None:
        suggestion = reject_suggestion(args.queue, args.reject)
        print(f"Rejected: {suggestion['new_object']}")
        return

    suggestions = load_suggestions(args.queue)
    if not suggestions:
        print("No pending suggestions.")
        return
    for number, suggestion in enumerate(suggestions, start=1):
        print(f"{number}. {suggestion['new_object']} instead of {suggestion['guessed_object']}")
        print(f"   Proposed question: {suggestion['distinguishing_question']}")


if __name__ == "__main__":
    main()
