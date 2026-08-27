"""Run one manual object-game QA trial and save normalized evidence locally."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .game_trials import record_trial, summarize_trials
    from .object_game import ObjectGuessingGame
    from .robot_runtime import answer_from_text
except ImportError:  # Supports direct execution: python src/game_trial.py
    from game_trials import record_trial, summarize_trials
    from object_game import ObjectGuessingGame
    from robot_runtime import answer_from_text


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one human QA trial for the object guessing game.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "object_catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "game_trials.jsonl")
    parser.add_argument(
        "--target",
        help="Initial object label; you can correct it after the round. Use --list-targets to view choices.",
    )
    parser.add_argument("--trial-id", default="", help="Optional anonymous label, such as round-01.")
    parser.add_argument("--list-targets", action="store_true")
    args = parser.parse_args()

    catalog = ObjectGuessingGame.from_file(args.catalog)
    targets = {item.name.casefold(): item.name for item in catalog.objects}
    if args.list_targets:
        print("\n".join(sorted(targets.values())))
        return

    target_input = args.target or input("Target object for this QA round (tester only): ")
    target = targets.get(target_input.casefold().strip())
    if target is None:
        raise ValueError("Target must be one of the catalog objects. Use --list-targets to see the list.")

    turns: list[dict[str, str]] = []
    outcome = "low_confidence"
    while not catalog.should_guess() and not catalog.question_budget_reached():
        next_question = catalog.next_question()
        if next_question is None:
            break
        question_id, question = next_question
        raw_answer = input(f"\nRobot: {question}\nAnswer [yes/probably/maybe/probably not/no, or cancel]: ").strip()
        if raw_answer.casefold() in {"cancel", "quit", "stop"}:
            outcome = "cancelled"
            break
        answer = answer_from_text(raw_answer)
        turns.append({"question_id": question_id, "question": question, "answer": answer})
        catalog.answer(question_id, answer)
    else:
        outcome = "guessed" if catalog.should_guess() else "low_confidence"

    guess, confidence = catalog.best_guess()
    if outcome == "guessed":
        print(f"\nRobot final guess: {guess} ({confidence:.0%} confidence)")
    elif outcome == "cancelled":
        print(
            f"\nRound cancelled. Current candidate only: {guess} ({confidence:.0%} confidence). "
            "This was not a final robot guess."
        )
    else:
        print(
            f"\nNo final guess. Current candidate: {guess} ({confidence:.0%} confidence). "
            "The robot needs more evidence."
        )
    actual_target_input = input(f"Object you were actually thinking of [{target}]: ").strip()
    if actual_target_input:
        actual_target = targets.get(actual_target_input.casefold())
        if actual_target is None:
            raise ValueError("Actual object must be one of the catalog objects. Use --list-targets to see the list.")
        target = actual_target
    valid_for_training = False
    if outcome != "cancelled":
        verification = input(
            f"Did every answer describe '{target}'? Save as training data [yes/no]: "
        ).strip()
        valid_for_training = answer_from_text(verification) == "yes"
        if not valid_for_training:
            print("Saved as an exploratory test only. It will not affect training.")
    notes = input("Any confusing question? Press Enter for none: ")
    record_trial(
        args.output,
        target=target,
        turns=turns,
        guess=guess,
        confidence=confidence,
        outcome=outcome,
        trial_id=args.trial_id,
        notes=notes,
        valid_for_training=valid_for_training,
    )
    summary = summarize_trials(args.output)
    print(f"Saved. Completed: {summary['completed']}, correct: {summary['correct']}, low confidence: {summary['low_confidence']}.")


if __name__ == "__main__":
    main()
