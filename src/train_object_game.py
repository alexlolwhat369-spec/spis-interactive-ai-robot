"""Train object-game answer calibration from local human QA rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .object_game_training import DEFAULT_MIN_PER_SIDE, DEFAULT_PRIOR_STRENGTH, train_and_write
except ImportError:  # Supports direct execution: python src/train_object_game.py
    from object_game_training import DEFAULT_MIN_PER_SIDE, DEFAULT_PRIOR_STRENGTH, train_and_write


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the object guessing game from human QA rounds.")
    parser.add_argument("--trials", type=Path, default=ROOT / "data" / "game_trials.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "object_catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "object_game_calibration.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "object_game_training.json")
    parser.add_argument("--min-per-side", type=int, default=DEFAULT_MIN_PER_SIDE)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    calibration, report = train_and_write(
        args.trials,
        args.catalog,
        args.output,
        args.report,
        min_per_side=args.min_per_side,
        prior_strength=args.prior_strength,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "source_trials": report["source_trials"],
                "calibrated_questions": len(calibration["questions"]),
                "holdout": report["holdout"],
                "output": str(args.output),
                "report": str(args.report),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
