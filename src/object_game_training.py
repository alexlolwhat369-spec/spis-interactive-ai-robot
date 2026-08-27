"""Train calibrated object-game answer likelihoods from local human QA rounds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

try:
    from .game_trials import VALID_ANSWERS, load_trials
    from .object_game import Answer, ObjectGuessingGame, QuestionLikelihoods, default_likelihoods
except ImportError:  # Supports direct execution: python src/train_object_game.py
    from game_trials import VALID_ANSWERS, load_trials
    from object_game import Answer, ObjectGuessingGame, QuestionLikelihoods, default_likelihoods


SCHEMA_VERSION = 1
DEFAULT_MIN_PER_SIDE = 8
DEFAULT_PRIOR_STRENGTH = 8.0


def train_and_write(
    trials_path: Path,
    catalog_path: Path,
    calibration_path: Path,
    report_path: Path,
    *,
    min_per_side: int = DEFAULT_MIN_PER_SIDE,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    dry_run: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Calibrate answer likelihoods and write reproducible local artifacts."""
    if min_per_side < 1:
        raise ValueError("min_per_side must be at least one.")
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive.")
    catalog = ObjectGuessingGame.from_file(catalog_path)
    usable, ignored = _usable_trials(load_trials(trials_path), catalog)
    train_trials, holdout_trials = _split_trials(usable)
    evaluation_calibration, _ = build_calibration(train_trials, catalog, min_per_side, prior_strength)
    final_calibration, question_metrics = build_calibration(usable, catalog, min_per_side, prior_strength)
    calibration_document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_trials": len(usable),
        "min_per_side": min_per_side,
        "prior_strength": prior_strength,
        "questions": _serialize_calibration(final_calibration, question_metrics),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_trials": len(usable),
        "ignored_trials": ignored,
        "training_trials": len(train_trials),
        "holdout_trials": len(holdout_trials),
        "calibrated_questions": len(final_calibration),
        "holdout": {
            "baseline": evaluate_trials(holdout_trials, catalog, {}),
            "calibrated": evaluate_trials(holdout_trials, catalog, evaluation_calibration),
        },
        "question_metrics": question_metrics,
        "limitation": (
            "A calibration is only used when both truth sides meet min_per_side. "
            "The holdout replay evaluates answer likelihoods, not a fresh live conversation."
        ),
    }
    if not dry_run:
        _write_json(calibration_path, calibration_document)
        _write_json(report_path, report)
    return calibration_document, report


def build_calibration(
    trials: Iterable[dict[str, object]],
    catalog: ObjectGuessingGame,
    min_per_side: int,
    prior_strength: float,
) -> tuple[dict[str, QuestionLikelihoods], dict[str, dict[str, object]]]:
    """Estimate P(answer | catalog truth) per question with a fallback prior."""
    counts: dict[str, dict[bool, Counter[str]]] = defaultdict(lambda: {True: Counter(), False: Counter()})
    for trial in trials:
        target = _target_for_trial(trial, catalog)
        if target is None:
            continue
        turns = trial.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            question_id = turn.get("question_id")
            answer = turn.get("answer")
            if not isinstance(question_id, str) or answer not in VALID_ANSWERS or question_id not in catalog.questions:
                continue
            definition = catalog.questions[question_id]
            truth = target.category == definition.category if definition.category_probe else definition.yes_attribute in target.attributes
            counts[question_id][truth][str(answer)] += 1

    baseline = default_likelihoods()
    calibrated: dict[str, QuestionLikelihoods] = {}
    metrics: dict[str, dict[str, object]] = {}
    for question_id in catalog.questions:
        present_counts = counts[question_id][True]
        absent_counts = counts[question_id][False]
        present_total = sum(present_counts.values())
        absent_total = sum(absent_counts.values())
        uncertain_total = sum(present_counts[answer] + absent_counts[answer] for answer in ("probably", "maybe", "probably_not"))
        total = present_total + absent_total
        eligible = present_total >= min_per_side and absent_total >= min_per_side
        metrics[question_id] = {
            "present_samples": present_total,
            "absent_samples": absent_total,
            "uncertainty_rate": round(uncertain_total / total, 4) if total else None,
            "calibrated": eligible,
        }
        if not eligible:
            continue
        calibrated[question_id] = QuestionLikelihoods(
            _smoothed_distribution(present_counts, baseline.if_present, prior_strength),
            _smoothed_distribution(absent_counts, baseline.if_absent, prior_strength),
        )
    return calibrated, metrics


def evaluate_trials(
    trials: Iterable[dict[str, object]],
    catalog: ObjectGuessingGame,
    calibration: dict[str, QuestionLikelihoods],
) -> dict[str, object]:
    evaluated = 0
    correct = 0
    confidence_total = 0.0
    for trial in trials:
        target = _target_for_trial(trial, catalog)
        turns = trial.get("turns")
        if target is None or not isinstance(turns, list):
            continue
        game = ObjectGuessingGame(list(catalog.objects), catalog.questions, calibration)
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            question_id = turn.get("question_id")
            answer = turn.get("answer")
            if isinstance(question_id, str) and question_id in game.questions and answer in VALID_ANSWERS:
                game.answer(question_id, str(answer))
        guess, confidence = game.best_guess()
        evaluated += 1
        correct += guess.casefold() == target.name.casefold()
        confidence_total += confidence
    return {
        "trials": evaluated,
        "exact_accuracy": round(correct / evaluated, 4) if evaluated else None,
        "mean_confidence": round(confidence_total / evaluated, 4) if evaluated else None,
    }


def _usable_trials(trials: Iterable[dict[str, object]], catalog: ObjectGuessingGame) -> tuple[list[dict[str, object]], int]:
    usable: list[dict[str, object]] = []
    ignored = 0
    for trial in trials:
        if (
            trial.get("outcome") == "cancelled"
            or trial.get("valid_for_training") is not True
            or _target_for_trial(trial, catalog) is None
        ):
            ignored += 1
            continue
        usable.append(trial)
    return usable, ignored


def _split_trials(trials: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    training: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for index, trial in enumerate(trials):
        identity = f"{trial.get('trial_id', '')}|{trial.get('created_at', '')}|{trial.get('target', '')}|{index}"
        bucket = hashlib.sha256(identity.encode("utf-8")).digest()[0] % 5
        (holdout if bucket == 0 else training).append(trial)
    return training, holdout


def _target_for_trial(trial: dict[str, object], catalog: ObjectGuessingGame):
    target_name = trial.get("target")
    if not isinstance(target_name, str):
        return None
    return next((item for item in catalog.objects if item.name.casefold() == target_name.casefold()), None)


def _smoothed_distribution(
    counts: Counter[str],
    prior: dict[Answer, float] | object,
    prior_strength: float,
) -> dict[Answer, float]:
    prior_map = dict(prior)
    total = sum(counts.values()) + prior_strength
    return {
        answer: (counts[answer] + prior_strength * float(prior_map[answer])) / total
        for answer in prior_map
    }


def _serialize_calibration(
    calibration: dict[str, QuestionLikelihoods],
    metrics: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        question_id: {
            "if_present": dict(likelihoods.if_present),
            "if_absent": dict(likelihoods.if_absent),
            "present_samples": metrics[question_id]["present_samples"],
            "absent_samples": metrics[question_id]["absent_samples"],
        }
        for question_id, likelihoods in calibration.items()
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
