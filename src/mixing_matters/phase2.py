"""Phase 2 statistics: does architecture interact with evidence position.

Turns a dense ten-position sweep, run for several models over the same
question set, into position curves and paired bootstrap contrasts. Only
``gold`` condition records (gold_position 0 through 9) feed these
statistics; ``closed_book`` and ``oracle`` records carry the floor and
ceiling anchors that are already attached to every gold record as
``floor_accuracy``/``ceiling_accuracy`` and are not otherwise used here.

The question is the unit of analysis throughout: a bootstrap resample
draws whole questions with replacement, and a drawn question brings all
ten of its positions along. When more than one model is in scope for a
single call, one resample of question ids is drawn per iteration and
reused to recompute every model's contrast in that draw, which is what
makes model comparisons paired.
"""

import random
from collections.abc import Iterable

GOLD_POSITIONS = tuple(range(10))
PRIMACY_POSITIONS = (0, 1)
CENTER_POSITIONS = (4, 5)
RECENCY_POSITIONS = (8, 9)
VALID_CONDITIONS = ("closed_book", "oracle", "gold")
DEFAULT_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260130


def _validate_records(records: Iterable[dict]) -> None:
    for record in records:
        condition = record.get("condition")
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"unknown condition: {condition!r}")

        gold_position = record.get("gold_position")
        if condition == "gold":
            if not isinstance(gold_position, int) or isinstance(gold_position, bool):
                raise ValueError(
                    f"gold_position must be an int for condition gold: {gold_position!r}"
                )
            if gold_position not in GOLD_POSITIONS:
                raise ValueError(f"gold_position out of range 0-9: {gold_position!r}")
        elif gold_position is not None:
            raise ValueError(
                f"gold_position must be null for condition {condition!r}: {gold_position!r}"
            )

        if not record.get("model_key"):
            raise ValueError("model_key is required")
        if not record.get("question_id"):
            raise ValueError("question_id is required")

        score = record.get("score")
        if score is not None and not isinstance(score, (int, float)):
            raise ValueError(f"score must be numeric or null: {score!r}")


def _complete_bundles(
    records: Iterable[dict],
) -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    """Build per-model question bundles from the gold condition records.

    Returns (bundles, excluded_record_count, excluded_question_count, total_question_count).
    ``bundles[model][question_id]`` is a dict of position -> score, present only for
    questions where all ten positions were scored (non-null) for that model.
    """
    records = list(records)
    _validate_records(records)

    raw: dict[str, dict[str, dict[int, float]]] = {}
    excluded_record_count: dict[str, int] = {}
    seen_questions: dict[str, set[str]] = {}

    for record in records:
        if record["condition"] != "gold":
            continue
        model = record["model_key"]
        question_id = record["question_id"]
        seen_questions.setdefault(model, set()).add(question_id)

        if record["score"] is None:
            excluded_record_count[model] = excluded_record_count.get(model, 0) + 1
            continue

        positions = raw.setdefault(model, {}).setdefault(question_id, {})
        position = record["gold_position"]
        if position in positions:
            raise ValueError(
                f"duplicate gold record for {model}/{question_id} at position {position}"
            )
        positions[position] = float(record["score"])

    bundles: dict[str, dict[str, dict[int, float]]] = {}
    excluded_question_count: dict[str, int] = {}
    total_question_count: dict[str, int] = {}

    for model, question_ids in seen_questions.items():
        total_question_count[model] = len(question_ids)
        excluded_record_count.setdefault(model, 0)
        complete = {}
        excluded = 0
        for question_id in question_ids:
            positions = raw.get(model, {}).get(question_id, {})
            if set(positions) == set(GOLD_POSITIONS):
                complete[question_id] = positions
            else:
                excluded += 1
        bundles[model] = complete
        excluded_question_count[model] = excluded

    return bundles, excluded_record_count, excluded_question_count, total_question_count


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    return values[int(0.025 * n)], values[int(0.975 * n)]


def _two_sided_p_value(values: list[float]) -> float:
    n = len(values)
    below = sum(1 for value in values if value <= 0) / n
    above = sum(1 for value in values if value >= 0) / n
    return min(1.0, 2 * min(below, above))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down correction, returned in the same order as ``p_values``."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        remaining = len(p_values) - rank
        running_max = max(running_max, min(1.0, p_values[index] * remaining))
        adjusted[index] = running_max
    return adjusted


def _resamples(question_ids: list[str], n_resamples: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [rng.choices(question_ids, k=len(question_ids)) for _ in range(n_resamples)]


def _pooled_mean(
    bundle: dict[str, dict[int, float]], question_ids: list[str], positions: tuple[int, ...]
) -> float:
    scores = [
        bundle[question_id][position] for question_id in question_ids for position in positions
    ]
    return _mean(scores)


def _edge_value(
    bundle: dict[str, dict[int, float]], question_ids: list[str], edge_positions: tuple[int, ...]
) -> float:
    return _pooled_mean(bundle, question_ids, edge_positions) - _pooled_mean(
        bundle, question_ids, CENTER_POSITIONS
    )


def _contrast(estimate: float, distribution: list[float]) -> dict[str, float]:
    ci_low, ci_high = _percentile_ci(distribution)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": _two_sided_p_value(distribution),
    }


def _attach_holm(primacy: dict[str, float], recency: dict[str, float]) -> None:
    holm_primacy, holm_recency = holm_adjust([primacy["p_value"], recency["p_value"]])
    primacy["p_value_holm"] = holm_primacy
    recency["p_value_holm"] = holm_recency


def position_curve(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Per model_key, accuracy and a 95% CI at each gold position 0-9.

    Returns, for each model_key::

        {
            "positions": {
                position: {"accuracy": float, "ci_low": float, "ci_high": float,
                            "question_count": int}
                for position in range(10)
            },
            "excluded_record_count": int,
            "excluded_question_count": int,
        }

    ``question_count`` is the number of questions with all ten gold positions
    scored for that model; it is the same for every position because a
    question missing any one position is excluded from all of them.
    """
    bundles, excluded_record_count, excluded_question_count, _ = _complete_bundles(records)

    result = {}
    for model, bundle in bundles.items():
        question_ids = sorted(bundle)
        resamples = _resamples(question_ids, n_resamples, BOOTSTRAP_SEED)

        positions = {}
        for position in GOLD_POSITIONS:
            values = [bundle[question_id][position] for question_id in question_ids]
            accuracy = _mean(values)
            distribution = [
                _mean([bundle[question_id][position] for question_id in sample])
                for sample in resamples
            ]
            ci_low, ci_high = _percentile_ci(distribution) if distribution else (accuracy, accuracy)
            positions[position] = {
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "question_count": len(question_ids),
            }

        result[model] = {
            "positions": positions,
            "excluded_record_count": excluded_record_count.get(model, 0),
            "excluded_question_count": excluded_question_count.get(model, 0),
        }
    return result


def edges(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Per model_key, the primacy and recency edge contrasts.

    primacy = mean accuracy over positions 0,1 minus mean accuracy over positions 4,5
    recency = mean accuracy over positions 8,9 minus mean accuracy over positions 4,5

    Returns, for each model_key::

        {
            "primacy": {"estimate": float, "ci_low": float, "ci_high": float,
                        "p_value": float, "p_value_holm": float},
            "recency": {...},
            "question_count": int,
            "excluded_record_count": int,
            "excluded_question_count": int,
        }

    When several models are present, one resample of question ids is drawn
    per iteration and reused for every model's contrast in that draw, so
    ``question_count`` is the intersection of the models' complete question
    sets and Holm correction is applied across the two edges within a model.
    """
    bundles, excluded_record_count, _, total_question_count = _complete_bundles(records)
    models = sorted(bundles)
    if not models:
        raise ValueError("no gold records found")

    universe = sorted(set.intersection(*(set(bundles[model]) for model in models)))
    resamples = _resamples(universe, n_resamples, BOOTSTRAP_SEED)

    result = {}
    for model in models:
        bundle = bundles[model]
        primacy_estimate = _edge_value(bundle, universe, PRIMACY_POSITIONS)
        recency_estimate = _edge_value(bundle, universe, RECENCY_POSITIONS)
        primacy_distribution = [
            _edge_value(bundle, sample, PRIMACY_POSITIONS) for sample in resamples
        ]
        recency_distribution = [
            _edge_value(bundle, sample, RECENCY_POSITIONS) for sample in resamples
        ]

        primacy = _contrast(primacy_estimate, primacy_distribution)
        recency = _contrast(recency_estimate, recency_distribution)
        _attach_holm(primacy, recency)

        result[model] = {
            "primacy": primacy,
            "recency": recency,
            "question_count": len(universe),
            "excluded_record_count": excluded_record_count.get(model, 0),
            "excluded_question_count": total_question_count.get(model, 0) - len(universe),
        }
    return result


def interaction(
    records: Iterable[dict],
    first_model: str,
    second_model: str,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """Difference in primacy and recency edges between two models.

    This is the paired comparison that answers whether architecture and
    evidence position interact: both models' edges are recomputed from the
    same bootstrap draw of question ids in every resample.

    Returns::

        {
            "primacy": {"estimate": float, "ci_low": float, "ci_high": float,
                        "p_value": float, "p_value_holm": float},
            "recency": {...},
            "question_count": int,
            "excluded_record_count": {first_model: int, second_model: int},
            "excluded_question_count": {first_model: int, second_model: int},
        }

    ``estimate`` is ``first_model``'s edge minus ``second_model``'s edge.
    """
    bundles, excluded_record_count, _, total_question_count = _complete_bundles(records)
    for model in (first_model, second_model):
        if model not in bundles:
            raise ValueError(f"model not found in records: {model!r}")

    universe = sorted(set(bundles[first_model]) & set(bundles[second_model]))
    if not universe:
        raise ValueError(f"no questions complete for both {first_model!r} and {second_model!r}")
    resamples = _resamples(universe, n_resamples, BOOTSTRAP_SEED)

    def edge_diff(question_ids: list[str], edge_positions: tuple[int, ...]) -> float:
        first = _edge_value(bundles[first_model], question_ids, edge_positions)
        second = _edge_value(bundles[second_model], question_ids, edge_positions)
        return first - second

    primacy_estimate = edge_diff(universe, PRIMACY_POSITIONS)
    recency_estimate = edge_diff(universe, RECENCY_POSITIONS)
    primacy_distribution = [edge_diff(sample, PRIMACY_POSITIONS) for sample in resamples]
    recency_distribution = [edge_diff(sample, RECENCY_POSITIONS) for sample in resamples]

    primacy = _contrast(primacy_estimate, primacy_distribution)
    recency = _contrast(recency_estimate, recency_distribution)
    _attach_holm(primacy, recency)

    return {
        "primacy": primacy,
        "recency": recency,
        "question_count": len(universe),
        "excluded_record_count": {
            first_model: excluded_record_count.get(first_model, 0),
            second_model: excluded_record_count.get(second_model, 0),
        },
        "excluded_question_count": {
            first_model: total_question_count.get(first_model, 0) - len(universe),
            second_model: total_question_count.get(second_model, 0) - len(universe),
        },
    }
