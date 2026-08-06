import random

import pytest

from mixing_matters.phase2 import edges, holm_adjust, interaction, position_curve

FLOOR = 0.1
CEILING = 0.9


def _gold_record(model_key, question_id, position, score):
    return {
        "model_key": model_key,
        "question_id": question_id,
        "condition": "gold",
        "gold_position": position,
        "score": score,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
    }


def _question_records(model_key, question_id, scores_by_position):
    return [
        _gold_record(model_key, question_id, position, score)
        for position, score in scores_by_position.items()
    ]


def _uniform_bundle(model_key, question_count, edge_score, center_score):
    """All questions get the same edge/center split: no noise, a clean planted effect."""
    positions = {
        0: edge_score,
        1: edge_score,
        2: center_score,
        3: center_score,
        4: center_score,
        5: center_score,
        6: center_score,
        7: center_score,
        8: center_score,
        9: center_score,
    }
    records = []
    for question in range(question_count):
        records += _question_records(model_key, f"q{question}", positions)
    return records


def _noisy_records(model_key, question_count, primacy_p, center_p, recency_p, seed):
    rng = random.Random(seed)
    records = []
    for question in range(question_count):
        primacy_score = 1.0 if rng.random() < primacy_p else 0.0
        center_score = 1.0 if rng.random() < center_p else 0.0
        recency_score = 1.0 if rng.random() < recency_p else 0.0
        positions = {
            0: primacy_score,
            1: primacy_score,
            2: center_score,
            3: center_score,
            4: center_score,
            5: center_score,
            6: center_score,
            7: center_score,
            8: recency_score,
            9: recency_score,
        }
        records += _question_records(model_key, f"q{question}", positions)
    return records


def test_edges_detects_planted_primacy_effect_without_recency_effect():
    records = _noisy_records(
        "model_a", question_count=300, primacy_p=0.85, center_p=0.30, recency_p=0.30, seed=7
    )
    result = edges(records, n_resamples=500)["model_a"]

    assert result["primacy"]["ci_low"] > 0
    assert result["recency"]["ci_low"] <= 0 <= result["recency"]["ci_high"]
    assert result["question_count"] == 300
    assert result["excluded_record_count"] == 0
    assert result["excluded_question_count"] == 0


def test_position_curve_accuracy_and_counts_on_uniform_data():
    records = _uniform_bundle("model_a", question_count=50, edge_score=1.0, center_score=0.0)
    result = position_curve(records, n_resamples=100)["model_a"]

    assert result["positions"][0]["accuracy"] == 1.0
    assert result["positions"][4]["accuracy"] == 0.0
    for position in range(10):
        assert result["positions"][position]["question_count"] == 50
    assert result["excluded_record_count"] == 0
    assert result["excluded_question_count"] == 0


def test_determinism_across_repeated_calls_and_shuffled_order():
    records = _noisy_records(
        "model_a", question_count=80, primacy_p=0.8, center_p=0.4, recency_p=0.3, seed=11
    )

    first = edges(records, n_resamples=200)
    second = edges(records, n_resamples=200)
    assert first == second

    shuffled = list(records)
    random.Random(3).shuffle(shuffled)
    third = edges(shuffled, n_resamples=200)
    assert first == third

    curve_first = position_curve(records, n_resamples=200)
    curve_shuffled = position_curve(shuffled, n_resamples=200)
    assert curve_first == curve_shuffled


def test_interaction_is_paired_and_tight_for_a_constant_offset():
    base = _noisy_records(
        "model_a", question_count=150, primacy_p=0.8, center_p=0.35, recency_p=0.35, seed=5
    )
    offset_records = []
    for record in base:
        offset_record = dict(record, model_key="model_b", score=record["score"] + 5.0)
        offset_records.append(offset_record)

    result = interaction(base + offset_records, "model_a", "model_b", n_resamples=300)

    assert abs(result["primacy"]["estimate"]) < 1e-9
    assert abs(result["recency"]["estimate"]) < 1e-9
    assert result["primacy"]["ci_high"] - result["primacy"]["ci_low"] < 1e-9
    assert result["recency"]["ci_high"] - result["recency"]["ci_low"] < 1e-9
    assert result["question_count"] == 150


def test_holm_adjust_matches_hand_computed_values():
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm_adjust([0.01, 0.20]) == pytest.approx([0.02, 0.20])
    # Both unadjusted p-values are large enough that Holm clamps both to 1.0,
    # and the running-max step keeps the result monotone.
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_exclusion_accounting_for_null_scores_and_missing_positions():
    records = _uniform_bundle("model_a", question_count=10, edge_score=1.0, center_score=0.0)

    # q0 has one null-scored position: excluded as a record and as an incomplete question.
    records = [
        dict(record, score=None)
        if record["question_id"] == "q0" and record["gold_position"] == 3
        else record
        for record in records
    ]
    # q1 is missing its position-7 record entirely: excluded as an incomplete question only.
    records = [
        record
        for record in records
        if not (record["question_id"] == "q1" and record["gold_position"] == 7)
    ]

    curve = position_curve(records, n_resamples=50)["model_a"]
    assert curve["excluded_record_count"] == 1
    assert curve["excluded_question_count"] == 2
    assert curve["positions"][0]["question_count"] == 8

    contrasts = edges(records, n_resamples=50)["model_a"]
    assert contrasts["excluded_record_count"] == 1
    assert contrasts["excluded_question_count"] == 2
    assert contrasts["question_count"] == 8


def test_interaction_rejects_missing_model():
    records = _uniform_bundle("model_a", question_count=5, edge_score=1.0, center_score=0.0)
    with pytest.raises(ValueError, match="model not found"):
        interaction(records, "model_a", "model_b", n_resamples=20)


def test_rejects_gold_position_out_of_range():
    records = _uniform_bundle("model_a", question_count=5, edge_score=1.0, center_score=0.0)
    records[0] = dict(records[0], gold_position=10)
    with pytest.raises(ValueError, match="gold_position"):
        position_curve(records, n_resamples=20)
