import random

import pytest

from mixing_matters.analysis import (
    bootstrap_paired_edges,
    planning_sample_size,
    summarize,
    validate_negative,
    validate_order,
    validate_phase1,
)


def test_discordance_and_planning_table():
    records = []
    for qid, first, middle in (("a", 1, 0), ("b", 0, 1), ("c", 1, 1), ("d", 0, 0)):
        for condition, score in (("gold_first", first), ("gold_middle", middle)):
            records.append(
                {
                    "question_id": qid,
                    "condition": condition,
                    "score": score,
                    "score_normalized_em": score,
                    "score_first_line": score,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                }
            )
    result = summarize(records)
    assert result["discordance"] == 0.5
    assert result["difference_counts"] == {"-1": 1, "0": 2, "1": 1}
    assert result["mean_floor_accuracy"] == 0.1
    assert result["mean_ceiling_accuracy"] == 0.9
    assert result["score_variant_accuracy"]["gold_first"]["score_normalized_em"] == 0.5
    assert result["score_variant_accuracy"]["gold_middle"]["score_first_line"] == 0.5
    assert planning_sample_size(0.20) == 1156
    assert planning_sample_size(0.25) == 1460
    assert planning_sample_size(0.35) == 2068


def test_summarize_skips_null_score_records_and_reports_exclusion():
    records = []
    for qid, first, middle in (("a", 1, 0), ("b", 0, 1), ("c", 1, 1), ("d", 0, 0)):
        for condition, score in (
            ("closed_book", 0),
            ("oracle", 1),
            ("gold_first", first),
            ("gold_middle", middle),
        ):
            records.append(
                {
                    "question_id": qid,
                    "condition": condition,
                    "score": score,
                    "score_normalized_em": score,
                    "score_first_line": score,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                }
            )
    # A question whose scoring failed on every condition: null score, and null
    # floor/ceiling anchors since those are derived from the same scoring path.
    for condition in ("closed_book", "oracle", "gold_first", "gold_middle"):
        records.append(
            {
                "question_id": "failed",
                "condition": condition,
                "score": None,
                "score_normalized_em": None,
                "score_first_line": None,
                "floor_accuracy": None,
                "ceiling_accuracy": None,
            }
        )

    result = summarize(records)

    assert result["excluded_null_score_count"] == 4
    assert result["paired_count"] == 4
    assert result["mean_floor_accuracy"] == 0.1
    assert result["mean_ceiling_accuracy"] == 0.9
    assert result["accuracy"]["closed_book"] == 0
    assert result["accuracy"]["oracle"] == 1
    assert result["score_variant_accuracy"]["closed_book"]["score"] == 0
    assert result["score_variant_accuracy"]["oracle"]["score"] == 1


def test_analysis_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="complete pairs"):
        summarize([{"question_id": "a", "condition": "gold_first", "score": 1}])


def test_phase1_validation_rejects_partial_results():
    with pytest.raises(ValueError, match="incomplete"):
        validate_phase1([])


def _provenance_fields(model_revision: str = "a") -> dict:
    return {
        "model_name": "m",
        "model_revision": model_revision,
        "data_revision": "d",
        "data_sha256": "s",
        "positive_control_sha256": "p",
        "random_seed": 1,
        "software_versions": {
            "python": "p",
            "torch": "t",
            "transformers": "x",
            "cuda": "c",
            "driver": "550.54",
            "gpu": "g",
            "attention_implementation": "eager",
            "dtype": "torch.bfloat16",
        },
    }


def test_phase1_validation_rejects_mixed_provenance():
    records = []
    for index in range(200):
        for condition in ("closed_book", "oracle", "gold_first", "gold_middle"):
            records.append(
                {
                    "question_id": str(index),
                    "source_index": index,
                    "condition": condition,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                    **_provenance_fields(model_revision="a" if index else "b"),
                }
            )
    with pytest.raises(ValueError, match="model_revision"):
        validate_phase1(records)


def test_phase1_validation_rejects_missing_dtype():
    records = []
    for index in range(200):
        for condition in ("closed_book", "oracle", "gold_first", "gold_middle"):
            fields = _provenance_fields()
            if index == 0 and condition == "closed_book":
                del fields["software_versions"]["dtype"]
            records.append(
                {
                    "question_id": str(index),
                    "source_index": index,
                    "condition": condition,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                    **fields,
                }
            )
    with pytest.raises(ValueError, match="software_versions.dtype"):
        validate_phase1(records)


def test_phase1_validation_rejects_mixed_floor_ceiling():
    records = []
    for index in range(200):
        for condition in ("closed_book", "oracle", "gold_first", "gold_middle"):
            # question 0's own gold_middle record disagrees with its other three
            # conditions on ceiling_accuracy: that must be rejected even though
            # ceiling legitimately varies from one question to the next.
            ceiling = 0.5 if (index == 0 and condition == "gold_middle") else 0.9
            records.append(
                {
                    "question_id": str(index),
                    "source_index": index,
                    "condition": condition,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": ceiling,
                    **_provenance_fields(),
                }
            )
    with pytest.raises(ValueError, match="mixed floor/ceiling"):
        validate_phase1(records)


def test_phase1_validation_accepts_complete_matched_run():
    records = []
    for index in range(200):
        for condition in ("closed_book", "oracle", "gold_first", "gold_middle"):
            records.append(
                {
                    "question_id": str(index),
                    "source_index": index,
                    "condition": condition,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                    **_provenance_fields(),
                }
            )
    validate_phase1(records)


def test_bootstrap_paired_edges():
    rng = random.Random(42)
    flat = {str(question): {position: 0.5 for position in range(10)} for question in range(100)}
    primacy, recency = bootstrap_paired_edges(flat, rng, 100)
    assert primacy[0] <= 0 <= primacy[1]
    assert recency[0] <= 0 <= recency[1]

    edged = {
        str(question): {position: 0.8 if position in (0, 1) else 0.4 for position in range(10)}
        for question in range(100)
    }
    primacy, _ = bootstrap_paired_edges(edged, rng, 100)
    assert primacy[0] > 0


def test_validate_negative():
    records = []
    for i in range(100):
        for pos in range(10):
            records.append(
                {
                    "question_id": str(i),
                    "gold_position": pos,
                    "score": 0.5,
                    "floor_accuracy": 0.5,
                    "prompt_token_count": 100,
                }
            )
    validate_negative(records)

    above_floor = [dict(record, score=0.6) for record in records]
    with pytest.raises(ValueError, match="differs from floor"):
        validate_negative(above_floor)

    stretched = []
    for i in range(100):
        for pos in range(10):
            stretched.append(
                {
                    "question_id": str(i),
                    "gold_position": pos,
                    "score": 0.5,
                    "floor_accuracy": 0.5,
                    "prompt_token_count": 100 + (pos % 2),
                }
            )
    with pytest.raises(ValueError, match="non-invariance"):
        validate_negative(stretched)

    edged = [
        dict(record, score=0.8 if record["gold_position"] in (0, 1) else 0.4) for record in records
    ]
    with pytest.raises(ValueError, match="flatness CI for primacy"):
        validate_negative(edged)


def test_validate_order():
    records = [
        {
            "question_id": str(question),
            "gold_position": position,
            "permutation_id": permutation,
            "score": 0.5 + 0.01 * permutation,
            "prompt_token_count": 100,
        }
        for question in range(10)
        for position in (0, 4, 9)
        for permutation in range(3)
    ]
    validate_order(records)

    shifted = [dict(record, score=0.5 + 0.2 * record["permutation_id"]) for record in records]
    with pytest.raises(ValueError, match="spans more than 0.1"):
        validate_order(shifted)

    stretched = [
        dict(record, prompt_token_count=100 + record["permutation_id"]) for record in records
    ]
    with pytest.raises(ValueError, match="changed prompt length"):
        validate_order(stretched)


def test_binary_null_passes_gates():
    rng = random.Random(0)
    negative = []
    for question in range(200):
        floor = float(rng.random() < 0.15)
        for position in range(10):
            negative.append(
                {
                    "question_id": str(question),
                    "gold_position": position,
                    "score": float(rng.random() < 0.15),
                    "floor_accuracy": floor,
                    "prompt_token_count": 1500,
                }
            )
    validate_negative(negative)

    order = [
        {
            "question_id": str(question),
            "gold_position": position,
            "permutation_id": permutation,
            "score": float(rng.random() < 0.40),
            "prompt_token_count": 1500,
        }
        for question in range(200)
        for position in (0, 4, 9)
        for permutation in range(3)
    ]
    validate_order(order)


def test_gates_reject_unscored_units():
    negative = [
        {
            "question_id": str(question),
            "gold_position": position,
            "score": None if question == 3 else 0.5,
            "floor_accuracy": 0.5,
            "prompt_token_count": 100,
        }
        for question in range(10)
        for position in range(10)
    ]
    with pytest.raises(ValueError, match="no scored position"):
        validate_negative(negative)

    order = [
        {
            "question_id": str(question),
            "gold_position": position,
            "permutation_id": permutation,
            "score": None if position == 4 else 0.5,
            "prompt_token_count": 100,
        }
        for question in range(10)
        for position in (0, 4, 9)
        for permutation in range(3)
    ]
    with pytest.raises(ValueError, match="no scored permutation"):
        validate_order(order)
