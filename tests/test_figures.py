import json
import random

import pytest

from mixing_matters.figures import (
    PHASE1_CONDITIONS,
    kv_position_curve,
    phase1_condition_accuracy,
    phase2_summary,
    write_figures,
    write_phase2_figures,
)
from mixing_matters.phase2 import edges as phase2_edges
from mixing_matters.phase2 import position_curve as phase2_position_curve


def _kv_records(n_bundles: int = 40, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for control_id in range(n_bundles):
        for slot in range(10):
            probability = 0.9 if slot in (0, 9) else 0.3
            score = float(rng.random() < probability)
            records.append(
                {"control_id": control_id, "condition": f"kv_position_{slot}", "score": score}
            )
    return records


def _phase1_records(n_questions: int = 60, n_anchors: int = 20, seed: int = 2) -> list[dict]:
    rng = random.Random(seed)
    probabilities = {
        "closed_book": 0.3,
        "oracle": 0.9,
        "gold_first": 0.8,
        "gold_middle": 0.5,
    }
    records = []
    for qid in range(n_questions):
        floor = float(rng.random() < 0.3)
        ceiling = float(rng.random() < 0.9)
        for condition, probability in probabilities.items():
            if condition in ("closed_book", "oracle") and qid >= n_anchors:
                continue
            score = float(rng.random() < probability)
            records.append(
                {
                    "question_id": str(qid),
                    "condition": condition,
                    "score": score,
                    "score_normalized_em": score,
                    "score_first_line": score,
                    "floor_accuracy": floor,
                    "ceiling_accuracy": ceiling,
                }
            )
    return records


def test_kv_position_curve_is_deterministic():
    records = _kv_records()
    first = kv_position_curve(records)
    second = kv_position_curve(records)
    assert first == second


def test_kv_position_curve_detects_known_edge_effect():
    records = _kv_records()
    result = kv_position_curve(records)
    assert result["n_bundles"] == 40
    assert result["slots"][0]["accuracy"] > result["slots"][4]["accuracy"]
    edge_low, _ = result["slots"][0]["ci"]
    _, middle_high = result["slots"][4]["ci"]
    assert edge_low > middle_high


def test_kv_position_curve_rejects_incomplete_bundle():
    records = [
        record for record in _kv_records(n_bundles=1) if record["condition"] != "kv_position_5"
    ]
    with pytest.raises(ValueError, match="missing kv-position slots"):
        kv_position_curve(records)


def test_kv_position_curve_rejects_unknown_condition():
    with pytest.raises(ValueError, match="unexpected kv-position condition"):
        kv_position_curve([{"control_id": 0, "condition": "gold_first", "score": 1.0}])


def test_phase1_condition_accuracy_is_deterministic():
    records = _phase1_records()
    first = phase1_condition_accuracy(records)
    second = phase1_condition_accuracy(records)
    assert first == second


def test_phase1_condition_accuracy_orders_conditions_correctly():
    records = _phase1_records()
    result = phase1_condition_accuracy(records)
    assert result["n_questions"] == 60
    oracle = result["conditions"]["oracle"]["primary"]
    closed_book = result["conditions"]["closed_book"]["primary"]
    assert oracle["accuracy"] > closed_book["accuracy"]
    assert oracle["ci"][0] > closed_book["ci"][1]
    # The scoring variants mirror the primary score in this synthetic fixture.
    assert result["conditions"]["oracle"]["normalized_em"]["accuracy"] == oracle["accuracy"]
    assert result["conditions"]["oracle"]["first_line"]["accuracy"] == oracle["accuracy"]


def test_phase1_condition_accuracy_rejects_missing_condition():
    records = [record for record in _phase1_records() if record["condition"] != "oracle"]
    with pytest.raises(ValueError, match="oracle"):
        phase1_condition_accuracy(records)


def _paired_contrast_records(n_questions: int = 200, seed: int = 3) -> list[dict]:
    # Both conditions share a per-question "difficulty" draw, so their marginal
    # means vary together with a wide spread while the paired difference
    # (a small, mostly-fixed offset plus tiny independent noise) has very low
    # variance. This mirrors the real correlation between gold_first and
    # gold_middle, which both score the same question.
    rng = random.Random(seed)
    records = []
    for qid in range(n_questions):
        difficulty = rng.uniform(0.0, 1.0)
        first = difficulty + rng.uniform(-0.02, 0.02)
        middle = difficulty - 0.05 + rng.uniform(-0.02, 0.02)
        for condition, score in (
            ("closed_book", difficulty * 0.5),
            ("oracle", difficulty),
            ("gold_first", first),
            ("gold_middle", middle),
        ):
            records.append(
                {
                    "question_id": str(qid),
                    "condition": condition,
                    "score": score,
                    "score_normalized_em": score,
                    "score_first_line": score,
                }
            )
    return records


def test_phase1_condition_accuracy_reports_paired_contrast_ci():
    records = _paired_contrast_records()
    result = phase1_condition_accuracy(records)

    diff = result["gold_first_minus_gold_middle"]
    assert diff["estimate"] > 0
    assert diff["ci"][0] > 0, "paired difference CI should exclude zero"

    first = result["conditions"]["gold_first"]["primary"]
    middle = result["conditions"]["gold_middle"]["primary"]
    assert first["ci"][0] < middle["ci"][1], "marginal intervals should overlap"


def test_write_figures_creates_expected_files(tmp_path):
    kv_records = _kv_records()
    phase1_records = _phase1_records()
    paths = write_figures(kv_records, phase1_records, tmp_path / "figures")
    for path in paths:
        assert path.exists()
    assert {path.name for path in paths} == {
        "kv-position-curve.png",
        "phase1-condition-accuracy.png",
        "figures-summary.json",
    }
    summary_path = next(path for path in paths if path.name == "figures-summary.json")
    summary = json.loads(summary_path.read_text())
    assert set(summary) == {"kv_position_curve", "phase1_condition_accuracy"}
    assert set(summary["phase1_condition_accuracy"]["conditions"]) == set(PHASE1_CONDITIONS)


def test_write_figures_refuses_to_overwrite(tmp_path):
    kv_records = _kv_records()
    phase1_records = _phase1_records()
    directory = tmp_path / "figures"
    write_figures(kv_records, phase1_records, directory)
    with pytest.raises(FileExistsError):
        write_figures(kv_records, phase1_records, directory)


def _phase2_gold_records(
    model_key: str,
    question_count: int,
    primacy_p: float,
    center_p: float,
    recency_p: float,
    seed: int,
) -> list[dict]:
    """A model's ten-position sweep with an independent coin flip per position band.

    Every question gets the same primacy/center/recency draw across its two
    positions in that band, mirroring a real sweep where nearby positions are
    correlated within a question.
    """
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
        question_id = f"q{question}"
        for position, score in positions.items():
            records.append(
                {
                    "model_key": model_key,
                    "question_id": question_id,
                    "condition": "gold",
                    "gold_position": position,
                    "score": score,
                    "floor_accuracy": 0.1,
                    "ceiling_accuracy": 0.9,
                }
            )
    return records


def _three_model_sweep(question_count: int = 300) -> list[dict]:
    """model_a has a planted primacy effect; model_b and model_c are both null."""
    records = []
    records += _phase2_gold_records(
        "model_a", question_count, primacy_p=0.85, center_p=0.30, recency_p=0.30, seed=101
    )
    records += _phase2_gold_records(
        "model_b", question_count, primacy_p=0.30, center_p=0.30, recency_p=0.30, seed=202
    )
    records += _phase2_gold_records(
        "model_c", question_count, primacy_p=0.30, center_p=0.30, recency_p=0.30, seed=303
    )
    return records


def _find_interaction(summary: dict, first_model: str, second_model: str) -> dict:
    return next(
        entry
        for entry in summary["interactions"]
        if entry["first_model"] == first_model and entry["second_model"] == second_model
    )


def test_phase2_summary_is_deterministic_across_calls_and_shuffled_order():
    records = _three_model_sweep(question_count=80)

    first = phase2_summary(records, n_resamples=200)
    second = phase2_summary(records, n_resamples=200)
    assert first == second

    shuffled = list(records)
    random.Random(9).shuffle(shuffled)
    third = phase2_summary(shuffled, n_resamples=200)
    assert first == third


def test_phase2_summary_detects_interaction_only_for_the_model_with_the_planted_effect():
    records = _three_model_sweep()
    summary = phase2_summary(records, n_resamples=500)

    planted = _find_interaction(summary, "model_a", "model_b")
    assert planted["primacy"]["ci_low"] > 0

    null_pair = _find_interaction(summary, "model_b", "model_c")
    assert null_pair["primacy"]["ci_low"] <= 0 <= null_pair["primacy"]["ci_high"]


def test_phase2_summary_reports_same_counts_as_phase2():
    records = _three_model_sweep(question_count=60)
    summary = phase2_summary(records, n_resamples=50)

    expected_curve = phase2_position_curve(records, n_resamples=50)
    expected_edges = phase2_edges(records, n_resamples=50)

    for model in summary["models"]:
        curve_positions = summary["position_curve"][model]["positions"]
        expected_positions = expected_curve[model]["positions"]
        for position in range(10):
            assert (
                curve_positions[position]["question_count"]
                == expected_positions[position]["question_count"]
            )
        assert (
            summary["position_curve"][model]["excluded_record_count"]
            == expected_curve[model]["excluded_record_count"]
        )
        assert (
            summary["position_curve"][model]["excluded_question_count"]
            == expected_curve[model]["excluded_question_count"]
        )
        assert summary["edges"][model]["question_count"] == expected_edges[model]["question_count"]
        assert (
            summary["edges"][model]["excluded_record_count"]
            == expected_edges[model]["excluded_record_count"]
        )
        assert (
            summary["edges"][model]["excluded_question_count"]
            == expected_edges[model]["excluded_question_count"]
        )


def test_write_phase2_figures_creates_expected_files(tmp_path):
    records = _three_model_sweep(question_count=60)
    paths = write_phase2_figures(records, tmp_path / "phase2-figures", n_resamples=50)
    for path in paths:
        assert path.exists()
    assert {path.name for path in paths} == {
        "position-curves.png",
        "position-edges.png",
        "phase2-summary.json",
    }
    summary_path = next(path for path in paths if path.name == "phase2-summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["models"] == ["model_a", "model_b", "model_c"]
    assert len(summary["interactions"]) == 3


def test_write_phase2_figures_refuses_to_overwrite(tmp_path):
    records = _three_model_sweep(question_count=60)
    directory = tmp_path / "phase2-figures"
    write_phase2_figures(records, directory, n_resamples=50)
    with pytest.raises(FileExistsError):
        write_phase2_figures(records, directory, n_resamples=50)
