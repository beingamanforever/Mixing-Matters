import json
import random

import pytest

from mixing_matters.figures import (
    PHASE1_CONDITIONS,
    kv_position_curve,
    phase1_condition_accuracy,
    write_figures,
)


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
