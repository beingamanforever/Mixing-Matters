import pytest

from mixing_matters.phase7_variants import (
    VARIANTS,
    phase7_variant_summary,
    variant_curve,
    variant_edges,
)

FLOOR = 0.05
CEILING = 0.95


def _gold(model_key, qid, position, score, variant=None):
    record = {
        "model_key": model_key,
        "question_id": qid,
        "condition": "gold",
        "gold_position": position,
        "score": score,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
        "prompt_token_count": 1500,
    }
    if variant is not None:
        record["prompt_variant"] = variant
    return record


def _bundle(model_key, variant, count, edge_score, center_score):
    records = []
    for q in range(count):
        for position in range(10):
            score = edge_score if position in (0, 1, 8, 9) else center_score
            records.append(_gold(model_key, f"q{q}", position, score, variant=variant))
    return records


def test_variants_enum():
    assert VARIANTS == ("baseline", "question_first", "bookend", "gold_padded")


def test_variant_edges_groups_by_model_and_variant():
    records = []
    records += _bundle("pythia-2.8b", "baseline", 30, 0.9, 0.3)
    records += _bundle("pythia-2.8b", "question_first", 30, 0.6, 0.4)
    records += _bundle("mamba-2.8b", "question_first", 30, 0.85, 0.5)
    result = variant_edges(records, n_resamples=200)
    assert set(result) == {"pythia-2.8b", "mamba-2.8b"}
    assert set(result["pythia-2.8b"]) == {"baseline", "question_first"}
    assert result["pythia-2.8b"]["baseline"]["primacy"]["estimate"] > 0
    assert result["mamba-2.8b"]["question_first"]["primacy"]["estimate"] > 0


def test_variant_edges_defaults_missing_variant_to_baseline():
    """Legacy Phase 2 records without a prompt_variant field must still land in ``baseline``."""
    records = []
    for q in range(20):
        for position in range(10):
            score = 0.7 if position in (0, 1, 8, 9) else 0.3
            records.append(_gold("pythia-2.8b", f"q{q}", position, score, variant=None))
    result = variant_edges(records, n_resamples=200)
    assert "baseline" in result["pythia-2.8b"]


def test_variant_curve_returns_positions_per_variant():
    records = _bundle("pythia-2.8b", "question_first", 30, 0.8, 0.4)
    result = variant_curve(records, n_resamples=200)
    assert "question_first" in result["pythia-2.8b"]
    positions = result["pythia-2.8b"]["question_first"]["positions"]
    assert set(positions) == set(range(10))


def test_phase7_variant_summary_joins_curve_and_edges():
    records = _bundle("pythia-2.8b", "baseline", 30, 0.9, 0.3)
    records += _bundle("pythia-2.8b", "question_first", 30, 0.5, 0.5)
    summary = phase7_variant_summary(records, n_resamples=200)
    assert "position_curve" in summary
    assert "edges" in summary


def test_variant_edges_raises_on_missing_model_key():
    records = _bundle("pythia-2.8b", "baseline", 20, 0.9, 0.3)
    records[0].pop("model_key")
    with pytest.raises(ValueError):
        variant_edges(records, n_resamples=100)
