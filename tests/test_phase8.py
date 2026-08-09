from pathlib import Path

import pytest

from mixing_matters.figures import write_phase8_figures
from mixing_matters.models import MODELS, PHASE8_SYSTEMS
from mixing_matters.phase8 import phase8_summary

FLOOR = 0.05
CEILING = 0.95


def test_phase8_systems_registered():
    """Every Phase 8 system is declared in MODELS and labeled phase8_system."""
    for key in PHASE8_SYSTEMS:
        assert key in MODELS, key
        assert MODELS[key].phase8_system == "phase8-systems", key


def test_phase8_system_families():
    """Phase 8 covers a hybrid Mamba plus two dense-attention transformers."""
    families = {key: MODELS[key].family for key in PHASE8_SYSTEMS}
    assert families == {
        "nemotron-h-8b": "nemotron-h",
        "llama-3.1-8b": "llama",
        "qwen2.5-7b": "qwen2",
    }


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


def _uniform_bundle(model_key, question_count, edge_score, center_score):
    positions = {position: (edge_score if position in (0, 1, 8, 9) else center_score) for position in range(10)}
    records = []
    for question in range(question_count):
        for position, score in positions.items():
            records.append(_gold_record(model_key, f"q{question}", position, score))
    return records


def test_phase8_summary_shape_over_three_systems():
    """Position curves, edges, and pairwise interactions for every Phase 8 system."""
    records = []
    for model_key in PHASE8_SYSTEMS:
        records += _uniform_bundle(model_key, question_count=50, edge_score=1.0, center_score=0.3)

    summary = phase8_summary(records, n_resamples=200)

    assert summary["models"] == list(PHASE8_SYSTEMS)
    assert set(summary["position_curve"]) == set(PHASE8_SYSTEMS)
    assert set(summary["edges"]) == set(PHASE8_SYSTEMS)
    # Three systems produce three pairwise interactions.
    assert len(summary["interactions"]) == 3
    for interaction in summary["interactions"]:
        assert interaction["first_model"] in PHASE8_SYSTEMS
        assert interaction["second_model"] in PHASE8_SYSTEMS
        assert interaction["first_model"] != interaction["second_model"]

    for model_key in PHASE8_SYSTEMS:
        floor_ceiling = summary["floor_ceiling"][model_key]
        assert floor_ceiling["floor_accuracy"] == pytest.approx(FLOOR)
        assert floor_ceiling["ceiling_accuracy"] == pytest.approx(CEILING)
        descriptor = summary["system_descriptors"][model_key]
        assert descriptor["family"] == MODELS[model_key].family
        assert descriptor["repo"] == MODELS[model_key].repo


def test_phase8_summary_ignores_non_gold_records():
    """closed_book and oracle records carry anchors and are not scored curves."""
    records = _uniform_bundle("nemotron-h-8b", question_count=30, edge_score=1.0, center_score=0.5)
    # Add anchor records that must not influence the curve.
    records.append({
        "model_key": "nemotron-h-8b",
        "question_id": "q0",
        "condition": "closed_book",
        "gold_position": None,
        "score": 0.0,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
    })
    records.append({
        "model_key": "nemotron-h-8b",
        "question_id": "q0",
        "condition": "oracle",
        "gold_position": None,
        "score": 1.0,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
    })
    summary = phase8_summary(records, n_resamples=100)
    positions = summary["position_curve"]["nemotron-h-8b"]["positions"]
    # Edge positions (0,1,8,9) planted at 1.0, center at 0.5.
    assert positions[0]["accuracy"] == pytest.approx(1.0)
    assert positions[4]["accuracy"] == pytest.approx(0.5)


def test_write_phase8_figures_emits_three_artifacts(tmp_path: Path):
    records = []
    for model_key in PHASE8_SYSTEMS:
        records += _uniform_bundle(model_key, question_count=40, edge_score=1.0, center_score=0.2)

    output = tmp_path / "phase8-report"
    paths = write_phase8_figures(records, output, n_resamples=200)

    assert {path.name for path in paths} == {
        "position-curves.png",
        "position-edges.png",
        "phase8-summary.json",
    }
    for path in paths:
        assert path.exists()


def test_write_phase8_figures_refuses_overwrite(tmp_path: Path):
    records = []
    for model_key in PHASE8_SYSTEMS:
        records += _uniform_bundle(model_key, question_count=20, edge_score=1.0, center_score=0.4)

    output = tmp_path / "phase8-report"
    write_phase8_figures(records, output, n_resamples=100)
    with pytest.raises(FileExistsError):
        write_phase8_figures(records, output, n_resamples=100)
