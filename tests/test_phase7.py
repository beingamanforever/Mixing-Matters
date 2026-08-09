from pathlib import Path

import pytest

from mixing_matters.figures import write_phase7_figures
from mixing_matters.phase7 import (
    MODEL_LAYERS,
    depth_trend,
    length_sensitivity,
    phase7_summary,
    scoring_sensitivity,
)

FLOOR = 0.05
CEILING = 0.95


def _gold(model_key, qid, position, score, prompt_tokens=1500, normalized=None, first_line=None):
    return {
        "model_key": model_key,
        "question_id": qid,
        "condition": "gold",
        "gold_position": position,
        "score": score,
        "score_normalized_em": normalized if normalized is not None else score,
        "score_first_line": first_line if first_line is not None else score,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
        "prompt_token_count": prompt_tokens,
    }


def _uniform_bundle(model_key, count, edge_score, center_score, prompt_tokens=1500):
    records = []
    for q in range(count):
        for position in range(10):
            score = edge_score if position in (0, 1, 8, 9) else center_score
            records.append(_gold(model_key, f"q{q}", position, score, prompt_tokens))
    return records


def test_depth_trend_joins_layer_counts():
    records = []
    for model_key in ("pythia-160m", "pythia-1b", "pythia-2.8b"):
        records += _uniform_bundle(model_key, 40, 1.0, 0.3)
    trend = depth_trend(records, n_resamples=200)
    layers = {entry["model_key"]: entry["layers"] for entry in trend["models"]}
    assert layers == {
        "pythia-160m": MODEL_LAYERS["pythia-160m"],
        "pythia-1b": MODEL_LAYERS["pythia-1b"],
        "pythia-2.8b": MODEL_LAYERS["pythia-2.8b"],
    }
    for entry in trend["models"]:
        assert entry["family"] == "pythia"
        assert entry["question_count"] == 40


def test_depth_trend_skips_unregistered_models():
    records = _uniform_bundle("pythia-2.8b", 30, 1.0, 0.3)
    records += _uniform_bundle("not-a-real-model", 30, 1.0, 0.3)
    trend = depth_trend(records, n_resamples=100)
    keys = {entry["model_key"] for entry in trend["models"]}
    assert keys == {"pythia-2.8b"}


def test_scoring_sensitivity_reports_all_three_variants():
    records = []
    for r in _uniform_bundle("pythia-2.8b", 40, 1.0, 0.3):
        r["score_normalized_em"] = r["score"]
        r["score_first_line"] = 1.0 - r["score"]
        records.append(r)
    result = scoring_sensitivity(records, n_resamples=200)
    assert set(result["variants"]) == {"score", "score_normalized_em", "score_first_line"}
    # First-line variant inverts the arm, so its primacy should flip sign.
    primary = result["variants"]["score"]["pythia-2.8b"]["primacy"]["estimate"]
    first_line = result["variants"]["score_first_line"]["pythia-2.8b"]["primacy"]["estimate"]
    assert primary > 0 > first_line


def test_length_sensitivity_bins_and_edges():
    records = []
    for q in range(90):
        tokens = 1200 if q < 30 else (1500 if q < 60 else 1800)
        edge = 0.9 if tokens == 1800 else (0.5 if tokens == 1500 else 0.2)
        for position in range(10):
            score = edge if position in (0, 1, 8, 9) else 0.3
            records.append(_gold("pythia-2.8b", f"q{q}", position, score, tokens))
    result = length_sensitivity(records, n_resamples=200, n_bins=3)
    assert result["n_bins"] == 3
    assert len(result["bins"]) == 3
    # Longer prompt bins should show a larger primacy edge (planted).
    estimates = [entry["edges"]["pythia-2.8b"]["primacy"]["estimate"] for entry in result["bins"]]
    assert estimates[-1] > estimates[0]


def test_phase7_summary_joins_all_three_lenses():
    records = _uniform_bundle("pythia-2.8b", 40, 1.0, 0.3)
    summary = phase7_summary(records, n_resamples=200)
    assert "position_curve" in summary
    assert "depth_trend" in summary
    assert "scoring_sensitivity" in summary
    assert "length_sensitivity" in summary


def test_write_phase7_figures_emits_expected_paths(tmp_path: Path):
    records = _uniform_bundle("pythia-2.8b", 40, 1.0, 0.3)
    records += _uniform_bundle("pythia-410m", 40, 0.4, 0.2)
    paths = write_phase7_figures(records, tmp_path / "report", n_resamples=200)
    names = {path.name for path in paths}
    assert "phase7-summary.json" in names
    for path in paths:
        assert path.exists()


def test_write_phase7_figures_refuses_overwrite(tmp_path: Path):
    records = _uniform_bundle("pythia-2.8b", 20, 1.0, 0.3)
    write_phase7_figures(records, tmp_path / "report", n_resamples=100)
    with pytest.raises(FileExistsError):
        write_phase7_figures(records, tmp_path / "report", n_resamples=100)
