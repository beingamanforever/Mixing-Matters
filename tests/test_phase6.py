import pytest

from mixing_matters.figures import write_phase6_figures
from mixing_matters.phase6 import phase6_summary, task_comparison

FLOOR = 0.0
CEILING = 1.0


def _gold(model_key, length, qid, position, score):
    return {
        "model_key": model_key,
        "context_length": length,
        "question_id": qid,
        "condition": "gold",
        "gold_position": position,
        "score": score,
        "floor_accuracy": FLOOR,
        "ceiling_accuracy": CEILING,
    }


def _records(model_key, length, question_count, edge_score, center_score):
    records = []
    for question in range(question_count):
        # Every model sees the same needle instances, so question ids are shared
        # across models exactly as the runner produces them.
        qid = f"{length}-q{question}"
        for position in range(10):
            score = edge_score if position in (0, 1) else center_score
            records.append(_gold(model_key, length, qid, position, score))
    return records


def _dataset():
    records = []
    for length, primacy in ((1024, 1.0), (2048, 0.5)):
        records += _records("mamba-2.8b", length, 20, edge_score=primacy, center_score=0.2)
        records += _records("pythia-2.8b", length, 20, edge_score=0.2, center_score=0.2)
    return records


def test_phase6_summary_groups_by_length():
    summary = phase6_summary(_dataset(), n_resamples=200)
    assert set(summary["lengths"]) == {1024, 2048}
    for length in (1024, 2048):
        entry = summary["lengths"][length]
        assert entry["models"] == ["mamba-2.8b", "pythia-2.8b"]
        assert len(entry["interactions"]) == 1
        mamba_primacy = entry["edges"]["mamba-2.8b"]["primacy"]["estimate"]
        assert mamba_primacy > 0
    # A stronger planted primacy at 1024 must show as a larger mamba edge there.
    assert (
        summary["lengths"][1024]["edges"]["mamba-2.8b"]["primacy"]["estimate"]
        > summary["lengths"][2048]["edges"]["mamba-2.8b"]["primacy"]["estimate"]
    )


def test_phase6_summary_rejects_missing_length():
    records = _records("mamba-2.8b", 1024, 5, 1.0, 0.2)
    for record in records:
        del record["context_length"]
    with pytest.raises(ValueError):
        phase6_summary(records, n_resamples=50)


def test_task_comparison_flags_disagreement():
    niah_edges = {"m": {"primacy": {"estimate": 0.3}}}
    qa_edges = {"m": {"primacy": {"estimate": -0.2}}}
    rows = task_comparison(niah_edges, qa_edges, "primacy")
    assert rows[0]["agree"] is False
    assert rows[0]["difference"] == pytest.approx(0.5)


def test_write_phase6_figures(tmp_path):
    paths = write_phase6_figures(_dataset(), tmp_path, n_resamples=200)
    names = {path.name for path in paths}
    assert "position-curve-1024.png" in names
    assert "position-edges-2048.png" in names
    assert "phase6-summary.json" in names
    for path in paths:
        assert path.exists()


def test_write_phase6_figures_with_qa_comparison(tmp_path):
    qa = _records("mamba-2.8b", 4096, 20, edge_score=0.2, center_score=0.2)
    qa += _records("pythia-2.8b", 4096, 20, edge_score=0.2, center_score=0.2)
    # Strip context_length: QA records are plain Phase 2 gold records.
    for record in qa:
        del record["context_length"]
    paths = write_phase6_figures(_dataset(), tmp_path, qa_records=qa, n_resamples=200)
    names = {path.name for path in paths}
    assert "task-comparison-1024.png" in names


def test_write_phase6_figures_refuses_overwrite(tmp_path):
    write_phase6_figures(_dataset(), tmp_path, n_resamples=100)
    with pytest.raises(FileExistsError):
        write_phase6_figures(_dataset(), tmp_path, n_resamples=100)
