import pytest

np = pytest.importorskip("numpy")

from mixing_matters.probe import probe_gold_position


def _record(qid, position, vector, layer=12):
    return {
        "question_id": qid,
        "gold_position": position,
        "layer": layer,
        "hidden_state": list(vector),
    }


def test_probe_recovers_linearly_separable_position():
    """A feature that encodes edge-vs-middle should be decodable well above chance."""
    rng = np.random.default_rng(0)
    records = []
    for q in range(120):
        for position in range(10):
            # First feature carries the label signal; the rest are noise.
            if position in (0, 1, 8, 9):
                signal = 2.0
            elif position in (4, 5):
                signal = -2.0
            else:
                continue
            vector = [signal + rng.normal(0, 0.3)] + list(rng.normal(0, 1.0, size=7))
            records.append(_record(f"q{q}", position, vector))
    result = probe_gold_position(records, folds=5, epochs=300, lr=0.2)
    assert result["accuracy"] > 0.85
    assert result["shuffled_accuracy"] < 0.65
    assert result["n_features"] == 8
    assert result["layer"] == 12


def test_probe_shuffled_control_near_chance_when_no_signal():
    """Pure noise features give ~chance accuracy on both real and shuffled labels."""
    rng = np.random.default_rng(1)
    records = []
    for q in range(120):
        for position in (0, 1, 4, 5, 8, 9):
            vector = list(rng.normal(0, 1.0, size=8))
            records.append(_record(f"q{q}", position, vector))
    result = probe_gold_position(records, folds=5, epochs=200, lr=0.1)
    assert result["accuracy"] < 0.65
    assert result["shuffled_accuracy"] < 0.65


def test_probe_raises_without_edge_or_middle_records():
    records = [_record("q0", 3, [0.1, 0.2]), _record("q0", 6, [0.3, 0.4])]
    with pytest.raises(ValueError):
        probe_gold_position(records)
