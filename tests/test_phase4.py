import random

import pytest

from mixing_matters.models import SCALE_PAIRS
from mixing_matters.phase4 import scale_trend, trend_summary

FLOOR = 0.1
CEILING = 0.9

# (pair label, pythia model key, mamba model key), in the registry's size order.
PAIR_KEYS = [
    ("130m-160m", "pythia-160m", "mamba-130m"),
    ("370m-410m", "pythia-410m", "mamba-370m"),
    ("790m-1b", "pythia-1b", "mamba-790m"),
    ("1.4b-1.4b", "pythia-1.4b", "mamba-1.4b"),
    ("2.8b-2.8b", "pythia-2.8b", "mamba-2.8b"),
]


def _uniform_model_records(
    model_key: str,
    question_count: int,
    primacy_score: float,
    center_score: float,
    recency_score: float,
) -> list[dict]:
    """All questions get the same score per position band: no noise, a clean gap."""
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
    records = []
    for question in range(question_count):
        question_id = f"q{question}"
        for position, score in positions.items():
            records.append(
                {
                    "model_key": model_key,
                    "question_id": question_id,
                    "condition": "gold",
                    "gold_position": position,
                    "score": score,
                    "floor_accuracy": FLOOR,
                    "ceiling_accuracy": CEILING,
                }
            )
    return records


def _widening_primacy_gap_records(question_count: int = 20, pairs=PAIR_KEYS) -> list[dict]:
    """Mamba stays flat while Pythia's primacy score steps up at every larger pair."""
    records = []
    for index, (_, pythia_key, mamba_key) in enumerate(pairs):
        pythia_primacy = 0.5 + 0.05 * (index + 1)
        records += _uniform_model_records(pythia_key, question_count, pythia_primacy, 0.3, 0.3)
        records += _uniform_model_records(mamba_key, question_count, 0.5, 0.3, 0.3)
    return records


def _constant_gap_records(question_count: int = 20, pairs=PAIR_KEYS) -> list[dict]:
    """Every pair gets the same fixed primacy and recency gap."""
    records = []
    for _, pythia_key, mamba_key in pairs:
        records += _uniform_model_records(pythia_key, question_count, 0.6, 0.3, 0.3)
        records += _uniform_model_records(mamba_key, question_count, 0.5, 0.3, 0.3)
    return records


def test_scale_trend_orders_pairs_by_increasing_size():
    records = _widening_primacy_gap_records()
    result = scale_trend(records, n_resamples=50)

    assert [pair["pair"] for pair in result["pairs"]] == list(SCALE_PAIRS)
    assert result["missing_pairs"] == []


def test_scale_trend_recovers_widening_primacy_gap():
    records = _widening_primacy_gap_records()
    result = scale_trend(records, n_resamples=50)

    estimates = [pair["primacy_diff"]["estimate"] for pair in result["pairs"]]
    assert estimates == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.25])
    assert estimates == sorted(estimates)

    for pair in result["pairs"]:
        assert pair["recency_diff"]["estimate"] == pytest.approx(0.0)


def test_trend_summary_labels_widening_primacy_gap_as_grows():
    records = _widening_primacy_gap_records()
    result = scale_trend(records, n_resamples=50)
    summary = trend_summary(result)

    assert summary["primacy"]["smallest_pair"] == "130m-160m"
    assert summary["primacy"]["largest_pair"] == "2.8b-2.8b"
    assert summary["primacy"]["change"] == pytest.approx(0.20)
    assert summary["primacy"]["direction"] == "grows"
    assert summary["recency"]["direction"] == "stable"


def test_trend_summary_labels_constant_gap_as_stable():
    records = _constant_gap_records()
    result = scale_trend(records, n_resamples=50)
    summary = trend_summary(result)

    assert summary["primacy"]["change"] == pytest.approx(0.0)
    assert summary["primacy"]["direction"] == "stable"
    assert summary["recency"]["direction"] == "stable"


def test_scale_trend_is_deterministic_across_calls_and_shuffled_order():
    records = _widening_primacy_gap_records()

    first = scale_trend(records, n_resamples=50)
    second = scale_trend(records, n_resamples=50)
    assert first == second

    shuffled = list(records)
    random.Random(4).shuffle(shuffled)
    third = scale_trend(shuffled, n_resamples=50)
    assert first == third


def test_scale_trend_reports_missing_pairs_instead_of_skipping_silently():
    pairs_without_middle = [pair for pair in PAIR_KEYS if pair[0] != "790m-1b"]
    records = _widening_primacy_gap_records(pairs=pairs_without_middle)

    result = scale_trend(records, n_resamples=50)

    assert result["missing_pairs"] == ["790m-1b"]
    assert [pair["pair"] for pair in result["pairs"]] == [
        "130m-160m",
        "370m-410m",
        "1.4b-1.4b",
        "2.8b-2.8b",
    ]


def test_scale_trend_rejects_no_pairs_present():
    with pytest.raises(ValueError, match="no scale pairs"):
        scale_trend([], n_resamples=10)


def test_trend_summary_rejects_fewer_than_two_pairs():
    pairs_with_one = [PAIR_KEYS[0]]
    records = _widening_primacy_gap_records(pairs=pairs_with_one)
    result = scale_trend(records, n_resamples=50)

    with pytest.raises(ValueError, match="at least two"):
        trend_summary(result)
