import pytest

from mixing_matters.data import split_indices
from mixing_matters.run import (
    SEED,
    _require_mamba2_kernels,
    _require_mamba_kernels,
    plan,
    plan_negative,
    plan_order,
    plan_sweep,
)


def test_phase1_plan_stays_within_exploratory_split(row):
    rows = [row for _ in range(2655)]
    work = plan(rows)
    source_indices = {item[0] for item in work}
    _, confirmatory = split_indices(len(rows), SEED)
    assert source_indices.isdisjoint(confirmatory)


def test_phase1_plan_covers_all_conditions_for_every_question(row):
    rows = [row for _ in range(2655)]
    work = plan(rows)
    counts = {
        condition: sum(item[2] == condition for item in work)
        for condition in {item[2] for item in work}
    }
    assert len(work) == 800
    assert counts == {"gold_first": 200, "gold_middle": 200, "closed_book": 200, "oracle": 200}


def test_phase1_plan_orders_anchors_before_gold_conditions(row):
    rows = [row for _ in range(2655)]
    work = plan(rows)
    for group_start in range(0, len(work), 4):
        group = work[group_start : group_start + 4]
        indices = {item[0] for item in group}
        assert len(indices) == 1, "each group of four must cover a single question"
        assert [item[2] for item in group] == [
            "closed_book",
            "oracle",
            "gold_first",
            "gold_middle",
        ]


def test_plan_negative(row):
    rows = [row for _ in range(2655)]
    work = plan_negative(rows, n=200)
    assert len(work) == 2000  # 200 rows * 10 positions
    for item in work:
        assert item[2] == "negative_control"


def test_plan_order(row):
    rows = [row for _ in range(2655)]
    work = plan_order(rows, n=200, positions=(0, 4, 9), perms=3)
    assert len(work) == 1800  # 200 rows * 3 positions * 3 perms
    for item in work:
        assert item[2] == "distractor_order"


def test_plan_sweep_size_and_grouping(row):
    rows = [row for _ in range(900)]
    work = plan_sweep(rows, questions=50)
    assert len(work) == 600  # 12 * 50

    groups = [work[start : start + 12] for start in range(0, len(work), 12)]
    for group in groups:
        indices = {item[0] for item in group}
        assert len(indices) == 1, "each group of twelve must cover a single question"
        conditions = [item[2] for item in group]
        assert conditions == ["closed_book", "oracle"] + ["gold"] * 10
        positions = [item[3] for item in group]
        assert positions[:2] == [None, None]
        assert positions[2:] == list(range(10))


def test_plan_sweep_stays_within_exploratory_split(row):
    rows = [row for _ in range(2655)]
    work = plan_sweep(rows, questions=800)
    source_indices = {item[0] for item in work}
    _, confirmatory = split_indices(len(rows), SEED)
    assert source_indices.isdisjoint(confirmatory)


def test_plan_sweep_raises_when_item_count_does_not_match_questions(row):
    rows = [row for _ in range(3)]
    with pytest.raises(AssertionError, match="12 items per question"):
        plan_sweep(rows, questions=5)


def test_require_mamba_kernels_passes_when_all_functions_resolved():
    class FastMamba:
        selective_state_update = object()
        selective_scan_fn = object()
        mamba_inner_fn = object()
        causal_conv1d_fn = object()
        causal_conv1d_update = object()

    _require_mamba_kernels(FastMamba)


def test_require_mamba_kernels_fails_loudly_when_a_function_is_missing():
    class SlowMamba:
        selective_state_update = None
        selective_scan_fn = object()
        mamba_inner_fn = object()
        causal_conv1d_fn = None
        causal_conv1d_update = object()

    with pytest.raises(RuntimeError, match="selective_state_update"):
        _require_mamba_kernels(SlowMamba)


def test_require_mamba2_kernels_passes_when_fast_path_available():
    class FastMamba2:
        is_fast_path_available = True

    _require_mamba2_kernels(FastMamba2)


def test_require_mamba2_kernels_fails_loudly_when_fast_path_unavailable():
    class SlowMamba2:
        is_fast_path_available = False

    with pytest.raises(RuntimeError, match="is_fast_path_available"):
        _require_mamba2_kernels(SlowMamba2)
