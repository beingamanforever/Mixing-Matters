from mixing_matters.data import split_indices
from mixing_matters.run import SEED, plan, plan_negative, plan_order


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
