from mixing_matters.run import plan, plan_negative, plan_order


def test_phase1_plan_reuses_tracer_gold_first(row):
    rows = [row for _ in range(2655)]
    work = plan(rows)
    counts = {
        condition: sum(item[2] == condition for item in work)
        for condition in {item[2] for item in work}
    }
    assert len(work) == 500
    assert counts == {"gold_first": 200, "gold_middle": 200, "closed_book": 50, "oracle": 50}


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
