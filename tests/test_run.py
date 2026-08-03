from mixing_matters.run import plan


def test_phase1_plan_reuses_tracer_gold_first(row):
    rows = [row for _ in range(2655)]
    work = plan(rows)
    counts = {
        condition: sum(item[2] == condition for item in work)
        for condition in {item[2] for item in work}
    }
    assert len(work) == 500
    assert counts == {"gold_first": 200, "gold_middle": 200, "closed_book": 50, "oracle": 50}
