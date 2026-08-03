from lost_in_the_middle.metrics import best_subspan_em, normalize_answer


def test_official_scoring_behavior():
    assert normalize_answer("The Wilhelm Rontgen!") == "wilhelm rontgen"
    assert best_subspan_em("It was Wilhelm Rontgen.", ["Wilhelm Rontgen"]) == 1.0
