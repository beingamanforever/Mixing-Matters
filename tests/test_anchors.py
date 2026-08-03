from mixing_matters.anchors import build_prompt


def test_anchor_prompts(row):
    closed, closed_position = build_prompt(row, "closed_book")
    oracle, oracle_position = build_prompt(row, "oracle")
    middle, middle_position = build_prompt(row, "gold_middle")
    assert closed == "Question: Who?\nAnswer:"
    assert "Document [1](Title: d0) text 0" in oracle
    assert "Document [5](Title: d0) text 0" in middle
    assert (closed_position, oracle_position, middle_position) == (None, None, 4)
