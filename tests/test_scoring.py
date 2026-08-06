from mixing_matters.scoring import score_variants


def test_exact_substring_match():
    result = score_variants("The answer is Gold Bar.", ["Gold Bar"])
    assert result["score"] == 1.0
    assert result["score_first_line"] == 1.0
    assert result["score_normalized_em"] == 0.0

    exact = score_variants("Gold Bar", ["Gold Bar"])
    assert exact == {"score": 1.0, "score_normalized_em": 1.0, "score_first_line": 1.0}


def test_normalization_handles_case_articles_and_punctuation():
    result = score_variants("the Gold, Bar!", ["Gold Bar"])
    assert result["score_normalized_em"] == 1.0

    result = score_variants("the Gold, Bar! is not it", ["Gold Bar"])
    assert result["score"] == 1.0
    assert result["score_normalized_em"] == 0.0


def test_first_line_correct_but_later_lines_ramble():
    generation = "Gold Bar\nActually I am not sure, let me reconsider and say Silver instead."
    result = score_variants(generation, ["Gold Bar"])
    assert result["score_first_line"] == 1.0
    assert result["score_normalized_em"] == 1.0
    assert result["score"] == 1.0


def test_normalized_exact_match_ignores_continuation_past_the_answer():
    generation = " Gold Bar\n\nQuestion: who wrote it\nAnswer: someone else"
    result = score_variants(generation, ["Gold Bar"])
    assert result["score_normalized_em"] == 1.0

    wrong_line = "Silver Bar\nGold Bar"
    assert score_variants(wrong_line, ["Gold Bar"])["score_normalized_em"] == 0.0


def test_empty_generation_scores_zero():
    result = score_variants("", ["Gold Bar"])
    assert result == {"score": 0.0, "score_normalized_em": 0.0, "score_first_line": 0.0}
