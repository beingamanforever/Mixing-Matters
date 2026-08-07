import pytest

from mixing_matters import ruler


def _words(text: str) -> int:
    """A stand-in tokenizer: whitespace words. Length invariance is tokenizer-agnostic."""
    return len(text.split())


def test_make_instance_is_deterministic():
    first = ruler.make_instance(3)
    again = ruler.make_instance(3)
    assert first == again
    assert first["value"] != ruler.make_instance(4)["value"]
    assert len(first["value"]) == ruler.NUMBER_DIGITS
    assert first["value"].isdigit()
    assert "-" in first["key"]


def test_single_needle_template_is_singular():
    template = ruler._single_needle_template()
    assert template.startswith("A special magic {type_needle_v} is hidden")
    assert "What is the special magic" in template
    assert "provided text is" in template
    assert "are" not in template


def test_gold_prompt_places_needle_and_holds_length():
    instance = ruler.make_instance(0)
    needle = ruler.NEEDLE.format(key=instance["key"], value=instance["value"])
    lengths = set()
    for depth in ruler.DEPTHS:
        prompt = ruler.build_gold_prompt(instance, depth, num_haystack=30)
        assert needle in prompt
        assert prompt.count(instance["value"]) == 1
        lengths.add(_words(prompt))
    # The needle is one constant sentence inserted among a fixed noise count, so
    # every depth yields exactly the same whitespace-word length.
    assert len(lengths) == 1


def test_depth_moves_the_needle_front_to_back():
    instance = ruler.make_instance(1)
    needle = ruler.NEEDLE.format(key=instance["key"], value=instance["value"])
    first = ruler.build_gold_prompt(instance, 0, num_haystack=30)
    last = ruler.build_gold_prompt(instance, 9, num_haystack=30)
    body_first = first.split("\n", 1)[1]
    body_last = last.split("\n", 1)[1]
    assert body_first.index(needle) < body_last.index(needle)


def test_floor_prompt_has_no_needle():
    instance = ruler.make_instance(2)
    prompt = ruler.build_floor_prompt(instance, num_haystack=20)
    assert instance["value"] not in prompt
    assert instance["key"] in prompt


def test_ceiling_prompt_is_needle_only():
    instance = ruler.make_instance(2)
    prompt = ruler.build_ceiling_prompt(instance)
    assert instance["value"] in prompt
    assert ruler.NOISE_SENTENCE not in prompt


def test_solve_haystack_size_respects_budget():
    reference = ruler.make_instance(0)
    target, gen = 400, 32
    size = ruler.solve_haystack_size(_words, target, gen, reference)
    assert size >= ruler.MIN_HAYSTACK
    at_size = _words(ruler.build_gold_prompt(reference, 5, size)) + gen
    over = _words(ruler.build_gold_prompt(reference, 5, size + 1)) + gen
    assert at_size <= target < over


def test_solve_haystack_size_rejects_tiny_budget():
    reference = ruler.make_instance(0)
    with pytest.raises(ValueError):
        ruler.solve_haystack_size(_words, 20, 32, reference)


def test_score_variants_substring_match():
    scores = ruler.score_variants("the answer is 1234567 indeed", ["1234567"])
    assert scores["score"] == 1.0
    assert ruler.score_variants("nope 9999999", ["1234567"])["score"] == 0.0


def test_score_first_line_and_normalized():
    scores = ruler.score_variants("1234567\nextra 1234567", ["1234567"])
    assert scores["score_first_line"] == 1.0
    assert scores["score_normalized_em"] == 1.0
    trailing = ruler.score_variants("prefix\n1234567", ["1234567"])
    assert trailing["score_first_line"] == 0.0
    assert trailing["score"] == 1.0
