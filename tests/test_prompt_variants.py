import pytest
from lost_in_the_middle.prompting import Document

from mixing_matters.prompt_variants import VARIANTS, build_variant_prompt


def _documents(n: int = 3) -> list[Document]:
    return [
        Document(title=f"Title {index}", text=f"Body {index}")
        for index in range(n)
    ]


def test_variants_enum_defines_expected_orders():
    assert VARIANTS == ("baseline", "question_first", "bookend", "gold_padded")


def test_baseline_matches_liu_default():
    prompt = build_variant_prompt("Q?", _documents(3), variant="baseline")
    # Documents come before the trailing "Question:" line in the baseline layout.
    assert prompt.rindex("Document [1]") < prompt.rindex("Question:")
    assert prompt.count("Question:") == 1


def test_question_first_places_question_before_documents():
    prompt = build_variant_prompt("What is X?", _documents(3), variant="question_first")
    # Question appears above the document block.
    assert prompt.index("What is X?") < prompt.index("Document [1]")
    # Only one question appearance.
    assert prompt.count("Question:") == 1


def test_bookend_places_question_before_and_after_documents():
    prompt = build_variant_prompt("How many?", _documents(3), variant="bookend")
    first = prompt.index("Question:")
    last = prompt.rindex("Question:")
    assert first < prompt.index("Document [1]") < last
    assert prompt.count("Question:") == 2


def test_gold_padded_inserts_pad_between_documents_and_question():
    baseline = build_variant_prompt("Q?", _documents(3), variant="baseline")
    padded = build_variant_prompt(
        "Q?", _documents(3), variant="gold_padded", gold_padded_tokens=4, pad_token="XPAD"
    )
    assert padded.count("XPAD") == 4
    assert padded.index("Document [1]") < padded.index("XPAD") < padded.index("Question:")
    # Padding does not disturb the earlier prompt content.
    assert padded.startswith(baseline.split("\n\nQuestion:", 1)[0])


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_variant_prompt("Q?", _documents(3), variant="not-a-variant")


def test_gold_padded_zero_matches_baseline():
    baseline = build_variant_prompt("Q?", _documents(3), variant="baseline")
    padded = build_variant_prompt("Q?", _documents(3), variant="gold_padded", gold_padded_tokens=0)
    assert padded == baseline


def test_gold_padded_rejects_negative_tokens():
    with pytest.raises(ValueError):
        build_variant_prompt("Q?", _documents(3), variant="gold_padded", gold_padded_tokens=-1)
