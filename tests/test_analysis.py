import pytest

from mixing_matters.analysis import planning_sample_size, summarize, validate_phase1


def test_discordance_and_planning_table():
    records = []
    for qid, first, middle in (("a", 1, 0), ("b", 0, 1), ("c", 1, 1), ("d", 0, 0)):
        records += [
            {"question_id": qid, "condition": "gold_first", "score": first},
            {"question_id": qid, "condition": "gold_middle", "score": middle},
        ]
    result = summarize(records)
    assert result["discordance"] == 0.5
    assert result["difference_counts"] == {"-1": 1, "0": 2, "1": 1}
    assert planning_sample_size(0.20) == 1156
    assert planning_sample_size(0.25) == 1460
    assert planning_sample_size(0.35) == 2068


def test_analysis_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="complete pairs"):
        summarize([{"question_id": "a", "condition": "gold_first", "score": 1}])


def test_phase1_validation_rejects_partial_results():
    with pytest.raises(ValueError, match="incomplete"):
        validate_phase1([])


def test_phase1_validation_rejects_mixed_provenance():
    expected = {"gold_first": 200, "gold_middle": 200, "closed_book": 50, "oracle": 50}
    records = []
    for condition, count in expected.items():
        for index in range(count):
            source = index
            records.append(
                {
                    "question_id": str(source),
                    "source_index": source,
                    "condition": condition,
                    "model": "m",
                    "model_revision": "a" if index else "b",
                    "data_revision": "d",
                    "data_sha256": "s",
                    "positive_control_sha256": "p",
                    "seed": 1,
                    "torch": "t",
                    "transformers": "x",
                    "cuda": "c",
                    "gpu": "g",
                    "attention_implementation": "eager",
                }
            )
    with pytest.raises(ValueError, match="model_revision"):
        validate_phase1(records)
