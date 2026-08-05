import pytest

from mixing_matters.analysis import planning_sample_size, summarize, validate_phase1, bootstrap_paired_edges, validate_negative, validate_order
import random


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


def test_bootstrap_paired_edges():
    rng = random.Random(42)
    # flat fixture
    flat = {str(i): {pos: 0.5 for pos in range(10)} for i in range(100)}
    ci_prim, ci_rec = bootstrap_paired_edges(flat, rng, 100)
    assert ci_prim[0] <= 0 <= ci_prim[1]
    assert ci_rec[0] <= 0 <= ci_rec[1]

    # primacy fixture
    prim = {str(i): {pos: 0.8 if pos in (0, 1) else 0.4 for pos in range(10)} for i in range(100)}
    ci_prim2, ci_rec2 = bootstrap_paired_edges(prim, rng, 100)
    assert ci_prim2[0] > 0


def test_validate_negative():
    records = []
    for i in range(100):
        for pos in range(10):
            records.append({
                "question_id": str(i),
                "gold_position": pos,
                "score": 0.5,
                "floor_accuracy": 0.5,
                "prompt_token_count": 100
            })
    validate_negative(records)
    
    records_diff = [dict(r, score=0.6) for r in records]
    with pytest.raises(ValueError, match="differs from floor"):
        validate_negative(records_diff)
        
    records_len = []
    for i in range(100):
        for pos in range(10):
            records_len.append({
                "question_id": str(i),
                "gold_position": pos,
                "score": 0.5,
                "floor_accuracy": 0.5,
                "prompt_token_count": 100 + (pos % 2)
            })
    with pytest.raises(ValueError, match="non-invariance"):
        validate_negative(records_len)
        
    records_prim = [dict(r, score=0.8 if r["gold_position"] in (0,1) else 0.4) for r in records]
    with pytest.raises(ValueError, match="flatness CI for primacy"):
        validate_negative(records_prim)


def test_validate_order():
    records = []
    for i in range(10):
        for pos in (0, 4, 9):
            for perm in range(3):
                records.append({
                    "question_id": str(i),
                    "gold_position": pos,
                    "score": 0.5 + 0.01 * perm,
                    "prompt_token_count": 100
                })
    validate_order(records)
    
    records_high = []
    for i in range(10):
        for pos in (0, 4, 9):
            for perm in range(3):
                records_high.append({
                    "question_id": str(i),
                    "gold_position": pos,
                    "score": 0.5 + 0.2 * perm,
                    "prompt_token_count": 100
                })
    with pytest.raises(ValueError, match="> 0.10"):
        validate_order(records_high)
