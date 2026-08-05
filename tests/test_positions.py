import pytest
from mixing_matters.build_positions import dense_positions, place_gold, place_fake, negative_positions, shuffle_distractors


def test_gold_moves_without_changing_distractors(row):
    original = [doc["title"] for doc in row["ctxs"] if not doc["isgold"]]
    for position, moved in enumerate(dense_positions(row)):
        assert moved["ctxs"][position]["isgold"] is True
        assert [doc["title"] for doc in moved["ctxs"] if not doc["isgold"]] == original
    assert place_gold(row, 9)["ctxs"][-1]["title"] == "d0"


def test_place_fake(row):
    distractors_orig = [doc for doc in row["ctxs"] if doc["isgold"] is not True]
    
    for pos, fake_row in enumerate(negative_positions(row)):
        ctxs = fake_row["ctxs"]
        assert len(ctxs) == 10
        assert ctxs[pos]["isgold"] is True
        assert ctxs[pos]["title"] == distractors_orig[0]["title"]
        assert ctxs[pos]["text"] == distractors_orig[0]["text"]
        
        # the remaining 9 must be the original distractors
        remaining = [doc for doc in ctxs if not doc["isgold"]]
        assert len(remaining) == 9
        assert remaining == distractors_orig
        
    with pytest.raises(ValueError):
        place_fake(row, 10)


def test_shuffle_distractors(row):
    # test deterministic output for given seed
    shuffled_1 = shuffle_distractors(row, 4, "seed_a")
    shuffled_2 = shuffle_distractors(row, 4, "seed_a")
    shuffled_3 = shuffle_distractors(row, 4, "seed_b")
    
    assert len(shuffled_1["ctxs"]) == 10
    assert shuffled_1["ctxs"][4]["isgold"] is True
    
    # deterministic for same seed
    assert shuffled_1 == shuffled_2
    
    # different for different seed (with high probability, as there are 9! permutations)
    assert shuffled_1 != shuffled_3
    
    # same set of distractors
    distractors_orig = [doc for doc in row["ctxs"] if doc["isgold"] is not True]
    dist_1 = [doc for doc in shuffled_1["ctxs"] if doc["isgold"] is not True]
    
    assert len(dist_1) == 9
    # elements are the same, order is different
    assert sorted(distractors_orig, key=lambda x: x["title"]) == sorted(dist_1, key=lambda x: x["title"])
