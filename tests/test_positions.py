import pytest

from mixing_matters.build_positions import (
    dense_positions,
    negative_positions,
    place_fake,
    place_gold,
    shuffle_distractors,
)


def test_gold_moves_without_changing_distractors(row):
    original = [doc["title"] for doc in row["ctxs"] if not doc["isgold"]]
    for position, moved in enumerate(dense_positions(row)):
        assert moved["ctxs"][position]["isgold"] is True
        assert [doc["title"] for doc in moved["ctxs"] if not doc["isgold"]] == original
    assert place_gold(row, 9)["ctxs"][-1]["title"] == "d0"


def test_place_fake(row):
    originals = [doc for doc in row["ctxs"] if doc["isgold"] is not True]

    for pos, faked in enumerate(negative_positions(row)):
        ctxs = faked["ctxs"]
        assert len(ctxs) == 10
        assert ctxs[pos]["isgold"] is True
        assert ctxs[pos]["title"] == originals[0]["title"]
        assert ctxs[pos]["text"] == originals[0]["text"]

        # the remaining 9 must be the original distractors
        remaining = [doc for doc in ctxs if not doc["isgold"]]
        assert len(remaining) == 9
        assert remaining == originals

    with pytest.raises(ValueError):
        place_fake(row, 10)


def test_shuffle_distractors(row):
    # test deterministic output for given seed
    first = shuffle_distractors(row, 4, "seed_a")
    repeat = shuffle_distractors(row, 4, "seed_a")
    other = shuffle_distractors(row, 4, "seed_b")

    assert len(first["ctxs"]) == 10
    assert first["ctxs"][4]["isgold"] is True

    # deterministic for same seed
    assert first == repeat

    # different for different seed (with high probability, as there are 9! permutations)
    assert first != other

    # same set of distractors
    originals = [doc for doc in row["ctxs"] if doc["isgold"] is not True]
    shuffled = [doc for doc in first["ctxs"] if doc["isgold"] is not True]

    assert len(shuffled) == 9
    # elements are the same, order is different
    assert sorted(originals, key=lambda x: x["title"]) == sorted(shuffled, key=lambda x: x["title"])


def test_unseeded_permutation_keeps_order(row):
    for position in range(10):
        assert shuffle_distractors(row, position, None) == place_gold(row, position)
