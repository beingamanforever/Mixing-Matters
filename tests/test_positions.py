from mixing_matters.build_positions import dense_positions, place_gold


def test_gold_moves_without_changing_distractors(row):
    original = [doc["title"] for doc in row["ctxs"] if not doc["isgold"]]
    for position, moved in enumerate(dense_positions(row)):
        assert moved["ctxs"][position]["isgold"] is True
        assert [doc["title"] for doc in moved["ctxs"] if not doc["isgold"]] == original
    assert place_gold(row, 9)["ctxs"][-1]["title"] == "d0"
