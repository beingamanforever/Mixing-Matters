from mixing_matters.data import split_indices


def test_split_is_deterministic_disjoint_and_exhaustive():
    exploratory, heldout = split_indices()
    assert (exploratory, heldout) == split_indices()
    assert len(exploratory) == 800
    assert len(heldout) == 1855
    assert set(exploratory).isdisjoint(heldout)
    assert set(exploratory + heldout) == set(range(2655))
