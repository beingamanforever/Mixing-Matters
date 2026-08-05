import random
from copy import deepcopy

from .data import validate_row


def place_gold(row: dict, position: int) -> dict:
    validate_row(row)
    if position not in range(10):
        raise ValueError("gold position must be between 0 and 9")
    result = deepcopy(row)
    gold = next(doc for doc in result["ctxs"] if doc["isgold"] is True)
    distractors = [doc for doc in result["ctxs"] if doc["isgold"] is not True]
    distractors.insert(position, gold)
    result["ctxs"] = distractors
    return result


def dense_positions(row: dict) -> list[dict]:
    return [place_gold(row, position) for position in range(10)]


def place_fake(row: dict, position: int) -> dict:
    validate_row(row)
    if position not in range(10):
        raise ValueError("gold position must be between 0 and 9")
    result = deepcopy(row)
    distractors = [doc for doc in result["ctxs"] if doc["isgold"] is not True]
    fake_gold = deepcopy(distractors[0])
    fake_gold["isgold"] = True
    distractors.insert(position, fake_gold)
    result["ctxs"] = distractors
    return result


def negative_positions(row: dict) -> list[dict]:
    return [place_fake(row, position) for position in range(10)]


def shuffle_distractors(row: dict, gold_position: int, perm_seed: str | None) -> dict:
    """Permute the nine distractors while holding the gold document in place.

    A None seed keeps the dataset order, so one permutation reproduces the exact
    ordering used by the main position sweep.
    """
    result = place_gold(row, gold_position)
    if perm_seed is None:
        return result
    ctxs = result["ctxs"]
    gold = ctxs[gold_position]
    distractors = ctxs[:gold_position] + ctxs[gold_position + 1 :]

    rng = random.Random(perm_seed)
    rng.shuffle(distractors)

    distractors.insert(gold_position, gold)
    result["ctxs"] = distractors
    return result
