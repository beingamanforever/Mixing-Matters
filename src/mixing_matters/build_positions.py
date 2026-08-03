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
