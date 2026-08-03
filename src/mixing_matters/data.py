import gzip
import hashlib
import json
import random
from pathlib import Path

from . import UPSTREAM_COMMIT

ROW_COUNT = 2655
EXPLORE_COUNT = 800


def read_rows(path: Path, expected_count: int = ROW_COUNT) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(rows)}")
    for row in rows:
        validate_row(row)
    return rows


def validate_row(row: dict) -> None:
    if not row.get("question") or not row.get("answers"):
        raise ValueError("each row needs a question and answers")
    documents = row.get("ctxs", [])
    if len(documents) != 10 or sum(doc.get("isgold") is True for doc in documents) != 1:
        raise ValueError("each row must contain one gold document and nine distractors")


def question_id(row: dict, source_index: int) -> str:
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    value = f"{UPSTREAM_COMMIT}:{source_index}:{canonical}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def split_indices(count: int = ROW_COUNT, seed: int = 240521) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return indices[:EXPLORE_COUNT], indices[EXPLORE_COUNT:]
