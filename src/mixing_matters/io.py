import gzip
import json
from collections.abc import Iterable
from pathlib import Path


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from a plain or gzipped file.

    Callers keep passing paths under ``artifacts/`` and ``runs/`` where the
    on-disk layout has switched to ``sweep.jsonl.gz`` for space, so this
    detects gzip by filename suffix and reads binary otherwise.
    """
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as stream:
            return [json.loads(line) for line in stream]
    with path.open() as stream:
        return [json.loads(line) for line in stream]
