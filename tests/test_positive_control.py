import json

import pytest

from mixing_matters.positive_control import (
    EXAMPLES,
    KEYS,
    POSITIONS,
    control_examples,
    validate_control,
)


def test_control_is_deterministic():
    assert control_examples() == control_examples()
    assert len(control_examples()) == EXAMPLES
    assert {len(example["pairs"]) for example in control_examples()} == {KEYS}
    assert set(POSITIONS.values()) <= set(range(KEYS))


def test_control_gate(tmp_path):
    metadata = {
        "model": "m",
        "model_revision": "r",
        "seed": 1,
        "python": "p",
        "torch": "t",
        "transformers": "x",
        "cuda": "c",
        "gpu": "g",
        "attention_implementation": "eager",
    }
    path = tmp_path / "control.jsonl"
    with path.open("x") as stream:
        for index in range(EXAMPLES):
            for condition in POSITIONS:
                score = float(condition != "kv_middle")
                record = {
                    "control_id": index,
                    "condition": condition,
                    "score": score,
                    "prompt_token_count": 100,
                    **metadata,
                }
                stream.write(json.dumps(record) + "\n")
    assert validate_control(path, metadata)

    records = path.read_text().replace('"score": 1.0', '"score": 0.0')
    failed = tmp_path / "failed.jsonl"
    failed.write_text(records)
    with pytest.raises(ValueError, match="failed"):
        validate_control(failed, metadata)
