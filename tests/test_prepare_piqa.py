"""Tests for deterministic PIQA validation-data preparation."""

import json
from pathlib import Path

import pytest

from scripts.prepare_piqa import EXPECTED_ROWS, prepare_piqa


def test_prepare_piqa_joins_all_rows_deterministically(tmp_path: Path) -> None:
    examples_path, labels_path = _write_sources(tmp_path)
    output_path = tmp_path / "piqa_valid.jsonl"

    prepare_piqa(examples_path, labels_path, output_path)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == EXPECTED_ROWS
    assert records[0] == {"goal": "goal 0", "sol1": "a", "sol2": "b", "label": 0}
    assert records[-1]["label"] == 1


def test_prepare_piqa_refuses_to_overwrite(tmp_path: Path) -> None:
    examples_path, labels_path = _write_sources(tmp_path)
    output_path = tmp_path / "piqa_valid.jsonl"
    output_path.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_piqa(examples_path, labels_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "preserve me\n"


def test_prepare_piqa_rejects_mismatched_rows_without_output(tmp_path: Path) -> None:
    examples_path, labels_path = _write_sources(tmp_path)
    labels_path.write_text("0\n", encoding="utf-8")
    output_path = tmp_path / "piqa_valid.jsonl"

    with pytest.raises(ValueError, match="row count mismatch"):
        prepare_piqa(examples_path, labels_path, output_path)

    assert not output_path.exists()


def test_prepare_piqa_requires_complete_validation_split(tmp_path: Path) -> None:
    examples_path, labels_path = _write_sources(tmp_path)
    examples_path.write_text(
        f"{json.dumps({'goal': 'goal', 'sol1': 'a', 'sol2': 'b'})}\n", encoding="utf-8"
    )
    labels_path.write_text("0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"expected {EXPECTED_ROWS} PIQA rows"):
        prepare_piqa(examples_path, labels_path, tmp_path / "piqa_valid.jsonl")


def test_prepare_piqa_rejects_non_binary_label_without_output(tmp_path: Path) -> None:
    examples_path, labels_path = _write_sources(tmp_path, last_label="2")
    output_path = tmp_path / "piqa_valid.jsonl"

    with pytest.raises(ValueError, match="invalid label"):
        prepare_piqa(examples_path, labels_path, output_path)

    assert not output_path.exists()


def _write_sources(tmp_path: Path, last_label: str = "1") -> tuple[Path, Path]:
    """Write a complete synthetic PIQA split and return its source paths."""
    examples_path = tmp_path / "valid.jsonl"
    labels_path = tmp_path / "valid-labels.lst"
    examples = (
        json.dumps({"goal": f"goal {index}", "sol1": "a", "sol2": "b"})
        for index in range(EXPECTED_ROWS)
    )
    labels = [str(index % 2) for index in range(EXPECTED_ROWS)]
    labels[-1] = last_label
    examples_text = "\n".join(examples)
    labels_text = "\n".join(labels)
    examples_path.write_text(f"{examples_text}\n", encoding="utf-8")
    labels_path.write_text(f"{labels_text}\n", encoding="utf-8")
    return examples_path, labels_path
