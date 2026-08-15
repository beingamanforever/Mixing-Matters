#!/usr/bin/env python
"""Prepare the pinned PIQA validation split for the Phase 3 benchmark gate."""

import argparse
import json
from pathlib import Path

EXPECTED_ROWS = 1_838
REQUIRED_FIELDS = frozenset({"goal", "sol1", "sol2"})
VALID_LABELS = frozenset({"0", "1"})


def main() -> None:
    """Parse command-line paths and prepare the PIQA validation file."""
    parser = _build_parser()
    args = parser.parse_args()
    prepare_piqa(args.examples, args.labels, args.output)


def prepare_piqa(examples_path: Path, labels_path: Path, output_path: Path) -> None:
    """Join PIQA examples and labels into a new validated JSONL file."""
    examples = _read_examples(examples_path)
    labels = _read_labels(labels_path)
    _validate_row_counts(examples, labels)

    with output_path.open("x", encoding="utf-8") as output:
        for example, label in zip(examples, labels, strict=True):
            record = {**example, "label": label}
            output.write(f"{json.dumps(record, ensure_ascii=False)}\n")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="PIQA valid.jsonl path")
    parser.add_argument("--labels", type=Path, required=True, help="PIQA valid-labels.lst path")
    parser.add_argument("--output", type=Path, required=True, help="New labeled JSONL path")
    return parser


def _read_examples(path: Path) -> list[dict[str, object]]:
    """Read and validate PIQA JSONL examples."""
    examples: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"empty example at line {line_number}")
            example = json.loads(line)
            if not isinstance(example, dict):
                raise TypeError(f"example at line {line_number} is not a JSON object")
            missing_fields = REQUIRED_FIELDS.difference(example)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"example at line {line_number} is missing: {missing}")
            if "label" in example:
                raise ValueError(f"example at line {line_number} already has a label")
            examples.append(example)
    return examples


def _read_labels(path: Path) -> list[int]:
    """Read PIQA labels and require every value to be zero or one."""
    labels: list[int] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            label = line.strip()
            if label not in VALID_LABELS:
                raise ValueError(f"invalid label at line {line_number}: {label!r}")
            labels.append(int(label))
    return labels


def _validate_row_counts(examples: list[dict[str, object]], labels: list[int]) -> None:
    """Require matching source counts and the complete PIQA validation split."""
    if len(examples) != len(labels):
        raise ValueError(
            f"PIQA row count mismatch: {len(examples)} examples and {len(labels)} labels"
        )
    if len(examples) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} PIQA rows, found {len(examples)}")


if __name__ == "__main__":
    main()
