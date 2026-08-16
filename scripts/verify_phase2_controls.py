#!/usr/bin/env python3
"""Strictly verify the complete Phase 2 certification-control bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import string
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mixing_matters.analysis import validate_negative, validate_order
from mixing_matters.data import question_id, read_rows, split_indices
from mixing_matters.positive_control import validate_control

EXPECTED_DATA_REVISION = "29b8a6d042ce29abccee3db1a73171a107d7e6af"
EXPECTED_DATA_SHA256 = "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9"
EXPECTED_MODEL_NAME = "EleutherAI/pythia-2.8b"
EXPECTED_MODEL_REVISION = "2a259cdd96a4beb1cdf467512e3904197345f6a9"
EXPECTED_NEGATIVE_RECORDS = 2000
EXPECTED_ORDER_RECORDS = 1800
EXPECTED_QUESTIONS = 200
EXPECTED_POSITIONS = tuple(range(10))
EXPECTED_ORDER_POSITIONS = (0, 4, 9)
EXPECTED_PERMUTATIONS = (0, 1, 2)
EXPECTED_SETTINGS: dict[str, Any] = {
    "temperature": 0,
    "top_p": 1,
    "top_k": None,
    "max_new_tokens": 32,
    "random_seed": 240521,
    "manual_seed": 240521,
}
SOFTWARE_FIELDS = ("python", "torch", "transformers", "cuda", "attention_implementation")
POSITIVE_METADATA_FIELDS = (
    "model",
    "model_revision",
    "seed",
    "python",
    "torch",
    "transformers",
    "cuda",
    "gpu",
    "attention_implementation",
)


def verify_bundle(
    negative_path: Path,
    order_path: Path,
    positive_control_path: Path,
    data_path: Path,
    environment_path: Path,
    expected_repo_commit: str,
) -> dict[str, Any]:
    """Validate exact coverage, provenance, gates, and statistics for one bundle."""
    paths = (negative_path, order_path, positive_control_path)
    for path in paths:
        _require_no_failure_sidecar(path)

    negative_records = read_records(negative_path)
    order_records = read_records(order_path)
    positive_records = read_records(positive_control_path)
    positive_metadata, control_hashes = _verify_positive_control(
        positive_control_path, positive_records
    )
    expected_questions = _expected_questions(data_path)
    environment = _verify_environment(
        environment_path, data_path, expected_repo_commit, positive_metadata["python"]
    )

    _verify_negative(negative_records, expected_questions)
    _verify_order(order_records, expected_questions)

    negative_provenance = _verify_provenance(negative_records, control_hashes, "negative")
    order_provenance = _verify_provenance(order_records, control_hashes, "order")
    if (
        negative_provenance["positive_control_sha256"]
        != order_provenance["positive_control_sha256"]
    ):
        raise ValueError("negative and order controls use different positive-control hashes")
    if negative_provenance["software_versions"] != order_provenance["software_versions"]:
        raise ValueError("negative and order controls use different software provenance")
    expected_software = {field: positive_metadata[field] for field in SOFTWARE_FIELDS}
    if negative_provenance["software_versions"] != expected_software:
        raise ValueError("raw controls and positive control use different software provenance")
    if negative_provenance["run_id"] == order_provenance["run_id"]:
        raise ValueError("negative and order controls must have distinct run IDs")

    validate_negative(negative_records)
    validate_order(order_records)

    return {
        "schema_version": 1,
        "status": "passed",
        "artifacts": {
            "positive_control": _artifact_entry(positive_control_path, positive_records),
            "negative": _artifact_entry(negative_path, negative_records),
            "order": _artifact_entry(order_path, order_records),
            "dataset": _file_entry(data_path),
            "environment": _file_entry(environment_path),
        },
        "protocol": {
            "model_name": EXPECTED_MODEL_NAME,
            "model_revision": EXPECTED_MODEL_REVISION,
            "data_revision": EXPECTED_DATA_REVISION,
            "data_sha256": EXPECTED_DATA_SHA256,
            "positive_control_sha256": negative_provenance["positive_control_sha256"],
            "repository_commit": expected_repo_commit,
            "settings": EXPECTED_SETTINGS,
        },
        "run_ids": {
            "negative": negative_provenance["run_id"],
            "order": order_provenance["run_id"],
        },
        "software_versions": negative_provenance["software_versions"],
        "environment": environment,
        "negative": _negative_summary(negative_records),
        "order": _order_summary(order_records),
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a plain or gzipped JSONL file and reject malformed records."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    records: list[dict[str, Any]] = []
    try:
        with opener(path, "rt") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path} record {line_number} is not a JSON object")
                records.append(record)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL artifact {path}: {error}") from error
    return records


def artifact_sha256(path: Path) -> str:
    """Hash the exact artifact bytes stored on disk."""
    return _hash_stream(path.open("rb"))


def jsonl_sha256(path: Path) -> str:
    """Hash JSONL content, transparently decompressing a gzip container."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    return _hash_stream(opener(path, "rb"))


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    with stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write a validated summary exclusively and deterministically."""
    with path.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _verify_positive_control(
    path: Path, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], set[str]]:
    if len(records) != 500:
        raise ValueError(f"expected 500 positive-control records, found {len(records)}")
    _require_binary_scores(records, "positive control")
    for index, record in enumerate(records, 1):
        control_id = record.get("control_id")
        if (
            not isinstance(control_id, int)
            or isinstance(control_id, bool)
            or control_id not in range(50)
        ):
            raise ValueError(
                f"positive control record {index} has invalid control_id: {control_id!r}"
            )
        for field in ("prompt", "generation", "gold"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"positive control record {index} has invalid {field}")
        for field in ("prompt_token_count", "generated_token_count"):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"positive control record {index} has invalid {field}: {value!r}")
    metadata = {
        field: _single_value(records, field, "positive control")
        for field in POSITIVE_METADATA_FIELDS
    }
    if metadata["model"] != EXPECTED_MODEL_NAME:
        raise ValueError(f"positive control has wrong model: {metadata['model']!r}")
    if metadata["model_revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"positive control has wrong model revision: {metadata['model_revision']!r}"
        )
    if type(metadata["seed"]) is not int or metadata["seed"] != EXPECTED_SETTINGS["random_seed"]:
        raise ValueError(f"positive control has wrong seed: {metadata['seed']!r}")
    validate_control(path, metadata)
    return metadata, {artifact_sha256(path), jsonl_sha256(path)}


def _expected_questions(data_path: Path) -> set[str]:
    digest = artifact_sha256(data_path)
    if digest != EXPECTED_DATA_SHA256:
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {EXPECTED_DATA_SHA256}, found {digest}"
        )
    rows = read_rows(data_path)
    exploratory, _ = split_indices(len(rows), EXPECTED_SETTINGS["random_seed"])
    return {question_id(rows[index], index) for index in exploratory[:EXPECTED_QUESTIONS]}


def _verify_environment(
    path: Path,
    data_path: Path,
    expected_repo_commit: str,
    expected_python_version: str,
) -> dict[str, Any]:
    if len(expected_repo_commit) != 40 or any(
        character not in string.hexdigits for character in expected_repo_commit
    ):
        raise ValueError("expected repository commit must be a full 40-character SHA")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid environment manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("environment manifest has wrong schema_version")
    repository = manifest.get("repository")
    if not isinstance(repository, dict):
        raise ValueError("environment manifest has invalid repository")
    if repository.get("commit") != expected_repo_commit or repository.get("status") != []:
        raise ValueError("environment manifest has wrong repository provenance")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("environment manifest has invalid dataset")
    if dataset.get("sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("environment manifest has wrong dataset SHA-256")
    dataset_path = dataset.get("path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("environment manifest has invalid dataset path")
    protocol = manifest.get("protocol")
    expected_protocol = {
        "model": "pythia-2.8b",
        "model_revision": EXPECTED_MODEL_REVISION,
        "seed": EXPECTED_SETTINGS["random_seed"],
        "negative_n": EXPECTED_QUESTIONS,
        "negative_positions": list(EXPECTED_POSITIONS),
        "order_n": EXPECTED_QUESTIONS,
        "order_positions": list(EXPECTED_ORDER_POSITIONS),
        "order_permutations": len(EXPECTED_PERMUTATIONS),
        "temperature": EXPECTED_SETTINGS["temperature"],
        "top_p": EXPECTED_SETTINGS["top_p"],
        "top_k": EXPECTED_SETTINGS["top_k"],
        "max_new_tokens": EXPECTED_SETTINGS["max_new_tokens"],
    }
    if not isinstance(protocol, dict) or any(
        field not in protocol
        or type(protocol[field]) is not type(expected)
        or protocol[field] != expected
        for field, expected in expected_protocol.items()
    ):
        raise ValueError("environment manifest has wrong protocol configuration")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("environment manifest has incomplete runtime provenance")
    required_strings = {
        "python_executable": runtime.get("python_executable"),
        "python_version": runtime.get("python_version"),
        "platform": runtime.get("platform"),
        "nvidia_smi": runtime.get("nvidia_smi"),
    }
    if any(not isinstance(value, str) or not value for value in required_strings.values()):
        raise ValueError("environment manifest has incomplete runtime provenance")
    if runtime["python_version"] != expected_python_version:
        raise ValueError("environment manifest has wrong Python version")
    packages = runtime.get("packages")
    if (
        not isinstance(packages, list)
        or not packages
        or any(not isinstance(package, str) or not package for package in packages)
    ):
        raise ValueError("environment manifest has incomplete package provenance")
    if manifest.get("stages") != [
        "positive-control",
        "certify-negative",
        "certify-order",
        "verify-controls",
    ]:
        raise ValueError("environment manifest has wrong stage order")
    return {
        "repository_commit": expected_repo_commit,
        "dataset_sha256": EXPECTED_DATA_SHA256,
        "model_revision": EXPECTED_MODEL_REVISION,
    }


def _verify_negative(records: list[dict[str, Any]], expected_questions: set[str]) -> None:
    if len(records) != EXPECTED_NEGATIVE_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_NEGATIVE_RECORDS} negative records, found {len(records)}"
        )
    _require_binary_scores(records, "negative")
    _require_exact_value(records, "condition", "negative_control", "negative")
    _require_exact_value(records, "gold_present", False, "negative")
    _require_exact_value(records, "permutation_id", None, "negative")
    _require_exact_value(records, "permutation_seed", None, "negative")
    _require_complete_generations(records, "negative")
    if any(
        not isinstance(record.get("gold_position"), int)
        or isinstance(record.get("gold_position"), bool)
        for record in records
    ):
        raise ValueError("negative has invalid gold_position type")
    if any(
        not isinstance(record.get("fake_source_index"), int)
        or isinstance(record.get("fake_source_index"), bool)
        for record in records
    ):
        raise ValueError("negative has invalid fake_source_index")
    questions = _questions(records, "negative")
    if questions != expected_questions:
        raise ValueError("negative control uses the wrong seeded question sample")
    expected = {
        (question_id, position) for question_id in questions for position in EXPECTED_POSITIONS
    }
    observed = {(record.get("question_id"), record.get("gold_position")) for record in records}
    if observed != expected:
        raise ValueError("negative control position coverage is incomplete or duplicated")
    if len(observed) != len(records):
        raise ValueError("negative control position coverage is duplicated")


def _verify_order(records: list[dict[str, Any]], expected_questions: set[str]) -> None:
    if len(records) != EXPECTED_ORDER_RECORDS:
        raise ValueError(f"expected {EXPECTED_ORDER_RECORDS} order records, found {len(records)}")
    _require_binary_scores(records, "order")
    _require_exact_value(records, "condition", "distractor_order", "order")
    _require_exact_value(records, "gold_present", True, "order")
    _require_exact_value(records, "fake_source_index", None, "order")
    _require_complete_generations(records, "order")
    if any(
        not isinstance(record.get("gold_position"), int)
        or isinstance(record.get("gold_position"), bool)
        or not isinstance(record.get("permutation_id"), int)
        or isinstance(record.get("permutation_id"), bool)
        for record in records
    ):
        raise ValueError("order has invalid position or permutation type")
    questions = _questions(records, "order")
    if questions != expected_questions:
        raise ValueError("order control uses the wrong seeded question sample")
    expected = {
        (question_id, position, permutation)
        for question_id in questions
        for position in EXPECTED_ORDER_POSITIONS
        for permutation in EXPECTED_PERMUTATIONS
    }
    observed = {
        (record.get("question_id"), record.get("gold_position"), record.get("permutation_id"))
        for record in records
    }
    if observed != expected:
        raise ValueError("order control position/permutation coverage is incomplete or duplicated")
    if len(observed) != len(records):
        raise ValueError("order control position/permutation coverage is duplicated")
    for record in records:
        permutation = record["permutation_id"]
        expected_seed = (
            None
            if permutation == 0
            else hashlib.sha256(
                f"{record['question_id']}:{record['gold_position']}:{permutation:02d}".encode()
            ).hexdigest()
        )
        if record.get("permutation_seed") != expected_seed:
            raise ValueError("order control has invalid deterministic permutation seed")


def _questions(records: list[dict[str, Any]], label: str) -> set[str]:
    questions: set[str] = set()
    for record in records:
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{label} has invalid question_id: {question_id!r}")
        questions.add(question_id)
    if len(questions) != EXPECTED_QUESTIONS:
        raise ValueError(f"expected {EXPECTED_QUESTIONS} {label} questions, found {len(questions)}")
    return questions


def _verify_provenance(
    records: list[dict[str, Any]], control_hashes: set[str], label: str
) -> dict[str, Any]:
    exact = {
        "model_name": EXPECTED_MODEL_NAME,
        "model_revision": EXPECTED_MODEL_REVISION,
        "data_revision": EXPECTED_DATA_REVISION,
        "data_sha256": EXPECTED_DATA_SHA256,
        **EXPECTED_SETTINGS,
    }
    for field, expected in exact.items():
        _require_exact_value(records, field, expected, label)
    run_id = _single_value(records, "run_id", label)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{label} has invalid run_id: {run_id!r}")
    software_versions = {
        field: _single_nested_value(records, "software_versions", field, label)
        for field in SOFTWARE_FIELDS
    }
    if any(value is None for value in software_versions.values()):
        raise ValueError(f"{label} has null software provenance: {software_versions}")
    control_sha256 = _single_value(records, "positive_control_sha256", label)
    if control_sha256 not in control_hashes:
        raise ValueError(f"{label} has wrong positive_control_sha256: {control_sha256!r}")
    return {
        "run_id": run_id,
        "positive_control_sha256": control_sha256,
        "software_versions": software_versions,
    }


def _require_binary_scores(records: Iterable[dict[str, Any]], label: str) -> None:
    for index, record in enumerate(records, 1):
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{label} record {index} has invalid score: {score!r}")
        if not math.isfinite(float(score)) or float(score) not in (0.0, 1.0):
            raise ValueError(f"{label} record {index} has invalid score: {score!r}")


def _require_complete_generations(records: list[dict[str, Any]], label: str) -> None:
    for index, record in enumerate(records, 1):
        for field in ("floor_accuracy", "ceiling_accuracy"):
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) not in (0.0, 1.0)
            ):
                raise ValueError(f"{label} record {index} has invalid {field}: {value!r}")
        prompt = record.get("prompt")
        response = record.get("model_response")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{label} record {index} has invalid prompt")
        if not isinstance(response, str):
            raise ValueError(f"{label} record {index} has invalid model_response")
        for field in ("prompt_token_count", "generated_token_count"):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} record {index} has invalid {field}: {value!r}")


def _require_exact_value(
    records: list[dict[str, Any]], field: str, expected: Any, label: str
) -> None:
    for record in records:
        if field not in record:
            raise ValueError(f"{label} has missing {field}")
        if type(record[field]) is not type(expected) or record[field] != expected:
            raise ValueError(
                f"{label} has wrong {field}: expected {expected!r}, found {record[field]!r}"
            )


def _single_value(records: list[dict[str, Any]], field: str, label: str) -> Any:
    values = [record.get(field) for record in records]
    first = values[0] if values else None
    if first is None or any(value != first for value in values):
        raise ValueError(f"{label} has mixed or missing {field}")
    return first


def _single_nested_value(records: list[dict[str, Any]], parent: str, field: str, label: str) -> Any:
    values = []
    for record in records:
        nested = record.get(parent)
        if not isinstance(nested, dict):
            raise ValueError(f"{label} has invalid {parent}")
        values.append(nested.get(field))
    first = values[0] if values else None
    if first is None or any(value != first for value in values):
        raise ValueError(f"{label} has mixed or missing {parent}.{field}")
    return first


def _artifact_entry(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "records": len(records),
        "sha256": artifact_sha256(path),
        "jsonl_sha256": jsonl_sha256(path),
    }


def _file_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": artifact_sha256(path)}


def _mean(records: Iterable[dict[str, Any]]) -> float:
    scores = [float(record["score"]) for record in records]
    return sum(scores) / len(scores)


def _negative_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_position = {
        str(position): _mean(record for record in records if record["gold_position"] == position)
        for position in EXPECTED_POSITIONS
    }
    center = (by_position["4"] + by_position["5"]) / 2
    return {
        "accuracy": _mean(records),
        "floor_accuracy": sum(float(record["floor_accuracy"]) for record in records) / len(records),
        "position_accuracy": by_position,
        "primacy": (by_position["0"] + by_position["1"]) / 2 - center,
        "recency": (by_position["8"] + by_position["9"]) / 2 - center,
        "question_count": EXPECTED_QUESTIONS,
        "record_count": len(records),
    }


def _order_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_permutation = {
        str(permutation): _mean(
            record for record in records if record["permutation_id"] == permutation
        )
        for permutation in EXPECTED_PERMUTATIONS
    }
    by_position = {
        str(position): _mean(record for record in records if record["gold_position"] == position)
        for position in EXPECTED_ORDER_POSITIONS
    }
    return {
        "accuracy": _mean(records),
        "accuracy_spread": max(by_permutation.values()) - min(by_permutation.values()),
        "permutation_accuracy": by_permutation,
        "position_accuracy": by_position,
        "question_count": EXPECTED_QUESTIONS,
        "record_count": len(records),
    }


def _require_no_failure_sidecar(path: Path) -> None:
    uncompressed = path.with_suffix("") if path.suffix == ".gz" else path
    candidates = {path.with_suffix(".failures.jsonl"), uncompressed.with_suffix(".failures.jsonl")}
    existing = sorted(str(candidate) for candidate in candidates if candidate.exists())
    if existing:
        raise ValueError(f"scoring-failure sidecar exists: {', '.join(existing)}")


def main() -> None:
    """Parse command-line paths, verify the bundle, and write its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--positive-control", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()
    summary = verify_bundle(
        arguments.negative,
        arguments.order,
        arguments.positive_control,
        arguments.data,
        arguments.environment,
        arguments.expected_repo_commit,
    )
    write_summary(arguments.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
