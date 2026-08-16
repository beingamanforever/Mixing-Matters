"""Tests for the strict Phase 2 certification-control wrapper and verifier."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
WRAPPER = REPOSITORY_ROOT / "scripts" / "phase2_controls.sh"
VERIFIER = REPOSITORY_ROOT / "scripts" / "verify_phase2_controls.py"
MODEL_REVISION = "2a259cdd96a4beb1cdf467512e3904197345f6a9"
DATA_REVISION = "29b8a6d042ce29abccee3db1a73171a107d7e6af"
DATA_SHA256 = "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9"
REPO_COMMIT = "a" * 40


@dataclass(frozen=True)
class ControlBundle:
    """Paths forming one complete verifier input bundle."""

    negative: Path
    order: Path
    positive: Path
    data: Path
    environment: Path

    def __iter__(self) -> Iterator[Path]:
        """Keep the three raw control paths convenient to unpack in tests."""
        yield self.negative
        yield self.order
        yield self.positive


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    """Load the verifier as a module without making scripts a package."""
    spec = importlib.util.spec_from_file_location("verify_phase2_controls", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pinned_dataset(tmp_path_factory: pytest.TempPathFactory, verifier: ModuleType) -> Path:
    """Create a structurally valid dataset and pin its test-only digest."""
    path = tmp_path_factory.mktemp("phase2-data") / "data.jsonl"
    rows = [
        {
            "question": f"question {index}",
            "answers": ["answer"],
            "ctxs": [
                {"title": f"title {document}", "text": "text", "isgold": document == 0}
                for document in range(10)
            ],
        }
        for index in range(2655)
    ]
    _write_jsonl(path, rows)
    verifier.EXPECTED_DATA_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


@pytest.fixture
def exact_bundle(tmp_path: Path, verifier: ModuleType, pinned_dataset: Path) -> ControlBundle:
    """Write one exact, statistically flat control bundle."""
    positive = tmp_path / "positive-control.jsonl"
    _write_jsonl(positive, _positive_records())
    control_hash = hashlib.sha256(positive.read_bytes()).hexdigest()
    questions = sorted(verifier._expected_questions(pinned_dataset))
    data_hash = verifier.EXPECTED_DATA_SHA256
    negative = tmp_path / "negative.jsonl"
    order = tmp_path / "order.jsonl"
    environment = tmp_path / "environment.json"
    _write_jsonl(negative, _negative_records(control_hash, questions, data_hash))
    _write_jsonl(order, _order_records(control_hash, questions, data_hash))
    environment.write_text(
        json.dumps(_environment_manifest(pinned_dataset, data_hash), sort_keys=True) + "\n"
    )
    return ControlBundle(negative, order, positive, pinned_dataset, environment)


def test_wrapper_pins_exact_protocol_and_completion_order() -> None:
    """The wrapper must expose no reduced or reordered canonical control path."""
    script = WRAPPER.read_text()

    assert "EXPECTED_REPO_COMMIT is required" in script
    assert f"MODEL_REVISION={MODEL_REVISION}" in script
    assert f"DATA_SHA256={DATA_SHA256}" in script
    assert "NEGATIVE_N=200" in script
    assert "ORDER_N=200" in script
    assert "ORDER_POSITIONS=(0 4 9)" in script
    assert "ORDER_PERMS=3" in script
    positive = "mixing_matters.cli positive-control"
    negative = "mixing_matters.cli certify-negative"
    order = "mixing_matters.cli certify-order"
    assert script.index(positive) < script.index(negative)
    assert script.index(negative) < script.index(order)
    assert script.index(order) < script.index("verify_phase2_controls.py")
    assert script.index("verify_phase2_controls.py") < script.index('touch "$RUN_DIR/COMPLETE"')
    assert script.index("COMPLETE_REPO_STATUS") < script.index('touch "$RUN_DIR/COMPLETE"')
    assert script.index('mv "$SUMMARY_PENDING" "$SUMMARY"') < script.index(
        'touch "$RUN_DIR/COMPLETE"'
    )
    assert "trap 'mark_failed \"$LINENO\"' ERR" in script


def test_wrapper_marks_failed_run(tmp_path: Path) -> None:
    """A failure after claiming the output directory must leave FAILED only."""
    run_directory = tmp_path / "run"
    environment = os.environ.copy()
    environment.update(
        {
            "DATA": str(tmp_path / "missing-dataset.jsonl.gz"),
            "EXPECTED_REPO_COMMIT": "unused",
            "PYTHON": sys.executable,
        }
    )

    result = subprocess.run(
        ["bash", str(WRAPPER), str(run_directory)],
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "dataset not found" in result.stderr
    assert (run_directory / "FAILED").read_text().startswith("exit_status=1\nline=")
    assert not (run_directory / "COMPLETE").exists()


def test_verifier_accepts_exact_plain_and_gzipped_bundles(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
    tmp_path: Path,
) -> None:
    """Exact JSONL and transparently compressed artifacts must both pass."""
    negative, order, positive = exact_bundle
    summary = _verify_bundle(verifier, exact_bundle)

    assert summary["status"] == "passed"
    assert summary["negative"]["record_count"] == 2000
    assert summary["order"]["record_count"] == 1800
    assert summary["negative"]["primacy"] == 0.0
    assert summary["order"]["accuracy_spread"] == 0.0

    gz_negative = tmp_path / "negative.jsonl.gz"
    gz_order = tmp_path / "order.jsonl.gz"
    gz_positive = tmp_path / "positive-control.jsonl.gz"
    for source, target in (
        (negative, gz_negative),
        (order, gz_order),
        (positive, gz_positive),
    ):
        with source.open("rb") as input_stream, gzip.open(target, "wb") as output_stream:
            output_stream.write(input_stream.read())

    gz_summary = _verify_bundle(verifier, exact_bundle, gz_negative, gz_order, gz_positive)
    assert (
        gz_summary["protocol"]["positive_control_sha256"]
        == summary["protocol"]["positive_control_sha256"]
    )


@pytest.mark.parametrize("artifact", ["negative", "order"])
def test_verifier_rejects_wrong_counts(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
    artifact: str,
) -> None:
    """A truncated raw control must never reach the statistical gates."""
    negative, order, positive = exact_bundle
    path = negative if artifact == "negative" else order
    records = verifier.read_records(path)[:-1]
    _replace_jsonl(path, records)

    with pytest.raises(ValueError, match=f"expected .* {artifact} records"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_wrong_coverage(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Duplicate keys cannot substitute for a missing position/permutation cell."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(order)
    records[-1] = deepcopy(records[0])
    _replace_jsonl(order, records)

    with pytest.raises(ValueError, match="coverage"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_wrong_seeded_question_sample(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Structurally complete arbitrary question IDs are not the pinned split."""
    negative, _, _ = exact_bundle
    records = verifier.read_records(negative)
    replaced_question = records[0]["question_id"]
    for record in records:
        if record["question_id"] == replaced_question:
            record["question_id"] = "wrong-question"
    _replace_jsonl(negative, records)

    with pytest.raises(ValueError, match="wrong seeded question sample"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_null_score(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Null scores are failed generations, not valid control evidence."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(negative)
    records[0]["score"] = None
    _replace_jsonl(negative, records)

    with pytest.raises(ValueError, match="invalid score: None"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_incomplete_positive_generation(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Positive evidence must preserve the raw prompt and generation."""
    _, _, positive = exact_bundle
    records = verifier.read_records(positive)
    del records[0]["generation"]
    _replace_jsonl(positive, records)

    with pytest.raises(ValueError, match="invalid generation"):
        _verify_bundle(verifier, exact_bundle)


@pytest.mark.parametrize(("field", "value"), [("control_id", False), ("seed", 240521.0)])
def test_verifier_rejects_positive_type_substitution(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
    field: str,
    value: object,
) -> None:
    """Booleans and floats cannot substitute for positive-control integers."""
    _, _, positive = exact_bundle
    records = verifier.read_records(positive)
    if field == "seed":
        for record in records:
            record[field] = value
    else:
        records[0][field] = value
    _replace_jsonl(positive, records)

    with pytest.raises(ValueError, match=f"(invalid {field}|wrong {field})"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_incomplete_environment(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """A missing runtime field makes the environment manifest incomplete."""
    manifest = json.loads(exact_bundle.environment.read_text())
    del manifest["runtime"]["platform"]
    exact_bundle.environment.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="incomplete runtime provenance"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_missing_null_setting(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """A missing top-k field cannot masquerade as its required null value."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(negative)
    for record in records:
        del record["top_k"]
    _replace_jsonl(negative, records)

    with pytest.raises(ValueError, match="missing top_k"):
        _verify_bundle(verifier, exact_bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_sha256", "wrong", "wrong data_sha256"),
        ("positive_control_sha256", "wrong", "wrong positive_control_sha256"),
        ("run_id", "order-run", "distinct run IDs"),
    ],
)
def test_verifier_rejects_wrong_provenance(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
    field: str,
    value: str,
    message: str,
) -> None:
    """Dataset, control, and run identity must stay exact and control-local."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(negative)
    for record in records:
        record[field] = value
    _replace_jsonl(negative, records)

    with pytest.raises(ValueError, match=message):
        _verify_bundle(verifier, exact_bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_revision", "wrong"), ("temperature", 0.5), ("max_new_tokens", 31)],
)
def test_verifier_rejects_wrong_model_or_settings(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
    field: str,
    value: object,
) -> None:
    """Model and decoding settings are exact protocol constants."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(order)
    for record in records:
        record[field] = value
    _replace_jsonl(order, records)

    with pytest.raises(ValueError, match=f"wrong {field}"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_boolean_numeric_substitution(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """JSON booleans cannot substitute for numeric protocol constants."""
    negative, _, _ = exact_bundle
    records = verifier.read_records(negative)
    for record in records:
        record["temperature"] = False
    _replace_jsonl(negative, records)

    with pytest.raises(ValueError, match="wrong temperature"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_failure_sidecar(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Any scoring-failure sidecar makes the bundle ineligible."""
    negative, order, positive = exact_bundle
    negative.with_suffix(".failures.jsonl").write_text("")

    with pytest.raises(ValueError, match="scoring-failure sidecar exists"):
        _verify_bundle(verifier, exact_bundle)


def test_verifier_rejects_mixed_software_provenance(
    verifier: ModuleType,
    exact_bundle: ControlBundle,
) -> None:
    """Every record must carry one complete invariant runtime provenance."""
    negative, order, positive = exact_bundle
    records = verifier.read_records(order)
    records[0]["software_versions"]["torch"] = "wrong"
    _replace_jsonl(order, records)

    with pytest.raises(ValueError, match="mixed or missing software_versions.torch"):
        _verify_bundle(verifier, exact_bundle)


def _verify_bundle(
    verifier: ModuleType,
    bundle: ControlBundle,
    negative: Path | None = None,
    order: Path | None = None,
    positive: Path | None = None,
) -> dict[str, object]:
    return verifier.verify_bundle(
        negative or bundle.negative,
        order or bundle.order,
        positive or bundle.positive,
        bundle.data,
        bundle.environment,
        REPO_COMMIT,
    )


def _positive_records() -> list[dict[str, object]]:
    metadata = {
        "model": "EleutherAI/pythia-2.8b",
        "model_revision": MODEL_REVISION,
        "seed": 240521,
        "python": "3.12.0",
        "torch": "2.7.1",
        "transformers": "4.57.1",
        "cuda": "12.6",
        "gpu": "NVIDIA A40",
        "attention_implementation": "eager",
    }
    return [
        {
            "control_id": control_id,
            "condition": f"kv_position_{position}",
            "prompt": "prompt",
            "generation": "generation",
            "gold": "gold",
            "score": float(position in (0, 9)),
            "prompt_token_count": 100,
            "generated_token_count": 1,
            **metadata,
        }
        for control_id in range(50)
        for position in range(10)
    ]


def _negative_records(
    control_hash: str, questions: list[str], data_hash: str
) -> list[dict[str, object]]:
    return [
        {
            **_raw_provenance("negative-run", control_hash, data_hash),
            "condition": "negative_control",
            "gold_present": False,
            "fake_source_index": 1,
            "question_id": question,
            "source_index": source_index,
            "gold_position": position,
            "permutation_id": None,
            "permutation_seed": None,
            "score": 0.0,
            "floor_accuracy": 0.0,
            "ceiling_accuracy": 1.0,
            "prompt": "prompt",
            "model_response": "response",
            "prompt_token_count": 100,
            "generated_token_count": 1,
        }
        for source_index, question in enumerate(questions)
        for position in range(10)
    ]


def _order_records(
    control_hash: str, questions: list[str], data_hash: str
) -> list[dict[str, object]]:
    return [
        {
            **_raw_provenance("order-run", control_hash, data_hash),
            "condition": "distractor_order",
            "gold_present": True,
            "fake_source_index": None,
            "question_id": question,
            "source_index": source_index,
            "gold_position": position,
            "permutation_id": permutation,
            "permutation_seed": _permutation_seed(question, position, permutation),
            "score": 0.0,
            "floor_accuracy": 0.0,
            "ceiling_accuracy": 1.0,
            "prompt": "prompt",
            "model_response": "response",
            "prompt_token_count": 100,
            "generated_token_count": 1,
        }
        for source_index, question in enumerate(questions)
        for position in (0, 4, 9)
        for permutation in range(3)
    ]


def _raw_provenance(run_id: str, control_hash: str, data_hash: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
        "max_new_tokens": 32,
        "random_seed": 240521,
        "manual_seed": 240521,
        "model_name": "EleutherAI/pythia-2.8b",
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "data_sha256": data_hash,
        "positive_control_sha256": control_hash,
        "software_versions": {
            "python": "3.12.0",
            "torch": "2.7.1",
            "transformers": "4.57.1",
            "cuda": "12.6",
            "attention_implementation": "eager",
        },
    }


def _permutation_seed(question: str, position: int, permutation: int) -> str | None:
    if permutation == 0:
        return None
    value = f"{question}:{position}:{permutation:02d}"
    return hashlib.sha256(value.encode()).hexdigest()


def _environment_manifest(data_path: Path, data_hash: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": {"path": "/repo", "commit": REPO_COMMIT, "status": []},
        "dataset": {"path": str(data_path), "sha256": data_hash},
        "runtime": {
            "python_executable": sys.executable,
            "python_version": "3.12.0",
            "platform": "test",
            "packages": ["torch==2.7.1"],
            "nvidia_smi": "NVIDIA-SMI test",
        },
        "protocol": {
            "model": "pythia-2.8b",
            "model_revision": MODEL_REVISION,
            "seed": 240521,
            "negative_n": 200,
            "negative_positions": list(range(10)),
            "order_n": 200,
            "order_positions": [0, 4, 9],
            "order_permutations": 3,
            "temperature": 0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 32,
        },
        "stages": [
            "positive-control",
            "certify-negative",
            "certify-order",
            "verify-controls",
        ],
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _replace_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.unlink()
    _write_jsonl(path, records)
