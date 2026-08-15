#!/usr/bin/env bash
# Run the matched Phase 3 checkpoints through validate, control, and sweep.
# Usage: scripts/phase3.sh <run-directory>
# The run directory and every output below it are immutable.
set -euo pipefail

RUN_DIR=${1:?usage: scripts/phase3.sh <run-directory>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIXING=${MIXING:-$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)}
MEGATRON=${MEGATRON:-/root/Megatron-LM}
VENV=${VENV:-/root/mm-venv/bin/python}
CKPT_ROOT=${CKPT_ROOT:-/root/checkpoints}
DATA=$MIXING/data/nq-open-10_total_documents_gold_at_0.jsonl.gz
PIQA=$MIXING/data/piqa_valid.jsonl
LAUNCHER=${LAUNCHER:-$MIXING/scripts/run_phase3_baremetal.sh}
PIQA_SAMPLES=${MM_PIQA_SAMPLES:-1838}
QUESTIONS=${MM_QUESTIONS:-800}
EXPECTED_PIQA_SAMPLES=1838
EXPECTED_QUESTIONS=800
EXPECTED_DATA_SHA256=192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9
EXPECTED_PIQA_SHA256=61533005e22f175534909b1ec8eacb6da03c233933558d1bafb15787453b1f55
EXPECTED_MEGATRON_SHA=df61e60bf5670b1196fcae2264311401d3bb82db
PURE_REVISION=b915550c63ba9359f88f44d1f6a600d85af27302
HYBRID_REVISION=35e8852e2240b350ac2fe2a3b8aa341b5930018e
PURE_WEIGHT_SHA256=47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb
HYBRID_WEIGHT_SHA256=480964a9dca70e3dcbf348bbf812ed2388476c2b2968cb903cca666d08547399
TOKENIZER_SHA256=5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09
LATEST_SHA256=3933e3274b63dc01fd286d6d939a626867c83a4603fc4347e0f8b8856f1b98fd
TOKENIZER=mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model

fail() {
  echo "$1" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || fail "required file not found: $1"
}

require_sha256() {
  local path=$1
  local expected=$2
  local digest
  digest=$(sha256sum "$path" | awk '{print $1}')
  [ "$digest" = "$expected" ] || \
    fail "SHA-256 mismatch for $path: expected $expected, found $digest"
  printf '%s' "$digest"
}

[ ! -e "$RUN_DIR" ] || fail "run directory already exists: $RUN_DIR"
[ -x "$VENV" ] || fail "venv python is not executable: $VENV"
[ "$PIQA_SAMPLES" = "$EXPECTED_PIQA_SAMPLES" ] || \
  fail "MM_PIQA_SAMPLES must be $EXPECTED_PIQA_SAMPLES, found $PIQA_SAMPLES"
[ "$QUESTIONS" = "$EXPECTED_QUESTIONS" ] || \
  fail "MM_QUESTIONS must be $EXPECTED_QUESTIONS, found $QUESTIONS"
CANONICAL_LAUNCHER=$MIXING/scripts/run_phase3_baremetal.sh
[ "$LAUNCHER" = "$CANONICAL_LAUNCHER" ] || \
  fail "launcher must be the canonical repository launcher: $CANONICAL_LAUNCHER"
require_file "$LAUNCHER"

MIXING_STATUS=$(git -C "$MIXING" status --porcelain --untracked-files=normal)
[ -z "$MIXING_STATUS" ] || fail "Mixing-Matters checkout is dirty: $MIXING"
MIXING_SHA=$(git -C "$MIXING" rev-parse HEAD)

MEGATRON_STATUS=$(git -C "$MEGATRON" status --porcelain --untracked-files=normal)
[ -z "$MEGATRON_STATUS" ] || fail "Megatron checkout is dirty: $MEGATRON"
MEGATRON_SHA=$(git -C "$MEGATRON" rev-parse HEAD)
[ "$MEGATRON_SHA" = "$EXPECTED_MEGATRON_SHA" ] || \
  fail "Megatron checkout must be $EXPECTED_MEGATRON_SHA, found $MEGATRON_SHA"

require_file "$DATA"
require_file "$PIQA"
for model in mamba2-8b mamba2-hybrid-8b; do
  require_file "$CKPT_ROOT/$model/latest_checkpointed_iteration.txt"
  require_file "$CKPT_ROOT/$model/release/mp_rank_00/model_optim_rng.pt"
  require_file "$CKPT_ROOT/$model/$TOKENIZER"
done

DATA_DIGEST=$(sha256sum "$DATA" | awk '{print $1}')
[ "$DATA_DIGEST" = "$EXPECTED_DATA_SHA256" ] || \
  fail "dataset SHA-256 mismatch: expected $EXPECTED_DATA_SHA256, found $DATA_DIGEST"
PIQA_DIGEST=$(sha256sum "$PIQA" | awk '{print $1}')
[ "$PIQA_DIGEST" = "$EXPECTED_PIQA_SHA256" ] || \
  fail "PIQA SHA-256 mismatch: expected $EXPECTED_PIQA_SHA256, found $PIQA_DIGEST"
LAUNCHER_DIGEST=$(sha256sum "$LAUNCHER" | awk '{print $1}')
PURE_WEIGHT_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-8b/release/mp_rank_00/model_optim_rng.pt" "$PURE_WEIGHT_SHA256")
HYBRID_WEIGHT_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-hybrid-8b/release/mp_rank_00/model_optim_rng.pt" "$HYBRID_WEIGHT_SHA256")
PURE_TOKENIZER_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-8b/$TOKENIZER" "$TOKENIZER_SHA256")
HYBRID_TOKENIZER_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-hybrid-8b/$TOKENIZER" "$TOKENIZER_SHA256")
PURE_LATEST_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-8b/latest_checkpointed_iteration.txt" "$LATEST_SHA256")
HYBRID_LATEST_DIGEST=$(require_sha256 \
  "$CKPT_ROOT/mamba2-hybrid-8b/latest_checkpointed_iteration.txt" "$LATEST_SHA256")

mkdir -p "$RUN_DIR"
"$VENV" - "$RUN_DIR" "$MIXING" "$MIXING_SHA" "$MEGATRON" "$MEGATRON_SHA" \
  "$DATA" "$DATA_DIGEST" "$PIQA" "$PIQA_DIGEST" "$CKPT_ROOT" "$LAUNCHER" "$PIQA_SAMPLES" \
  "$LAUNCHER_DIGEST" "$QUESTIONS" "$PURE_REVISION" "$HYBRID_REVISION" "$PURE_WEIGHT_DIGEST" \
  "$HYBRID_WEIGHT_DIGEST" "$PURE_TOKENIZER_DIGEST" "$HYBRID_TOKENIZER_DIGEST" \
  "$PURE_LATEST_DIGEST" "$HYBRID_LATEST_DIGEST" <<'PY'
"""Write the typed Phase 3 environment manifest before execution."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


class RepositoryState(TypedDict):
    """Pinned state of one source checkout."""

    path: str
    sha: str
    status: list[str]


class DatasetState(TypedDict):
    """Identity of one immutable evaluation input."""

    path: str
    sha256: str


class Configuration(TypedDict):
    """Run parameters shared by both checkpoint stages."""

    checkpoint_root: str
    launcher: str
    launcher_sha256: str
    output_directory: str
    piqa_file: str
    piqa_samples: int
    questions: int
    venv_python: str


class CommandStage(TypedDict):
    """One ordered launcher invocation."""

    model: str
    stage: str
    command: list[str]


class EnvironmentManifest(TypedDict):
    """Complete provenance required to reproduce a Phase 3 run."""

    schema_version: int
    created_at_utc: str
    method: str
    python: str
    platform: str
    repositories: dict[str, RepositoryState]
    dataset: DatasetState
    piqa: DatasetState
    packages: list[str]
    nvidia_smi: str
    checkpoint_revisions: dict[str, str]
    checkpoint_files: dict[str, dict[str, str]]
    configuration: Configuration
    command_stages: list[CommandStage]


def main(arguments: list[str]) -> None:
    """Create an exclusive environment manifest from validated inputs."""
    (
        run_directory,
        mixing_path,
        mixing_sha,
        megatron_path,
        megatron_sha,
        data_path,
        data_sha256,
        piqa_path,
        piqa_sha256,
        checkpoint_root,
        launcher,
        piqa_samples,
        launcher_sha256,
        questions,
        pure_revision,
        hybrid_revision,
        pure_weight_sha256,
        hybrid_weight_sha256,
        pure_tokenizer_sha256,
        hybrid_tokenizer_sha256,
        pure_latest_sha256,
        hybrid_latest_sha256,
    ) = arguments
    stage_pairs = (
        ("mamba2-8b", "validate"),
        ("mamba2-hybrid-8b", "validate"),
        ("mamba2-8b", "kv"),
        ("mamba2-8b", "sweep"),
        ("mamba2-hybrid-8b", "kv"),
        ("mamba2-hybrid-8b", "sweep"),
    )
    command_stages: list[CommandStage] = [
        {
            "model": model,
            "stage": stage,
            "command": ["bash", launcher, model, stage],
        }
        for model, stage in stage_pairs
    ]
    manifest: EnvironmentManifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "bare-metal Megatron-LM",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repositories": {
            "mixing_matters": {"path": mixing_path, "sha": mixing_sha, "status": []},
            "megatron": {"path": megatron_path, "sha": megatron_sha, "status": []},
        },
        "dataset": {"path": data_path, "sha256": data_sha256},
        "piqa": {"path": piqa_path, "sha256": piqa_sha256},
        "packages": _capture([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "nvidia_smi": _capture(["nvidia-smi"]),
        "checkpoint_revisions": {
            "mamba2-8b": pure_revision,
            "mamba2-hybrid-8b": hybrid_revision,
        },
        "checkpoint_files": {
            "mamba2-8b": {
                "latest_checkpointed_iteration.txt": pure_latest_sha256,
                "model_optim_rng.pt": pure_weight_sha256,
                "tokenizer.model": pure_tokenizer_sha256,
            },
            "mamba2-hybrid-8b": {
                "latest_checkpointed_iteration.txt": hybrid_latest_sha256,
                "model_optim_rng.pt": hybrid_weight_sha256,
                "tokenizer.model": hybrid_tokenizer_sha256,
            },
        },
        "configuration": {
            "checkpoint_root": checkpoint_root,
            "launcher": launcher,
            "launcher_sha256": launcher_sha256,
            "output_directory": run_directory,
            "piqa_file": piqa_path,
            "piqa_samples": int(piqa_samples),
            "questions": int(questions),
            "venv_python": sys.executable,
        },
        "command_stages": command_stages,
    }
    manifest_path = Path(run_directory, "environment.json")
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


def _capture(command: list[str]) -> str:
    """Capture one required environment command."""
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


main(sys.argv[1:])
PY

run_stage() {
  local model=$1
  local stage=$2
  local log="$RUN_DIR/$model-$stage.log"
  local target
  echo "== Phase 3 $model $stage"
  MIXING="$MIXING" MEGATRON="$MEGATRON" CKPT_ROOT="$CKPT_ROOT" OUT_DIR="$RUN_DIR" \
    VENV="$VENV" MM_PIQA_SAMPLES="$PIQA_SAMPLES" MM_QUESTIONS="$QUESTIONS" \
    bash "$LAUNCHER" "$model" "$stage" 2>&1 | tee "$log"
  if [ "$stage" = validate ]; then
    if [ "$model" = mamba2-8b ]; then
      target='79\.82'
    else
      target='79\.65'
    fi
    grep -Eq "^PIQA acc=[0-9]+\\.[0-9]{2} target=$target delta=[+-][0-9]+\\.[0-9]{2} \\[PASS\\]$" \
      "$log" || fail "PIQA validation did not pass for $model; see $log"
  elif [ "$stage" = kv ]; then
    validate_artifact "$model" "$stage" "$RUN_DIR/$model-positive-control.jsonl"
  else
    validate_artifact "$model" "$stage" "$RUN_DIR/$model-sweep.jsonl"
  fi
}

validate_artifact() {
  local model=$1
  local stage=$2
  local artifact=$3
  "$VENV" - "$model" "$stage" "$artifact" <<'PY'
"""Validate one completed Phase 3 artifact before continuing."""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

KV_CONDITIONS = {f"kv_position_{position}" for position in range(10)}
SWEEP_CONDITIONS = {"closed_book", "oracle", "gold"}
EXPECTED_REVISIONS = {
    "mamba2-8b": "b915550c63ba9359f88f44d1f6a600d85af27302",
    "mamba2-hybrid-8b": "35e8852e2240b350ac2fe2a3b8aa341b5930018e",
}


def fail(message: str) -> None:
    """Stop artifact validation with a concise diagnostic."""
    raise ValueError(message)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact and reject malformed or non-object records."""
    records: list[dict[str, Any]] = []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                if not isinstance(record, dict):
                    fail(f"record {line_number} is not a JSON object")
                records.append(record)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid JSONL: {error}")
    return records


def require_fields(record: dict[str, Any], fields: set[str], index: int) -> None:
    """Require the stable fields used to audit one record."""
    missing = sorted(fields - record.keys())
    if missing:
        fail(f"record {index} is missing fields: {', '.join(missing)}")


def require_score(value: Any, field: str, index: int) -> None:
    """Require a finite, non-null numeric score."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
    ):
        fail(f"record {index} has invalid {field}: {value!r}")


def validate_common(record: dict[str, Any], model: str, index: int) -> None:
    """Validate model identity and the required Megatron execution path."""
    require_fields(record, {"model_revision", "score"}, index)
    if record["model_revision"] != EXPECTED_REVISIONS[model]:
        fail(f"record {index} has the wrong model revision")
    require_score(record["score"], "score", index)


def validate_kv(records: list[dict[str, Any]], model: str) -> None:
    """Validate all 50 by 10 key-value control conditions exactly once."""
    if len(records) != 500:
        fail(f"expected 500 KV records, found {len(records)}")
    keys: list[tuple[int, str]] = []
    for index, record in enumerate(records, 1):
        require_fields(
            record,
            {"control_id", "condition", "model_key", "execution_path"},
            index,
        )
        validate_common(record, model, index)
        control_id = record["control_id"]
        condition = record["condition"]
        if (
            isinstance(control_id, bool)
            or not isinstance(control_id, int)
            or control_id not in range(50)
        ):
            fail(f"record {index} has invalid control_id: {control_id!r}")
        if condition not in KV_CONDITIONS:
            fail(f"record {index} has invalid KV condition: {condition!r}")
        if record["model_key"] != model:
            fail(f"record {index} has the wrong model_key")
        if record["execution_path"] != "megatron_cuda_kernels":
            fail(f"record {index} did not use Megatron CUDA kernels")
        keys.append((control_id, condition))
    if len(set(keys)) != 500:
        fail("KV conditions are duplicated or incomplete")


def validate_sweep(records: list[dict[str, Any]], model: str) -> None:
    """Validate 800 complete question bundles with 12 conditions each."""
    if len(records) != 9600:
        fail(f"expected 9600 sweep records, found {len(records)}")
    keys: list[tuple[str, str, int | None]] = []
    conditions: Counter[str] = Counter()
    for index, record in enumerate(records, 1):
        require_fields(
            record,
            {
                "question_id",
                "condition",
                "gold_position",
                "score_normalized_em",
                "score_first_line",
                "floor_accuracy",
                "ceiling_accuracy",
                "software_versions",
            },
            index,
        )
        validate_common(record, model, index)
        question_id = record["question_id"]
        condition = record["condition"]
        position = record["gold_position"]
        if not isinstance(question_id, str) or not question_id:
            fail(f"record {index} has invalid question_id")
        if condition not in SWEEP_CONDITIONS:
            fail(f"record {index} has invalid sweep condition: {condition!r}")
        if condition == "gold":
            if isinstance(position, bool) or not isinstance(position, int) or position not in range(10):
                fail(f"record {index} has invalid gold_position: {position!r}")
        elif position is not None:
            fail(f"record {index} has non-null anchor position")
        for field in ("score_normalized_em", "score_first_line", "floor_accuracy", "ceiling_accuracy"):
            require_score(record[field], field, index)
        versions = record["software_versions"]
        if not isinstance(versions, dict):
            fail(f"record {index} has invalid software_versions")
        if versions.get("model_key") != model:
            fail(f"record {index} has the wrong model_key")
        if versions.get("execution_path") != "megatron_cuda_kernels":
            fail(f"record {index} did not use Megatron CUDA kernels")
        keys.append((question_id, condition, position))
        conditions[condition] += 1
    if len(set(keys)) != 9600:
        fail("sweep conditions are duplicated or incomplete")
    if conditions != {"closed_book": 800, "oracle": 800, "gold": 8000}:
        fail(f"unexpected sweep condition counts: {dict(conditions)}")
    if len({question_id for question_id, _, _ in keys}) != 800:
        fail("sweep must contain exactly 800 question bundles")


def main(arguments: list[str]) -> None:
    """Validate the requested artifact and reject any failure sidecar."""
    model, stage, artifact_text = arguments
    artifact = Path(artifact_text)
    failure_sidecar = artifact.with_suffix(".failures.jsonl")
    if failure_sidecar.exists():
        fail(f"scoring-failure sidecar exists: {failure_sidecar}")
    records = read_records(artifact)
    if stage == "kv":
        validate_kv(records, model)
    else:
        validate_sweep(records, model)


main(sys.argv[1:])
PY
}

run_stage mamba2-8b validate
run_stage mamba2-hybrid-8b validate
run_stage mamba2-8b kv
run_stage mamba2-8b sweep
run_stage mamba2-hybrid-8b kv
run_stage mamba2-hybrid-8b sweep

echo "Phase 3 run complete: $RUN_DIR"
