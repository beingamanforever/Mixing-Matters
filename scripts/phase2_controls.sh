#!/usr/bin/env bash
# Run the strict Phase 2 certification controls for Pythia-2.8B.
# Usage: EXPECTED_REPO_COMMIT=<sha> scripts/phase2_controls.sh <run-directory>
set -Eeuo pipefail

RUN_DIR=${1:?usage: EXPECTED_REPO_COMMIT=<sha> scripts/phase2_controls.sh <run-directory>}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?EXPECTED_REPO_COMMIT is required}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIXING=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
PYTHON=${PYTHON:-$MIXING/.venv/bin/python}
DATA=${DATA:-$MIXING/data/nq-open-10_total_documents_gold_at_0.jsonl.gz}
MODEL=pythia-2.8b
MODEL_REVISION=2a259cdd96a4beb1cdf467512e3904197345f6a9
DATA_SHA256=192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9
SEED=240521
NEGATIVE_N=200
ORDER_N=200
ORDER_POSITIONS=(0 4 9)
ORDER_PERMS=3

fail() {
  echo "$1" >&2
  return 1
}

mark_failed() {
  local status=$?
  local line=${1:-unknown}
  trap - ERR
  set +e
  printf 'exit_status=%s\nline=%s\n' "$status" "$line" > "$RUN_DIR/FAILED"
  exit "$status"
}

run_logged() {
  local log=$1
  shift
  "$@" 2>&1 | tee "$log"
}

if ! mkdir "$RUN_DIR"; then
  fail "could not claim exclusive run directory: $RUN_DIR"
fi
trap 'mark_failed "$LINENO"' ERR

[ -x "$PYTHON" ] || fail "Python is not executable: $PYTHON"
[ -f "$DATA" ] || fail "dataset not found: $DATA"

REPO_STATUS=$(git -C "$MIXING" status --porcelain --untracked-files=normal)
[ -z "$REPO_STATUS" ] || fail "Mixing-Matters checkout is dirty: $MIXING"
REPO_COMMIT=$(git -C "$MIXING" rev-parse HEAD)
[ "$REPO_COMMIT" = "$EXPECTED_REPO_COMMIT" ] || \
  fail "repository must be $EXPECTED_REPO_COMMIT, found $REPO_COMMIT"

FOUND_DATA_SHA256=$(sha256sum "$DATA" | awk '{print $1}')
[ "$FOUND_DATA_SHA256" = "$DATA_SHA256" ] || \
  fail "dataset SHA-256 mismatch: expected $DATA_SHA256, found $FOUND_DATA_SHA256"

export PYTHONPATH="$MIXING/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - "$RUN_DIR/environment.json" "$MIXING" "$REPO_COMMIT" "$DATA" \
  "$FOUND_DATA_SHA256" "$PYTHON" "$MODEL" "$MODEL_REVISION" "$SEED" "$NEGATIVE_N" \
  "$ORDER_N" "$ORDER_PERMS" "${ORDER_POSITIONS[@]}" <<'PY'
"""Write the deterministic Phase 2 control environment manifest."""

import json
import platform
import subprocess
import sys
from pathlib import Path


def capture(command: list[str]) -> str:
    """Capture one required environment command."""
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def main(arguments: list[str]) -> None:
    """Write the validated repository, runtime, and protocol configuration."""
    (
        output,
        repository,
        repository_commit,
        dataset,
        dataset_sha256,
        python,
        model,
        model_revision,
        seed,
        negative_n,
        order_n,
        order_perms,
        *order_positions,
    ) = arguments
    manifest = {
        "schema_version": 1,
        "repository": {"path": repository, "commit": repository_commit, "status": []},
        "dataset": {"path": dataset, "sha256": dataset_sha256},
        "runtime": {
            "python_executable": python,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": sorted(capture([sys.executable, "-m", "pip", "freeze"]).splitlines()),
            "nvidia_smi": capture(["nvidia-smi"]),
        },
        "protocol": {
            "model": model,
            "model_revision": model_revision,
            "seed": int(seed),
            "negative_n": int(negative_n),
            "negative_positions": list(range(10)),
            "order_n": int(order_n),
            "order_positions": [int(position) for position in order_positions],
            "order_permutations": int(order_perms),
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
    with Path(output).open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


main(sys.argv[1:])
PY

POSITIVE="$RUN_DIR/positive-control.jsonl"
NEGATIVE="$RUN_DIR/negative.jsonl"
ORDER="$RUN_DIR/order.jsonl"
SUMMARY="$RUN_DIR/summary.json"
SUMMARY_PENDING="$RUN_DIR/summary.pending.json"

run_logged "$RUN_DIR/positive-control.log" \
  "$PYTHON" -m mixing_matters.cli positive-control \
  --model "$MODEL" --revision "$MODEL_REVISION" --output "$POSITIVE"

run_logged "$RUN_DIR/negative.log" \
  "$PYTHON" -m mixing_matters.cli certify-negative \
  --data "$DATA" --output "$NEGATIVE" --revision "$MODEL_REVISION" \
  --positive-control "$POSITIVE" --n "$NEGATIVE_N"

run_logged "$RUN_DIR/order.log" \
  "$PYTHON" -m mixing_matters.cli certify-order \
  --data "$DATA" --output "$ORDER" --revision "$MODEL_REVISION" \
  --positive-control "$POSITIVE" --n "$ORDER_N" \
  --positions "${ORDER_POSITIONS[@]}" --perms "$ORDER_PERMS"

FINAL_REPO_STATUS=$(git -C "$MIXING" status --porcelain --untracked-files=normal)
[ -z "$FINAL_REPO_STATUS" ] || fail "Mixing-Matters checkout changed during the run: $MIXING"
FINAL_REPO_COMMIT=$(git -C "$MIXING" rev-parse HEAD)
[ "$FINAL_REPO_COMMIT" = "$EXPECTED_REPO_COMMIT" ] || \
  fail "repository changed during the run: expected $EXPECTED_REPO_COMMIT, found $FINAL_REPO_COMMIT"

run_logged "$RUN_DIR/verification.log" \
  "$PYTHON" "$MIXING/scripts/verify_phase2_controls.py" \
  --negative "$NEGATIVE" --order "$ORDER" --positive-control "$POSITIVE" \
  --data "$DATA" --environment "$RUN_DIR/environment.json" \
  --expected-repo-commit "$EXPECTED_REPO_COMMIT" --summary "$SUMMARY_PENDING"

COMPLETE_REPO_STATUS=$(git -C "$MIXING" status --porcelain --untracked-files=normal)
[ -z "$COMPLETE_REPO_STATUS" ] || \
  fail "Mixing-Matters checkout changed during verification: $MIXING"
COMPLETE_REPO_COMMIT=$(git -C "$MIXING" rev-parse HEAD)
[ "$COMPLETE_REPO_COMMIT" = "$EXPECTED_REPO_COMMIT" ] || \
  fail "repository changed during verification: expected $EXPECTED_REPO_COMMIT, found $COMPLETE_REPO_COMMIT"

mv "$SUMMARY_PENDING" "$SUMMARY"
touch "$RUN_DIR/COMPLETE"
trap - ERR
echo "Phase 2 controls complete: $RUN_DIR"
