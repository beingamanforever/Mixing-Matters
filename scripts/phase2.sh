#!/usr/bin/env bash
# Phase 2 end-to-end run for one model: key-value positive control (that
# model's own retrieval capability, not gated on), the ten-position sweep
# plus closed-book floor and oracle ceiling, and environment capture.
# Usage: scripts/phase2.sh <run-directory> <model-key> [model-revision]
# The run directory must not already exist. Raw outputs are never overwritten.
set -euo pipefail

RUN_DIR=${1:?usage: scripts/phase2.sh <run-directory> <model-key> [model-revision]}
MODEL_KEY=${2:?usage: scripts/phase2.sh <run-directory> <model-key> [model-revision]}
REVISION=${3:-}
PYTHON=${PYTHON:-.venv/bin/python}
DATA=${DATA:-data/nq-open-10_total_documents_gold_at_0.jsonl.gz}

if [ -e "$RUN_DIR" ]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

if [ ! -f "$DATA" ]; then
  "$PYTHON" -m mixing_matters.cli download --output "$DATA"
fi

"$PYTHON" - "$RUN_DIR" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

def capture(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": capture([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    "nvidia_smi": capture(["nvidia-smi"]),
    "git_commit": capture(["git", "rev-parse", "HEAD"]),
    "git_status": capture(["git", "status", "--porcelain"]).splitlines(),
}
Path(sys.argv[1], "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
PY

REVISION_ARGS=()
if [ -n "$REVISION" ]; then
  REVISION_ARGS=(--revision "$REVISION")
fi

# A model that cannot do key-value retrieval is a finding about the model,
# not a reason to abort the sweep, so this step never gates what follows.
"$PYTHON" -m mixing_matters.cli positive-control \
  --model "$MODEL_KEY" \
  --output "$RUN_DIR/positive-control.jsonl" \
  "${REVISION_ARGS[@]}"

"$PYTHON" -m mixing_matters.cli sweep \
  --model "$MODEL_KEY" \
  --data "$DATA" \
  --output "$RUN_DIR/sweep.jsonl" \
  --positive-control "$RUN_DIR/positive-control.jsonl" \
  "${REVISION_ARGS[@]}"

echo "Phase 2 run complete for $MODEL_KEY: $RUN_DIR"
ls -l "$RUN_DIR"
