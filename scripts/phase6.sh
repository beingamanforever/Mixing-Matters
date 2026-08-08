#!/usr/bin/env bash
# Phase 6 end-to-end run for one model: capture the environment, then the
# RULER niah_single_1 depth sweep at every requested context length.
# Usage: scripts/phase6.sh <run-directory> <model-key> [num-samples] [lengths...]
# The run directory must not already exist. Raw outputs are never overwritten.
set -euo pipefail

RUN_DIR=${1:?usage: scripts/phase6.sh <run-directory> <model-key> [num-samples] [lengths...]}
MODEL_KEY=${2:?usage: scripts/phase6.sh <run-directory> <model-key> [num-samples] [lengths...]}
NUM_SAMPLES=${3:-100}
shift $(( $# < 3 ? $# : 3 ))
LENGTHS=("$@")
PYTHON=${PYTHON:-.venv/bin/python}

if [ -e "$RUN_DIR" ]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

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

LENGTH_ARGS=()
if [ "${#LENGTHS[@]}" -gt 0 ]; then
  LENGTH_ARGS=(--lengths "${LENGTHS[@]}")
fi

"$PYTHON" -m mixing_matters.cli ruler-sweep \
  --model "$MODEL_KEY" \
  --output "$RUN_DIR/sweep.jsonl" \
  --num-samples "$NUM_SAMPLES" \
  "${LENGTH_ARGS[@]}"

echo "Phase 6 run complete for $MODEL_KEY: $RUN_DIR"
ls -l "$RUN_DIR"
