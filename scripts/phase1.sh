#!/usr/bin/env bash
# Phase 1 end-to-end run: key-value positive control, tracer, analysis, environment capture.
# Usage: scripts/phase1.sh <run-directory> [model-revision]
# The run directory must not already exist. Raw outputs are never overwritten.
set -euo pipefail

RUN_DIR=${1:?usage: scripts/phase1.sh <run-directory> [model-revision]}
REVISION=${2:-2a259cdd96a4beb1cdf467512e3904197345f6a9}
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

"$PYTHON" -m mixing_matters.cli positive-control \
  --output "$RUN_DIR/positive-control.jsonl" --revision "$REVISION"

"$PYTHON" -m mixing_matters.cli run \
  --data "$DATA" \
  --output "$RUN_DIR/tracer.jsonl" \
  --revision "$REVISION" \
  --positive-control "$RUN_DIR/positive-control.jsonl"

"$PYTHON" -m mixing_matters.cli analyze "$RUN_DIR/tracer.jsonl" > "$RUN_DIR/summary.json"

"$PYTHON" -m mixing_matters.cli figures \
  --kv "$RUN_DIR/positive-control.jsonl" \
  --phase1 "$RUN_DIR/tracer.jsonl" \
  --output "$RUN_DIR/figures"

"$PYTHON" -m mixing_matters.cli audit-sample \
  --results "$RUN_DIR/tracer.jsonl" \
  --output "$RUN_DIR/audit"

echo "Phase 1 run complete: $RUN_DIR"
ls -l "$RUN_DIR"
