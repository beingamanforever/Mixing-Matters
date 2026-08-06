# Phase 2 runbook

How to run the two Mamba sweeps on a fresh GPU host and bring the artifacts back.
Pythia is already complete and its artifacts are in `artifacts/phase2/pythia-2.8b/`.

## Status

| Model | State | Where |
|---|---|---|
| `pythia-2.8b` | completed, 9,600 records | `artifacts/phase2/pythia-2.8b/` |
| `mamba-2.8b` | not run | this runbook |
| `mamba2-2.7b` | not run | this runbook |

The A10G run of `mamba-2.8b` was stopped during its key-value control at 444 of 500 records.
That partial file was renamed to `positive-control.partial-444.jsonl` on that host and is not used.
Nothing was overwritten.

## What the host needs

- One NVIDIA GPU with at least 16 GB of memory. An L40S with 48 GB is ample.
- A CUDA driver new enough for the torch build below, and `nvcc` on `PATH` for the kernel build.
- Python 3.12.
- About 25 GB of free disk: 11.1 GB for `mamba-2.8b-hf`, 5.4 GB for `mamba2-2.7b-hf`, 5.7 GB for Pythia, and roughly 150 MB of output per model.
- Around 10 hours of wall clock on an A10G for both models, or an estimated 6 to 7 hours on an L40S.

## Setup

```bash
git clone <this repository> mixing-matters
cd mixing-matters
bash scripts/setup_gpu.sh
```

`scripts/setup_gpu.sh` builds a venv with the pinned runtime, compiles the Mamba CUDA kernels from source for the host's own compute capability, verifies that the kernel execution path the sweep requires is actually available, and prefetches all three checkpoints at their pinned revisions.
It stops with a clear error rather than continuing if the kernel path is unavailable or if the kernel install changed the torch version.

Two failure modes it exists to prevent, both encountered on the A10G host:

- Installing the kernel packages without `--no-deps` lets pip resolve torch and upgrade it. That replaced a working torch 2.7.1+cu126 with torch 2.13.0+cu130, which the driver could not run.
- The prebuilt kernel wheels advertised for cu12 and torch 2.7 are compiled against a CUDA 12.9 build of torch. Against a cu126 build they fail to load with `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`. Compiling against the installed torch avoids this.

Then fetch the dataset, which is checksum pinned:

```bash
PYTHONPATH=src .venv/bin/python -m mixing_matters.cli download
```

Confirm the harness passes its own tests on the host:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Validate the Mamba-2 conversion first

`state-spaces/mamba2-2.7b` cannot be loaded by transformers 4.57.1, because its config carries no `model_type`.
Phase 2 therefore loads `AntonV/mamba2-2.7b-hf`, a community conversion with no published numerical validation.
Check it against the original weights before trusting its numbers:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mamba2_conversion.py
```

On the A10G this produced 4 of 5 byte-identical greedy generations, top-1 next-token agreement on 4 of 5, and maximum absolute logit differences of 0.25 to 0.50 against a logit scale of 36 to 52.
Differences of that size are expected, since the two implementations order tensor contractions differently and bfloat16 makes that visible.
Record whatever the new host produces and compare it against those numbers.

## Run the sweeps

Each model runs its own key-value control and then the sweep: 800 questions, the gold document at every position 0 through 9, plus the closed-book floor and the oracle ceiling, for 9,600 generations.

```bash
PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase2.sh runs/phase2/mamba-2.8b mamba-2.8b
PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase2.sh runs/phase2/mamba2-2.7b mamba2-2.7b
```

Run them detached so an SSH drop cannot kill them:

```bash
setsid nohup bash -c '
for model in mamba-2.8b mamba2-2.7b; do
  PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase2.sh runs/phase2/$model $model || echo "FAILED $model"
done
echo ALL_MODELS_DONE
' > runs/phase2.log 2>&1 < /dev/null &
```

Progress is the record count, since records are written as they are produced:

```bash
wc -l runs/phase2/*/sweep.jsonl
```

Both models must report `cuda_kernels` as their execution path.
The runner raises rather than falling back to the reference path, so a run that starts has the path the study pinned.
The key-value control never gates the sweep: a model that cannot do key-value retrieval is a finding about that model, and Phase 1 already established that the pipeline detects the effect.

## Bring the artifacts back

```bash
rsync -az <host>:mixing-matters/runs/phase2/ ./runs/phase2/
```

Then produce the combined report across all three models:

```bash
uv run python -m mixing_matters.cli phase2-report \
  --results runs/phase2/pythia-2.8b/sweep.jsonl \
            runs/phase2/mamba-2.8b/sweep.jsonl \
            runs/phase2/mamba2-2.7b/sweep.jsonl \
  --output artifacts/phase2/report
```

That writes the three position curves on one axis, the primacy and recency edges per model, and a summary containing every pairwise interaction with its Holm-corrected p-value.

## Expectations to check against

Reference numbers from the A10G host, useful for spotting a misconfigured run:

- Pythia accuracy runs 0.266 at position 0, 0.180 at positions 4 and 5, and 0.279 at position 9, with floor 0.092 and ceiling 0.641.
- Prompt lengths across the 800 questions fall between 1,278 and 1,881 tokens. All three models share the GPT-NeoX-20B tokenizer and produce identical token counts for the same prompt.
- 775 questions hold prompt length constant across positions and 25 shift by exactly one token, from byte-pair merges at document boundaries. The runner allows a span of at most 2 tokens and records the observed span per record.
- Steady-state generation cost on the A10G was about 1.2 s for Pythia, 1.9 s for `mamba-2.8b`, and 2.05 s for `mamba2-2.7b` on a roughly 1,450-token prompt.
- Neither Mamba config exposes `max_position_embeddings`, so the recorded context limit is null for both and the overflow guard is skipped. Pythia keeps its real 2,048 limit.
