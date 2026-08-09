# Phase 8 runbook

Phase 8 is the descriptive system comparison.
It places three complete production 7-8B systems next to each other on the Lost-in-the-Middle 10-document multi-document QA task.
Unlike Phase 2 through Phase 5, this phase is not a matched control: the three systems differ from each other on many axes at once - architecture, pretraining corpus, token count, tokenizer, alignment status, depth, and positional encoding - so the sweep describes how full-system curves look side by side, not which single variable is responsible for any curve difference.

## Systems

| Model key | Family | Repo | Notes |
|---|---|---|---|
| `nemotron-h-8b` | nemotron-h | `nvidia/Nemotron-H-8B-Base-8K` | Hybrid Mamba-2 plus attention. 8K context. Public. |
| `llama-3.1-8b` | llama | `meta-llama/Llama-3.1-8B` | Dense attention with RoPE. 128K context. Gated; requires an HF token with access. |
| `qwen2.5-7b` | qwen2 | `Qwen/Qwen2.5-7B` | Dense attention with RoPE. 32K context. Public. |

All three checkpoints load through the standard transformers backend.
The execution path is resolved by family in `src/mixing_matters/run.py`:

- `llama` and `qwen2` run under the eager attention implementation to keep them on one deterministic pytorch reference path, mirroring Pythia.
- `nemotron-h` also pins the eager attention implementation, and its SSM blocks dispatch through `mamba-ssm` and `causal-conv1d`; the runner refuses to fall back to the SSM reference path.

The Phase 8 sweep raises the per-question prompt-token span tolerance from `MAX_PROMPT_TOKEN_SPAN=2` (tight enough for the GPTNeoX and Mamba tokenizers used in Phases 2 through 6) to 8, because the Llama, Qwen, and Nemotron-H tokenizers each have their own byte-pair merges at document boundaries.
The tolerance is a CLI knob, `--max-prompt-token-span`, so any Phase 8 run records the value it used in the raw JSONL.

## Data and scoring

The task is unchanged from Phase 2: the released Lost-in-the-Middle 10-document set, 800 exploratory questions per model, ten deterministic gold positions per question, plus a closed-book floor and an oracle ceiling.
The scoring code is the vendored `best_subspan_em` from the Liu et al. repository, unchanged; the two sensitivity variants (normalized exact match and first-line extraction) are also computed.
Every raw record is written as JSONL with the pinned model revision, tokenizer, execution path, driver, and CUDA information already recorded in Phase 2 and later.

## What the host needs

- One NVIDIA GPU with at least 32 GB of memory.
The reference host is a rented L40S 46 GB, sm_89.
BF16 weights for a single 7-8B model take about 15 GB; the peak with a 2K prompt fits comfortably below 30 GB.
- The runtime and CUDA kernels from `scripts/setup_gpu.sh`, unchanged from Phase 2 and Phase 6: torch 2.7.1+cu126, transformers 4.57.1, `mamba-ssm` 2.2.6.post3, `causal-conv1d` 1.5.3.post1, kernels compiled for the host's own compute capability.
- About 60 GB of free disk for the three checkpoints.
- A CUDA driver new enough for the torch build, and `nvcc` on `PATH` or under `/usr/local/cuda` for the kernel build.
- A Hugging Face access token with approved access to `meta-llama/Llama-3.1-8B`, exported as `HF_TOKEN` before the prefetch.

## Setup

```bash
apt-get update && apt-get install -y python3.12-venv build-essential
git clone <this repository> mixing-matters
cd mixing-matters
git checkout phase-8

export HF_TOKEN=<your-token>
huggingface-cli login --token "$HF_TOKEN" >/dev/null

bash scripts/setup_gpu.sh
```

`scripts/setup_gpu.sh` builds the venv, compiles the Mamba CUDA kernels for the host's compute capability, verifies that the kernel execution path the sweep requires is actually available, and prefetches every registered checkpoint (including the three Phase 8 systems).
The gated Llama-3.1 download uses the `HF_TOKEN` set above.
The kernel path check is the same one Phase 6 uses; it stops rather than continuing if the path is unavailable.

Confirm the harness passes its own tests on the host:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Run the sweeps

Each model runs the ten-position gold sweep with closed-book and oracle anchors for 800 exploratory questions: `800 * 12 = 9,600` generations per model, plus 500 key-value positive-control generations.

Detached, so an SSH drop cannot kill the run:

```bash
setsid nohup bash -c '
for model in nemotron-h-8b qwen2.5-7b llama-3.1-8b; do
  PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase8.sh runs/phase8/$model $model \
    || echo "FAILED $model"
done
echo ALL_MODELS_DONE
' > runs/phase8.log 2>&1 < /dev/null &
```

Progress is the record count, since records are written as they are produced:

```bash
wc -l runs/phase8/*/sweep.jsonl
```

The nemotron-h sweep must report `cuda_kernels` as its execution path; llama and qwen must report `pytorch_reference` with `eager` attention.
The runner raises rather than falling back, so a run that starts has the path this phase pinned.

## Bring the artifacts back and build the report

```bash
rsync -az <host>:mixing-matters/runs/phase8/ ./runs/phase8/
```

```bash
PYTHONPATH=src .venv/bin/python -m mixing_matters.cli phase8-report \
  --results runs/phase8/nemotron-h-8b/sweep.jsonl \
            runs/phase8/llama-3.1-8b/sweep.jsonl \
            runs/phase8/qwen2.5-7b/sweep.jsonl \
  --output artifacts/phase8/report
```

That writes `position-curves.png`, `position-edges.png`, and `phase8-summary.json` holding per-model curves, primacy and recency edges, every pairwise interaction, floor and ceiling means, and a small system descriptor block naming the family and repo.

## Reading the result

Phase 8 answers no matched-control question.
Every pairwise interaction jointly reflects architecture, corpus, token count, tokenizer, alignment status, depth, and positional encoding.
The intended use is to place the three systems side by side and describe how their curves differ; single-variable claims belong to Phase 2, Phase 3, and Phase 5.
