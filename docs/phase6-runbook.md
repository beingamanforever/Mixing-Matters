# Phase 6 runbook

Phase 6 is the generality check.
It asks whether the accuracy-versus-evidence-position curve that Phase 2 measures on multi-document QA also shows up on a synthetic needle-in-a-haystack task, RULER `niah_single_1`.
If the two tasks disagree, that disagreement is the finding, not a failure of either pipeline.

## Task

`niah_single_1` hides one fact sentence, "One of the special magic numbers for <key> is: <value>.", inside repeated noise sentences, then asks for the magic number.
The task strings (noise haystack, numeric needle, word key, template, and the `string_match` scoring) are vendored verbatim in `src/mixing_matters/ruler.py` from NVIDIA RULER at commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.

Two deliberate departures from running RULER's own script preserve the study's position intervention and reproducibility rules:

- The needle is placed at ten deterministic depths, 0 through 9, mirroring the ten gold positions of the QA sweep, instead of RULER's random per-sample depth.
The noise sentence count is held fixed within a length, so total prompt length is invariant across the ten depths, exactly as gold position is length-invariant in Phase 2.
The runner enforces this: it records the per-instance prompt-length span across depths and raises if it exceeds a few tokens of byte-pair-merge noise.
- The key word is drawn from a small vendored word list rather than the `wonderwords` package.
The key is only a label the question echoes; it does not affect retrieval difficulty, and vendoring the list keeps instances reproducible without an external corpus.
The numeric needle and the noise haystack are exact.

## Models and lengths

The same three Phase 2 checkpoints run on one host at two lengths.
Relative to Phase 2, the task, prompt construction, scorer, host, and sample size also change, so this phase is a generality check rather than a single-variable control.

| Model | Family | Repo |
|---|---|---|
| `pythia-2.8b` | pythia | `EleutherAI/pythia-2.8b` |
| `mamba-2.8b` | mamba | `state-spaces/mamba-2.8b-hf` |
| `mamba2-2.7b` | mamba2 | `AntonV/mamba2-2.7b-hf` |

Run at 1K and 2K tokens.
Both are within the 2,048-token context of all three models; 2K sits just under Pythia's hard limit once the 32-token generation budget is reserved.
The length target counts the generation budget, following RULER, so the prompt is at most `length - 32` tokens.
The matched 8B Mamba-2 comparison at 4K belongs to Phase 3, not here.

## What the host needs

- One NVIDIA GPU with at least 16 GB of memory.
The study host for Phase 6 is a Tesla T4 with 15 GB.
- The runtime and CUDA kernels from `scripts/setup_gpu.sh`: torch 2.7.1+cu126, transformers 4.57.1, `mamba-ssm` 2.2.6.post3, `causal-conv1d` 1.5.3.post1, kernels compiled for the host's own compute capability.
- About 25 GB of free disk for the three checkpoints plus a few MB of output per model.
- A CUDA driver new enough for the torch build, and `nvcc` on `PATH` or under `/usr/local/cuda` for the kernel build.

### Tesla T4 notes

The T4 is compute capability 7.5 (Turing), the first host in this study below Ampere.
Two consequences, both recorded in every record's `software_versions`:

- bf16 has no dedicated tensor-core path on Turing, so bf16 matmuls run through the CUDA cores and are slower than on an A10G or L40S.
The study pins bf16 where supported and Turing supports it functionally, so the dtype is unchanged; only throughput differs.
- All three models run on this one host, so the within-phase task and length contrasts are on a single execution path.
Do not compare absolute T4 accuracy against the A10G Phase 2 numbers as if the host were held fixed; compare curves within Phase 6.

On Debian the base image may lack `python3.12-venv`; install it before setup:

```bash
apt-get update && apt-get install -y python3.12-venv build-essential
```

## Setup

```bash
git clone <this repository> mixing-matters
cd mixing-matters
bash scripts/setup_gpu.sh
```

`scripts/setup_gpu.sh` builds the venv, compiles the Mamba CUDA kernels for the host's compute capability, verifies that the kernel execution path the sweep requires is actually available, and prefetches the checkpoints.
It stops rather than continuing if the kernel path is unavailable or if the kernel install changed the torch version.

Confirm the harness passes its own tests on the host:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Run the sweeps

Each model runs the depth sweep at 1K and 2K: for every needle instance, a closed-book floor with no needle, an oracle ceiling with the needle and no noise, and the needle at depths 0 through 9.
With 100 instances that is `100 * 12 * 2 = 2,400` generations per model.

```bash
for model in pythia-2.8b mamba-2.8b mamba2-2.7b; do
  PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase6.sh runs/phase6/$model $model 100 1024 2048
done
```

Run them detached so an SSH drop cannot kill them:

```bash
setsid nohup bash -c '
for model in pythia-2.8b mamba-2.8b mamba2-2.7b; do
  PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase6.sh runs/phase6/$model $model 100 1024 2048 || echo "FAILED $model"
done
echo ALL_MODELS_DONE
' > runs/phase6.log 2>&1 < /dev/null &
```

Progress is the record count, since records are written as they are produced:

```bash
wc -l runs/phase6/*/sweep.jsonl
```

All three models must report `cuda_kernels` (Mamba) or `pytorch_reference` (Pythia) as their execution path, the same paths Phase 2 pinned.
The runner raises rather than falling back, so a run that starts has the path the study pinned.

## Bring the artifacts back and build the report

```bash
rsync -az <host>:mixing-matters/runs/phase6/ ./runs/phase6/
```

```bash
PYTHONPATH=src .venv/bin/python -m mixing_matters.cli phase6-report \
  --results runs/phase6/pythia-2.8b/sweep.jsonl \
            runs/phase6/mamba-2.8b/sweep.jsonl \
            runs/phase6/mamba2-2.7b/sweep.jsonl \
  --qa-results runs/phase2/pythia-2.8b/sweep.jsonl \
               runs/phase2/mamba-2.8b/sweep.jsonl \
               runs/phase2/mamba2-2.7b/sweep.jsonl \
  --output artifacts/phase6/report
```

That writes a position curve and an edge chart per length, a niah-versus-QA edge comparison per length when `--qa-results` is given, and `phase6-summary.json` holding every per-length curve, edge, pairwise interaction, and the cross-task comparison.

## Reading the result

The comparison of interest is between tasks, not against an absolute bar.
For each model, `task_comparison` places the niah primacy and recency edges beside the Phase 2 QA edges and flags whether they agree in sign.
A single-needle numeric retrieval is far easier than 20-document QA, so the ceiling anchor should be near 1.0 while the closed-book floor is near 0.0; a floor above chance would mean the number is guessable and the task is broken.
Report a task disagreement as a task disagreement.
