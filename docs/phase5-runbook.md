# Phase 5 runbook

Phase 5 is the training-data control.
It holds the Mamba architecture fixed and changes only the pretraining corpus, from the Pile to SlimPajama, and asks whether that alone moves the accuracy-versus-evidence-position curve.

## Models

| Model | Corpus | Format | Where |
|---|---|---|---|
| `mamba-2.8b` | the Pile | HF (`state-spaces/mamba-2.8b-hf`) | loaded directly |
| `mamba-2.8b-slimpj` | SlimPajama | original state-spaces (`state-spaces/mamba-2.8b-slimpj`) | converted first |

Both models are the same 2.8B Mamba architecture with 64 layers and no positional encoding.
The only registered difference is the training corpus, so the contrast isolates the data.

The Phase 2 `mamba-2.8b` sweep in `artifacts/phase2/mamba-2.8b/` was produced on an A40.
Phase 5 re-runs the Pile arm on the same host as the SlimPajama arm so the corpus is the only variable that changes between the two curves, and the Phase 2 numbers become a reproducibility check rather than a term in the contrast.

## Why the SlimPajama checkpoint is converted

`state-spaces/mamba-2.8b-slimpj` is published only in the original state-spaces format.
Its config.json carries `d_model` and `n_layer` and no `model_type`, so `AutoModelForCausalLM` cannot load it, exactly like the Mamba-2 checkpoint in Phase 2.

Loading it through the authors' `mamba_ssm` runtime instead of transformers would change the execution path, which is a second variable.
To avoid that, the checkpoint is converted once to an HF-format `MambaForCausalLM` directory and loaded through the same transformers CUDA-kernel path as every other Mamba run.
The conversion follows the mapping transformers ships: the config is rebuilt from the state-spaces fields and the only weight rename is `backbone.embedding` to `backbone.embeddings`, with the LM head tied to the embedding.

## What the host needs

- One NVIDIA GPU with at least 16 GB of memory. The study host is an A10G with 23 GB.
- The runtime and CUDA kernels from `scripts/setup_gpu.sh`: torch 2.7.1+cu126, transformers 4.57.1, `mamba-ssm` 2.2.6.post3, `causal-conv1d` 1.5.3.post1, kernels compiled for the host's own compute capability.
- Disk for three checkpoints: about 11 GB for `mamba-2.8b-hf`, about 22 GB for the original SlimPajama checkpoint in fp32, and about 5 GB for its bf16 HF conversion, plus roughly 150 MB of output per model.
- Around 10 hours of wall clock on an A10G for both sweeps.

If the root disk is small, put the Hugging Face cache and the conversion on a roomier volume:

```bash
export HF_HOME=/large/volume/hf
export MIXING_MATTERS_CONVERTED_DIR=/large/volume/converted
```

## Setup

```bash
bash scripts/setup_gpu.sh
PYTHONPATH=src .venv/bin/python -m mixing_matters.cli download
PYTHONPATH=src .venv/bin/python -m pytest -q
```

`scripts/setup_gpu.sh` prefetches the HF checkpoints at their pinned revisions; the SlimPajama weights are fetched by the conversion step below because they are not in HF format.

## Convert and validate the SlimPajama checkpoint

```bash
PYTHONPATH=src .venv/bin/python scripts/convert_mamba_slimpj.py
PYTHONPATH=src .venv/bin/python scripts/validate_mamba_slimpj_conversion.py
```

The first command downloads the original checkpoint at its pinned revision, converts it, and writes the HF directory plus a `conversion-manifest.json` recording the source revision and the converted-weight checksums.
The second command compares the conversion against the original weights run through `mamba_ssm`, on greedy generations and next-token logits.
Record what it prints and compare it against the Phase 2 Mamba-2 numbers in `docs/phase2-runbook.md`: 4 of 5 byte-identical greedy generations, top-1 agreement on 4 of 5, and maximum absolute logit differences of 0.25 to 0.50 against a logit scale of 36 to 52.
Differences of that size are expected because the two implementations order tensor contractions differently and bfloat16 makes that visible.

## Run the sweeps

Each model runs its own key-value control, which never gates the sweep, and then the ten-position sweep plus the closed-book floor and oracle ceiling: 800 questions times twelve conditions, 9,600 generations.

```bash
setsid nohup bash -c '
for model in mamba-2.8b mamba-2.8b-slimpj; do
  PYTHON=.venv/bin/python PYTHONPATH=src bash scripts/phase2.sh runs/phase5/$model $model || echo "FAILED $model"
done
echo ALL_MODELS_DONE
' > runs/phase5.log 2>&1 < /dev/null &
```

Progress is the record count, since records are written as they are produced:

```bash
wc -l runs/phase5/*/sweep.jsonl
```

Both models must report `cuda_kernels` as their execution path.
The runner raises rather than falling back to the reference path, so a run that starts has the path the study pinned.
The SlimPajama records carry `training_corpus`, `checkpoint_format`, and `checkpoint_dir` in their metadata, and their `model_revision` is the pinned source revision from the conversion manifest.

## Bring the artifacts back and build the report

```bash
rsync -az <host>:mixing-matters/repo/runs/phase5/ ./runs/phase5/
```

Then build the data-control report, passing the Phase 2 Pythia and Pile-Mamba sweeps so the corpus effect is placed beside the architecture effect:

```bash
uv run python -m mixing_matters.cli phase5-report \
  --results runs/phase5/mamba-2.8b/sweep.jsonl \
            runs/phase5/mamba-2.8b-slimpj/sweep.jsonl \
  --architecture-results artifacts/phase2/pythia-2.8b/sweep.jsonl \
                         artifacts/phase2/mamba-2.8b/sweep.jsonl \
  --output artifacts/phase5/report
```

That writes the two corpus position curves on one axis, the primacy and recency edges per corpus, the corpus edge effect beside the Phase 2 architecture edge effect, and a summary with the Pile-minus-SlimPajama interaction and its Holm-corrected p-values.

## What Phase 5 can and cannot claim

Phase 5 changes the corpus with the architecture, tokenizer, parameter count, and depth all fixed, so a curve difference is attributable to the training data.
A null result is also informative: it means that within this architecture the corpus did not move the curve at the sample size run.
It does not isolate any single property of SlimPajama versus the Pile, only that the corpora differ.
