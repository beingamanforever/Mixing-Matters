# Phase 3 runbook

Phase 3 is a matched pure-versus-hybrid architecture comparison.
It asks whether the published pure and hybrid NVIDIA 8B checkpoints have different accuracy-versus-evidence-position curves.

## Models

| Model | Attention | Format | Where |
|---|---|---|---|
| `mamba2-8b` | none (pure Mamba-2) | Megatron-LM (`nvidia/mamba2-8b-3t-4k`) | Megatron backend |
| `mamba2-hybrid-8b` | ~7% of layers | Megatron-LM (`nvidia/mamba2-hybrid-8b-3t-4k`) | Megatron backend |

Both are NVIDIA 8B Mamba-2 models trained on the same 3.5T-token corpus with the same 256k tokenizer and the same 56-layer depth.
The pure configuration uses attention and MLP ratios `0.0 / 0.0`, while the hybrid uses `0.08 / 0.5`.
The comparison therefore changes both attention and MLP composition and must be interpreted as a composite pure-versus-hybrid architecture contrast, not an attention-only intervention.

## Why both models run through Megatron-LM

Both checkpoints are published only as Megatron-LM distributed checkpoints (`release/mp_rank_00/model_optim_rng.pt`, `latest_checkpointed_iteration.txt`, and a 256k SentencePiece tokenizer).
Their configs carry no `model_type` and no HF modeling class, so `AutoModelForCausalLM` cannot load them, and the hybrid has no HF modeling class at all.

Running the two models through two different runtimes would add an execution-path difference to the architecture contrast.
Both models therefore run through NVIDIA's own Megatron-LM `MambaModel` on one shared CUDA-kernel path.
The repository's `run.run_sweep` and `run.run_kv_control` accept an injected generator, so the Megatron backend drives the exact same prompt construction, floor and ceiling anchors, length-invariance check, scoring, and JSONL record schema as every other phase; only the function that turns a prompt into a greedy generation changes.

A community transformers port of the pure model exists, but it targets transformers 5 while the `mamba-ssm` CUDA kernels it needs import symbols that transformers 5 removed, so that path cannot load the fast kernels without a silent fallback to a numerically different reference path.
The Megatron path avoids that conflict and keeps both Phase 3 models on one validated runtime.

## Running it bare-metal, without a container

NVIDIA's own reference stack for these checkpoints is the `nvcr.io/nvidia/pytorch:24.01-py3` container plus Megatron-LM at commit `df61e60`.
That container was not available on the study's host, so this run reproduces the same trio of pinned packages (`mamba-ssm` 2.0.3, `causal-conv1d` 1.2.2.post1, the container's torch 2.2 / triton 2.1 pair) from prebuilt wheels in a plain venv instead of a container.

Two things the container's Python 3.10 does not need come up on Python 3.12:

- Megatron's `megatron/core/jit.py` sets its fuser to `torch.compile` for torch 2.2+, and Transformer Engine's own jit module does the same. TorchDynamo does not support Python 3.12 on torch 2.2. The fuser is only an elementwise-fusion optimization the SSM math never depends on because that runs in the `mamba-ssm` CUDA kernels. `scripts/setup_baremetal_megatron.sh` installs a venv startup shim through a `.pth` file that makes `torch.compile` a no-op on Python 3.12 before either package imports, leaving the pinned Megatron checkout unchanged. `scripts/run_phase3_baremetal.sh` also sets `NVTE_TORCH_COMPILE=0` and `NVTE_FLASH_ATTN=0` so Transformer Engine does not use its compile or flash-attention paths.
- `megatron.training` imports NVIDIA `apex` at module load, for fused optimizers and fused-norm training kernels. None of those run during greedy inference, which uses Transformer Engine's norms and never builds an optimizer, so the setup script writes a minimal `apex` stub that only satisfies the imports and raises if anything actually calls the training-only functions. Real apex needs a heavy source build that is unnecessary for this.

Transformer Engine itself is required: the mamba layer spec (`megatron.core.models.mamba.mamba_layer_specs.mamba_stack_spec`) hard-codes `TENorm`, `TELayerNormColumnParallelLinear`, `TEDotProductAttention`, and `TERowParallelLinear`, so swapping in a non-TE spec would change the checkpoint's weight names and break loading. Its Python core has a prebuilt wheel; its torch bindings do not, so those compile from source, capped to a few parallel jobs so the build cannot exhaust the host's memory.

An NGC container with docker and `nvidia-container-toolkit` remains a valid, arguably simpler way to run this if the host supports it: `scripts/run_phase3_in_container.sh` documents that path, unchanged from the original design. On the host used for this study, building the mamba-ssm kernels inside a container repeatedly made the host briefly unreachable over SSH, so this run used the bare-metal path throughout instead.

## What the host needs

- One NVIDIA GPU with at least 24 GB of memory. The study host is an A40 with 46 GB. Tensor-parallel size 1 fits a single card.
- Python 3.12, a C++ toolchain, and the CUDA 12.x toolkit (`nvcc` on `PATH` or under `/usr/local/cuda`).
- Megatron-LM checked out at commit `df61e60` (the "Add mamba" commit).
- Disk for two checkpoints: about 16 GB each, plus roughly 150 MB of output per model.
- Around 8 to 12 hours of wall clock on an A40 for both models' key-value control plus the ten-position sweep; the sweeps ran at roughly 1.8 seconds per generation on this host.

## Setup

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && git -C Megatron-LM checkout df61e60
VENV=/root/mm-venv MEGATRON=/root/Megatron-LM bash scripts/setup_baremetal_megatron.sh
```

This provisions the venv, installs the pinned prebuilt wheels, compiles the Transformer Engine torch bindings, writes the apex stub and Python 3.12 compatibility shim, and verifies the full import chain, including that the mamba stack spec resolves.

Download the two checkpoints (both are ungated) into per-model directories:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('nvidia/mamba2-8b-3t-4k', revision='b915550c63ba9359f88f44d1f6a600d85af27302', local_dir='checkpoints/mamba2-8b')"
python -c "from huggingface_hub import snapshot_download as s; s('nvidia/mamba2-hybrid-8b-3t-4k', revision='35e8852e2240b350ac2fe2a3b8aa341b5930018e', local_dir='checkpoints/mamba2-hybrid-8b')"
```

Download and verify the pinned Lost-in-the-Middle dataset before starting the run:

```bash
PYTHONPATH=src /root/mm-venv/bin/python -m mixing_matters.cli download
echo "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9  data/nq-open-10_total_documents_gold_at_0.jsonl.gz" | sha256sum --check -
```

The command must report `OK` before any Phase 3 stage runs.

Stage the PIQA validation set for the benchmark gate:

```bash
curl -fsSL -o valid.jsonl https://yonatanbisk.com/piqa/data/valid.jsonl
curl -fsSL -o valid-labels.lst https://yonatanbisk.com/piqa/data/valid-labels.lst
echo "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d  valid.jsonl" | sha256sum --check -
echo "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb  valid-labels.lst" | sha256sum --check -
python scripts/prepare_piqa.py \
  --examples valid.jsonl \
  --labels valid-labels.lst \
  --output data/piqa_valid.jsonl
echo "61533005e22f175534909b1ec8eacb6da03c233933558d1bafb15787453b1f55  data/piqa_valid.jsonl" | sha256sum --check -
```

The preparation command requires exactly 1,838 matching examples and binary labels, writes one JSON object per line with `goal`, `sol1`, `sol2`, and `label`, and refuses to overwrite an existing output.

`scripts/run_phase3_baremetal.sh <model_key> <stage>` runs one stage with the Megatron model arguments for that model (`--hybrid-attention-ratio 0.0 --hybrid-mlp-ratio 0.0` for the pure model, `0.08 / 0.5` for the hybrid).
`scripts/phase3.sh <run-directory>` is the required clean end-to-end entry point.
It refuses dirty or unpinned repository inputs, verifies the dataset, PIQA, exact checkpoint bytes, and checkpoint revisions, captures the environment, runs both validation gates before inference, and writes every output exclusively below a new run directory.
It fixes PIQA at 1,838 examples and the exploratory sweep at 800 questions, requires the repository launcher, records its SHA-256, and validates each model's 500-record key-value control and 9,600-record sweep before continuing.

## Run the clean validation and sweeps

```bash
setsid env \
  MIXING="$PWD" \
  MEGATRON=/root/Megatron-LM \
  VENV=/root/mm-venv/bin/python \
  CKPT_ROOT=/root/checkpoints \
  bash scripts/phase3.sh /root/outputs/phase3-clean \
  > /root/phase3-clean.log 2>&1 < /dev/null &
```

The validate stage prints greedy completions on a handful of factual prompts for a human to eyeball, then scores PIQA validation by continuation log-likelihood.
Waleffe et al. 2024 (arXiv:2406.07887) report PIQA 79.82 for the pure 8B Mamba-2 (Table 3) and 79.65 for the 8B Mamba-2-Hybrid (Table 7), both at 3.5T tokens; the gate is +/- 1.0 point on the full 1838-example set, checked against the number for the model under test.
A garbled checkpoint load, a wrong layer pattern, or a wrong tokenizer shows up as nonsense completions and a PIQA far from the target.
The orchestrator stops unless both PIQA gates report `PASS` before starting either key-value control or QA sweep.

This run measured PIQA 79.27 for the pure model (delta -0.55, pass) and PIQA 79.65 for the hybrid (delta +0.00, pass).

Each model runs its own key-value control, which never gates the sweep, and then the ten-position sweep plus the closed-book floor and oracle ceiling: 800 questions times twelve conditions.

Records are written as they are produced, so progress is the record count in `/root/outputs/phase3-clean/<model>-sweep.jsonl`.
Every record carries the pinned model revision and `execution_path="megatron_cuda_kernels"`.
The launcher fixes the model-specific attention and MLP ratios, prints the resolved layer pattern to the stage log, and the environment manifest records the pinned checkpoint files and launcher.
Both sweeps in this run completed with zero excluded questions and zero scoring failures.

## Bring the artifacts back and build the report

```bash
rsync -az <host>:outputs/ ./runs/phase3/
mixing-matters phase3-report \
  --results runs/phase3/mamba2-8b-sweep.jsonl runs/phase3/mamba2-hybrid-8b-sweep.jsonl \
  --output artifacts/phase3/report
```

That writes both position curves on one axis, the primacy and recency edges per model, the hybrid-minus-pure architecture contrast on each edge, and a summary with the paired-bootstrap contrast and its Holm-corrected p-values.

## What Phase 3 can and cannot claim

Phase 3 compares two matched published checkpoints, but the configured attention and MLP ratios both change.
A curve difference is therefore associated with the composite pure-versus-hybrid architecture change and cannot be attributed to attention alone.
A null result means that this composite architecture contrast did not measurably move the curve at the sample size run.
Because both Phase 3 models run on the Megatron path rather than the transformers path the Phase 2 and Phase 5 Mamba models use, the Phase 3 contrast is read on its own and is not placed beside another phase's contrast.
