# Phase 3 runbook

Phase 3 is the architecture control.
It holds the training data, tokenizer, parameter count, depth, and positional-encoding setup fixed and changes only whether attention layers are present, then asks whether that alone moves the accuracy-versus-evidence-position curve.

## Models

| Model | Attention | Format | Where |
|---|---|---|---|
| `mamba2-8b` | none (pure Mamba-2) | Megatron-LM (`nvidia/mamba2-8b-3t-4k`) | Megatron backend |
| `mamba2-hybrid-8b` | ~7% of layers | Megatron-LM (`nvidia/mamba2-hybrid-8b-3t-4k`) | Megatron backend |

Both are NVIDIA 8B Mamba-2 models trained on the same 3.5T-token corpus with the same 256k tokenizer, the same 56-layer depth, and no positional encoding.
The only registered difference is that the hybrid replaces roughly seven percent of its Mamba-2 layers with self-attention layers, so the contrast isolates the attention layers.

## Why both models run through Megatron-LM

Both checkpoints are published only as Megatron-LM distributed checkpoints (`release/mp_rank_00/model_optim_rng.pt`, `latest_checkpointed_iteration.txt`, and a 256k SentencePiece tokenizer).
Their configs carry no `model_type` and no HF modeling class, so `AutoModelForCausalLM` cannot load them, and the hybrid has no HF modeling class at all.

Running the two models through two different runtimes would change the execution path between them, which is a second variable on top of the attention layers.
To avoid that, both models run through NVIDIA's own Megatron-LM `MambaModel`, on one shared CUDA-kernel path, so the only thing that moves between the two curves is the presence of attention layers.
The repository's `run.run_sweep` and `run.run_kv_control` accept an injected generator, so the Megatron backend drives the exact same prompt construction, floor and ceiling anchors, length-invariance check, scoring, and JSONL record schema as every other phase; only the function that turns a prompt into a greedy generation changes.

A community transformers port of the pure model exists, but it targets transformers 5 while the `mamba-ssm` CUDA kernels it needs import symbols that transformers 5 removed, so that path cannot load the fast kernels without a silent fallback to a numerically different reference path.
The Megatron path avoids that conflict and keeps both Phase 3 models on one validated runtime.

## Running it bare-metal, without a container

NVIDIA's own reference stack for these checkpoints is the `nvcr.io/nvidia/pytorch:24.01-py3` container plus Megatron-LM at commit `df61e60`.
That container was not available on the study's host, so this run reproduces the same trio of pinned packages (`mamba-ssm` 2.0.3, `causal-conv1d` 1.2.2.post1, the container's torch 2.2 / triton 2.1 pair) from prebuilt wheels in a plain venv instead of a container.

Two things the container's Python 3.10 does not need come up on Python 3.12:

- Megatron's `megatron/core/jit.py` sets its fuser to `torch.compile` for torch 2.2+, and Transformer Engine's own jit module does the same. TorchDynamo does not support Python 3.12 on torch 2.2. The fuser is only an elementwise-fusion optimization the SSM math never depends on (that runs in the `mamba-ssm` CUDA kernels either way), so `scripts/setup_baremetal_megatron.sh` patches Megatron's `jit.py` to no-op on 3.12, and `scripts/run_phase3_baremetal.sh` sets `NVTE_TORCH_COMPILE=0` and `NVTE_FLASH_ATTN=0` so Transformer Engine's own jit path and its flash-attention backend do not hit the same call.
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

This provisions the venv, installs the pinned prebuilt wheels, compiles the Transformer Engine torch bindings, writes the apex stub, patches Megatron's `jit.py`, and verifies the full import chain, including that the mamba stack spec resolves.

Download the two checkpoints (both are ungated) into per-model directories:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('nvidia/mamba2-8b-3t-4k', local_dir='checkpoints/mamba2-8b')"
python -c "from huggingface_hub import snapshot_download as s; s('nvidia/mamba2-hybrid-8b-3t-4k', local_dir='checkpoints/mamba2-hybrid-8b')"
```

Stage the PIQA validation set for the benchmark gate, one JSON object per line with `goal`, `sol1`, `sol2`, and `label`:

```bash
curl -fsSL -o valid.jsonl https://yonatanbisk.com/piqa/data/valid.jsonl
curl -fsSL -o valid-labels.lst https://yonatanbisk.com/piqa/data/valid-labels.lst
# merge into data/piqa_valid.jsonl with the label field, 1838 examples
```

`scripts/run_phase3_baremetal.sh <model_key> <stage>` runs one stage with the Megatron model arguments for that model (`--hybrid-attention-ratio 0.0 --hybrid-mlp-ratio 0.0` for the pure model, `0.08 / 0.5` for the hybrid).

## Validate each checkpoint before its sweep

```bash
VENV=/root/mm-venv/bin/python bash scripts/run_phase3_baremetal.sh mamba2-8b validate
VENV=/root/mm-venv/bin/python bash scripts/run_phase3_baremetal.sh mamba2-hybrid-8b validate
```

The validate stage prints greedy completions on a handful of factual prompts for a human to eyeball, then scores PIQA validation by continuation log-likelihood.
Waleffe et al. 2024 (arXiv:2406.07887) report PIQA 79.82 for the pure 8B Mamba-2 (Table 3) and 79.65 for the 8B Mamba-2-Hybrid (Table 7), both at 3.5T tokens; the gate is +/- 1.0 point on the full 1838-example set, checked against the number for the model under test.
A garbled checkpoint load, a wrong layer pattern, or a wrong tokenizer shows up as nonsense completions and a PIQA far from the target.

This run measured PIQA 79.27 for the pure model (delta -0.55, pass) and PIQA 79.65 for the hybrid (delta +0.00, pass).

## Run the sweeps

Each model runs its own key-value control, which never gates the sweep, and then the ten-position sweep plus the closed-book floor and oracle ceiling: 800 questions times twelve conditions.

```bash
for model in mamba2-8b mamba2-hybrid-8b; do
  VENV=/root/mm-venv/bin/python bash scripts/run_phase3_baremetal.sh "$model" kv
  VENV=/root/mm-venv/bin/python bash scripts/run_phase3_baremetal.sh "$model" sweep
done
```

Records are written as they are produced, so progress is the record count in `outputs/<model>-sweep.jsonl`.
Every record carries `execution_path="megatron_cuda_kernels"`, the `hybrid_attention_ratio` and `hybrid_mlp_ratio` used, the resolved layer pattern, the checkpoint format and directory, and the pinned model revision.
Both sweeps in this run completed with zero excluded questions and zero scoring failures.

## Bring the artifacts back and build the report

```bash
rsync -az <host>:outputs/ ./runs/phase3/
mixing-matters phase3-report \
  --results runs/phase3/mamba2-8b-sweep.jsonl runs/phase3/mamba2-hybrid-8b-sweep.jsonl \
  --output artifacts/phase3/report
```

That writes both position curves on one axis, the primacy and recency edges per model, the hybrid-minus-pure attention effect on each edge, and a summary with the paired-bootstrap contrast and its Holm-corrected p-values.

## What Phase 3 can and cannot claim

Phase 3 changes only the presence of attention layers, with the training data, tokenizer, parameter count, depth, and positional encoding all fixed, so a curve difference is attributable to the attention layers.
A null result is also informative: it means that adding this fraction of attention did not move the curve at the sample size run.
Because both Phase 3 models run on the Megatron path rather than the transformers path the Phase 2 and Phase 5 Mamba models use, the Phase 3 contrast is read on its own and is not placed beside another phase's contrast.
