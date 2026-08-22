# Phase 3 artifacts

## Status

Completed. Both models run over the full 800-question exploratory set.

## Why one host, one execution path

Phase 3 compares released pure and hybrid Mamba-2 checkpoints while holding their corpus, tokenizer, approximate scale, depth, positional setup, host, and execution path fixed.
The checkpoints differ in both attention and MLP composition, so this is a composite pure-versus-hybrid contrast rather than an attention-only intervention.
Both checkpoints are Megatron-LM-only (see `docs/phase3-runbook.md`), so both run on one NVIDIA A40 through NVIDIA's own Megatron-LM `MambaModel`, bare-metal (no container; the mamba-ssm kernel build repeatedly made the host briefly unreachable inside a container).

## Experimental setup

Each model runs 9,600 generations: 800 exploratory questions, the gold document placed at every position 0 through 9, plus the closed-book floor and the oracle ceiling for each question. The question set is identical across models, which is what allows the model comparison to be paired.

Decoding is greedy (`top_k_sampling=1`), 32 maximum new tokens, seed 240521 for generation, `--seed 42` for Megatron's own initialization, bfloat16. Every record carries the model revision, the resolved execution path (`megatron_cuda_kernels`), the hybrid attention/MLP ratios, the resolved layer pattern, the checkpoint format and directory, and the software versions.

Runtime: torch 2.2.2+cu121, transformer-engine 1.11.0, mamba-ssm 2.0.3, causal-conv1d 1.2.2.post1, Megatron-LM at commit `df61e60`, CUDA 12.1, one NVIDIA A40, compute capability 8.6. See `environment.json` for the full package list and `docs/phase3-runbook.md` for how the bare-metal stack was assembled.

Models and pinned revisions:

| Key | Repository | Revision | Attention layers | PIQA (this run / published target) |
|---|---|---|---|---|
| `mamba2-8b` | `nvidia/mamba2-8b-3t-4k` | `b915550c63ba9359f88f44d1f6a600d85af27302` | none | 79.27 / 79.82 |
| `mamba2-hybrid-8b` | `nvidia/mamba2-hybrid-8b-3t-4k` | `35e8852e2240b350ac2fe2a3b8aa341b5930018e` | ~7% | 79.65 / 79.65 |

Both checkpoints are published only as Megatron-LM distributed checkpoints; transformers cannot load either one, and the hybrid has no transformers modeling class at all. Both are gated on their own published PIQA number (Waleffe et al. 2024, Tables 3 and 7) before the sweep, and both passed within the +/- 1.0 point tolerance.

Zero questions were excluded and zero generations failed scoring in either model's sweep.
The original producing manifest records a dirty source tree.
A later clean rerun at commit `33d6bb5` reproduced the summary byte-for-byte; see `rerun-33d6bb5/REPORT.md`.

See `REPORT.md` for the position-accuracy curves, the primacy/recency edges, and the hybrid-minus-pure composite effect.
