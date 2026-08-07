# Phase 4 artifacts

## Question

Does the difference in curve shape between Mamba and Pythia grow, shrink, or stay stable as model size increases?

Five matched size pairs are swept, all on one NVIDIA A40 with 44 GiB so hardware is not a second variable.

| Scale pair | Mamba | Pythia | Status |
|---|---|---|---|
| 130m-160m | `state-spaces/mamba-130m-hf` | `EleutherAI/pythia-160m` | completed |
| 370m-410m | `state-spaces/mamba-370m-hf` | `EleutherAI/pythia-410m` | in progress |
| 790m-1b | `state-spaces/mamba-790m-hf` | `EleutherAI/pythia-1b` | not started |
| 1.4b-1.4b | `state-spaces/mamba-1.4b-hf` | `EleutherAI/pythia-1.4b` | not started |
| 2.8b-2.8b | `state-spaces/mamba-2.8b-hf` | `EleutherAI/pythia-2.8b` | not started |

This directory fills in one pair at a time, since the ten sweeps run sequentially on one GPU.
The scale-trend figures and the grows/shrinks/stable verdict are produced once all five pairs are present, using `phase4-report`.

## Experimental setup

Each model runs 9,600 generations: 800 exploratory questions, the gold document at every position 0 through 9, plus the closed-book floor and the oracle ceiling per question.
Decoding is greedy with temperature 0, top_p 1, top_k unset, `num_beams` 1, 32 maximum new tokens, seed 240521, bfloat16.
Both families use the GPT-NeoX-20B tokenizer, so token counts match across a pair and the same 800 questions fit the 2,048-token context.
Every Mamba model runs on the CUDA kernel execution path, which the runner requires and records.

Runtime: torch 2.7.1+cu126, transformers 4.57.1, mamba-ssm 2.2.6.post3, causal-conv1d 1.5.3.post1, CUDA 12.6, driver 570.133.20, one NVIDIA A40, compute capability 8.6.

## Results so far

### 130m-160m

Edge contrasts, 10,000 paired bootstrap resamples over complete question bundles, Holm corrected across the two edge tests within a model:

| Model | Floor | Ceiling | Primacy edge | Recency edge |
|---|---|---|---|---|
| `mamba-130m` | 0.030 | 0.282 | +0.000, Holm p 1.000 | +0.040, Holm p below 0.0001 |
| `pythia-160m` | 0.000 | 0.009 | -0.001, Holm p 1.000 | -0.002, Holm p 1.000 |

Interaction, Mamba minus Pythia:

| Contrast | Estimate | Interval | Holm p |
|---|---|---|---|
| Primacy edge | +0.0006 | -0.0075 to +0.0094 | 0.911 |
| Recency edge | +0.0419 | see summary | below 0.0001 |

Read this pair with the ceiling in mind.
`pythia-160m` reaches an oracle ceiling of 0.009, so at 160M parameters the Pythia model cannot answer these questions even when the gold document is the only document present.
Its position curve is flat at the floor and carries no shape to compare.
`mamba-130m` reaches a ceiling of 0.282 and shows a recency edge with no primacy edge, the same shape the 2.8B Mamba models show in Phase 2.

The smallest pair is therefore near the capability floor for the Pythia model, and the primacy comparison at this size is not informative on its own.
It is retained as one point in the scale trend, where the larger pairs carry the capability needed to compare curve shape.

Position curve for this pair: `130m-160m/report/position-curves.png`.

## Files per pair

- `<model>-sweep.jsonl.gz`: 9,600 records, one per generation.
- `<model>-positive-control.jsonl.gz`: 500 key-value control records.
- `environment.json`: package versions, `nvidia-smi` output, and the git commit that produced the run.
- `report/position-curves.png`, `report/position-edges.png`, `report/phase2-summary.json`: the two-model comparison and the numbers behind it.

## Limits

Architecture, depth, positional encoding, and capability are partly confounded across a pair, and capability also changes along the size axis.
This phase describes how the curve-shape difference moves with size and cannot on its own prove that any one of those factors caused the trend.
