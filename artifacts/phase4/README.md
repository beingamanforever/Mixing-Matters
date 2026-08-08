# Phase 4 artifacts

## Question

Does the difference in curve shape between Mamba and Pythia grow, shrink, or stay stable as model size increases?

Five matched size pairs are swept, all on one NVIDIA A40 with 44 GiB so hardware is not a second variable.

| Scale pair | Mamba | Pythia | Status |
|---|---|---|---|
| 130m-160m | `state-spaces/mamba-130m-hf` | `EleutherAI/pythia-160m` | completed |
| 370m-410m | `state-spaces/mamba-370m-hf` | `EleutherAI/pythia-410m` | completed |
| 790m-1b | `state-spaces/mamba-790m-hf` | `EleutherAI/pythia-1b` | completed |
| 1.4b-1.4b | `state-spaces/mamba-1.4b-hf` | `EleutherAI/pythia-1.4b` | completed |
| 2.8b-2.8b | `state-spaces/mamba-2.8b-hf` | `EleutherAI/pythia-2.8b` | in progress |

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

### 370m-410m

Both models in this pair reach a usable oracle ceiling, so their position curves carry shape to compare.

| Model | Floor | Ceiling | Primacy edge | Recency edge |
|---|---|---|---|---|
| `mamba-370m` | 0.044 | 0.479 | +0.010, Holm p 0.043 | +0.078, Holm p below 0.0001 |
| `pythia-410m` | 0.029 | 0.357 | +0.013, Holm p 0.135 | +0.033, Holm p 0.001 |

Interaction, Mamba minus Pythia:

| Contrast | Estimate | Interval | Holm p |
|---|---|---|---|
| Primacy edge | -0.0025 | -0.0206 to +0.0162 | 0.793 |
| Recency edge | +0.0456 | see summary | below 0.0001 |

At this size the primacy edge is small in both models and the primacy difference between them is not distinguishable from zero.
This differs from the 2.8B pair reported in Phase 2, where the Pythia primacy edge is +0.052 and the Mamba primacy edge is near zero, giving a primacy difference of -0.053 that excludes zero.
The recency edge is larger in Mamba than in Pythia at this size, and the recency difference of +0.046 excludes zero.

The two completed interior points do not yet settle the trend.
The grows, shrinks, or stable verdict is computed from the smallest and largest size points once all five pairs are present.

Position curve for this pair: `370m-410m/report/position-curves.png`.

### 790m-1b

| Model | Ceiling | Primacy edge | Recency edge |
|---|---|---|---|
| `mamba-790m` | 0.539 | -0.010, Holm p 0.109 | +0.062, Holm p below 0.0001 |
| `pythia-1b` | 0.496 | +0.052, Holm p below 0.0001 | +0.071, Holm p below 0.0001 |

Interaction, Mamba minus Pythia:

| Contrast | Estimate | Interval | Holm p |
|---|---|---|---|
| Primacy edge | -0.0619 | -0.0831 to -0.0419 | below 0.0001 |
| Recency edge | -0.0087 | see summary | 0.481 |

At this size the Pythia model has a primacy edge of +0.052 and the Mamba model has none, giving a primacy difference of -0.062 that excludes zero.
The recency edge is similar in both.
This matches the 2.8B pair from Phase 2, where the primacy difference is -0.053.

Reading the three informative points together, the smallest pair aside, the primacy difference is close to zero at 410M, then -0.062 at the 790M-to-1B point, then -0.053 at 2.8B.
The Pythia primacy edge specifically strengthens with size, from +0.013 at Holm p 0.135 at 410M to +0.052 at Holm p below 0.0001 at 1B and again at 2.8B, while the Mamba primacy edge stays near zero at every size.
The gap between the two architectures in the primacy arm therefore appears to emerge with scale rather than being present at every size.
The formal grows, shrinks, or stable verdict from `phase4-report` compares the smallest and largest size endpoints and is produced once all five pairs are present.

Position curve for this pair: `790m-1b/report/position-curves.png`.

### 1.4b-1.4b

Both models in this pair reach a usable oracle ceiling, so their position curves carry shape to compare.

| Model | Ceiling | Primacy edge | Recency edge |
|---|---|---|---|
| `mamba-1.4b` | 0.554 | -0.016, Holm p 0.017 | +0.091, Holm p below 0.0001 |
| `pythia-1.4b` | 0.594 | +0.053, Holm p below 0.0001 | +0.068, Holm p below 0.0001 |

Interaction, Mamba minus Pythia:

| Contrast | Estimate | Interval | Holm p |
|---|---|---|---|
| Primacy edge | -0.0694 | -0.0931 to -0.0462 | below 0.0001 |
| Recency edge | +0.0225 | see summary | 0.111 |

At this size the Pythia model has a primacy edge of +0.053 and the Mamba model has a small negative one, giving a primacy difference of -0.069 that excludes zero.
The recency difference does not exclude zero.
This is the same direction and roughly the same magnitude as the 790M-to-1B and 2.8B pairs.

Reading the four informative points together, the smallest pair aside, the primacy difference is close to zero at 410M, then -0.062 at the 790M-to-1B point, -0.069 at 1.4B, and -0.053 at 2.8B.
The Pythia primacy edge specifically strengthens with size, from +0.013 at Holm p 0.135 at 410M to +0.052 or more from 1B upward, while the Mamba primacy edge stays at or below zero at every size.
The gap between the two architectures in the primacy arm therefore appears to emerge with scale rather than being present at every size.
The formal grows, shrinks, or stable verdict from `phase4-report` compares the smallest and largest size endpoints and is produced once all five pairs are present.

Position curve for this pair: `1.4b-1.4b/report/position-curves.png`.

## Files per pair

- `<model>-sweep.jsonl.gz`: 9,600 records, one per generation.
- `<model>-positive-control.jsonl.gz`: 500 key-value control records.
- `environment.json`: package versions, `nvidia-smi` output, and the git commit that produced the run.
- `report/position-curves.png`, `report/position-edges.png`, `report/phase2-summary.json`: the two-model comparison and the numbers behind it.

## Limits

Architecture, depth, positional encoding, and capability are partly confounded across a pair, and capability also changes along the size axis.
This phase describes how the curve-shape difference moves with size and cannot on its own prove that any one of those factors caused the trend.
