# Phase 8 artifacts

Descriptive system comparison across three complete 7-8B production systems on Lost-in-the-Middle 10-document QA.
The sweep is not a matched control: the systems differ from each other on many axes at once - architecture, pretraining corpus, token count, tokenizer, alignment status, depth, and positional encoding - so the phase describes how full-system position curves look side by side, not which single variable is responsible for any curve difference.
Runbook: `docs/phase8-runbook.md`.

## Systems and sweeps

| Model | Family | Repo | Sweep records | Positive control |
|---|---|---|---|---|
| `nemotron-h-8b` | nemotron-h | `nvidia/Nemotron-H-8B-Base-8K` | 9,600 (800 Q x 12 conditions) | 500 |
| `llama-3.1-8b` | llama | `meta-llama/Llama-3.1-8B` | 9,600 | 500 |
| `qwen2.5-7b` | qwen2 | `Qwen/Qwen2.5-7B` | 9,600 | 500 |

Nemotron-H ran on the eager attention implementation (its SSM blocks dispatch through `mamba-ssm` and `causal-conv1d`).
Llama and Qwen ran on the SDPA attention implementation, because eager on an 8B dense transformer is prohibitively slow on a single L40S and Phase 8 is a descriptive comparison rather than a matched control.
Every raw record carries the actual `execution_path` and `attention_implementation` used.

The prompt-token span tolerance was raised from `MAX_PROMPT_TOKEN_SPAN=2` (tight enough for the GPTNeoX and Mamba tokenizers used in Phases 2 through 6) to 8, because the three Phase 8 tokenizers each have different byte-pair merges at document boundaries.

## Report

`report/` holds the final three-model artifacts:

- `position-curves.png`: accuracy against gold position 0-9 for each system, with floor and ceiling reference lines.
- `position-edges.png`: primacy and recency edge bars per system, both with 95 percent bootstrap intervals.
- `phase8-summary.json`: the full machine-readable summary - per-model position curves, per-model edges, every pairwise interaction, floor and ceiling means, and a small system descriptor block.

## Per-model raw outputs

- `nemotron-h-8b/`: `sweep.jsonl`, `positive-control.jsonl`, `environment.json`.
- `llama-3.1-8b/`: `sweep.jsonl`, `positive-control.jsonl`, `environment.json`.
- `qwen2.5-7b/`: `sweep.jsonl`, `positive-control.jsonl`, `environment.json`.

## Headline numbers

Per-model edges over 800 exploratory questions each, Holm-adjusted across the two edges within a model:

| System | Floor | Ceiling | Primacy edge (95% CI) | Recency edge (95% CI) |
|---|---|---|---|---|
| Nemotron-H-8B | 0.311 | 0.588 | +0.067 [+0.044, +0.091], p = 0.0000 | +0.029 [+0.008, +0.049], p = 0.006 |
| Llama-3.1-8B | 0.315 | 0.782 | +0.076 [+0.057, +0.096], p = 0.0000 | +0.018 [+0.000, +0.034], p = 0.051 |
| Qwen2.5-7B | 0.280 | 0.815 | +0.041 [+0.021, +0.061], p = 0.0008 | +0.008 [-0.012, +0.027], p = 0.474 |

Pairwise interactions, edge of the first system minus edge of the second, paired bootstrap over 800 shared questions:

| Contrast | Primacy diff (95% CI) | Recency diff (95% CI) |
|---|---|---|
| Nemotron-H minus Llama-3.1 | -0.009 [-0.040, +0.021], p = 0.818 | +0.011 [-0.014, +0.037], p = 0.818 |
| Nemotron-H minus Qwen2.5 | +0.026 [-0.004, +0.056], p = 0.178 | +0.021 [-0.006, +0.049], p = 0.178 |
| Llama-3.1 minus Qwen2.5 | +0.036 [+0.009, +0.063], p = 0.016 | +0.010 [-0.015, +0.036], p = 0.447 |

## Reading the result

All three systems have a positive primacy edge that clears the multiplicity-corrected threshold: accuracy at gold positions 0-1 sits 4-8 percentage points above accuracy at positions 4-5.
The recency edge is smaller and only clearly non-zero for Nemotron-H-8B.
The one pairwise contrast that clears the Holm threshold on either edge is Llama-3.1 minus Qwen2.5 on primacy: Llama's primacy edge sits about 3.6 points above Qwen's.

Ceiling accuracy separates the systems by their QA capacity given the oracle document: Qwen2.5 (0.815) > Llama-3.1 (0.782) > Nemotron-H (0.588).
Floors sit near 0.28-0.32 for all three, so closed-book knowledge overlap is comparable.

Phase 8 answers no matched-control question.
Every pairwise interaction jointly reflects architecture, corpus, token count, tokenizer, alignment status, depth, and positional encoding.
Single-variable claims belong to Phase 2, Phase 3, and Phase 5.
