# Phase 8 artifacts

Descriptive system comparison across three complete 7-8B production systems on Lost-in-the-Middle 10-document QA.
The sweep is not a matched control: the systems differ from each other on many axes at once - architecture, pretraining corpus, token count, tokenizer, alignment status, depth, and positional encoding - so the phase describes how full-system position curves look side by side, not which single variable is responsible for any curve difference.
Runbook: `docs/phase8-runbook.md`.

## Sweep status

| Model | Family | Repo | Sweep | Positive control |
|---|---|---|---|---|
| `nemotron-h-8b` | nemotron-h | `nvidia/Nemotron-H-8B-Base-8K` | 9,600 records (800 Q x 12 conditions) | 500 records, kv accuracy ~0.32 average |
| `qwen2.5-7b` | qwen2 | `Qwen/Qwen2.5-7B` | 9,600 records | 500 records |
| `llama-3.1-8b` | llama | `meta-llama/Llama-3.1-8B` | in progress | in progress |

The Nemotron-H sweep ran on the eager attention implementation (its SSM blocks dispatch through `mamba-ssm` and `causal-conv1d`).
The Llama and Qwen sweeps run on the SDPA attention implementation; eager on an 8B dense transformer is prohibitively slow on a single L40S and Phase 8 is a descriptive comparison rather than a matched control, so the cross-family determinism trade-off is minor.
Every raw record still carries the actual `execution_path` and `attention_implementation` it used.

## Reports

`report-partial-nemotron-qwen/` is the intermediate two-model report emitted after Nemotron-H and Qwen2.5 finished; it will be superseded by `report/` once the Llama sweep completes.

## Per-model raw outputs

- `nemotron-h-8b/`: `sweep.jsonl`, `positive-control.jsonl`, `environment.json`.
- `qwen2.5-7b/`: `sweep.jsonl`, `positive-control.jsonl`, `environment.json`.
- `llama-3.1-8b/`: pending.

## Headline numbers so far

Nemotron-H-8B: primacy edge 0.067 (95% CI [0.044, 0.091], Holm-adjusted p = 0.0000), recency edge 0.029 ([0.008, 0.049], p = 0.006), floor 0.311, ceiling 0.588.

Qwen2.5-7B: primacy edge 0.041 ([0.021, 0.061], p = 0.0008), recency edge 0.008 ([-0.012, 0.027], p = 0.474), floor 0.280, ceiling 0.815.

Nemotron-H minus Qwen2.5 interaction: primacy 0.026 ([-0.004, 0.056], p = 0.178), recency 0.021 ([-0.006, 0.049], p = 0.178).
Both systems place accuracy at positions 0-1 above the middle, and the Nemotron-H curve also lifts at the last positions.
The two-system interaction does not clear the multiplicity-corrected threshold on either edge; adding the Llama result will refine that.
