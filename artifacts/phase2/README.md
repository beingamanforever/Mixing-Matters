# Phase 2 artifacts

## Status

`pythia-2.8b`: completed.
`mamba-2.8b`: in progress.
`mamba2-2.7b`: not started.

This directory is filled in one model at a time, because the three sweeps run sequentially on one GPU.
The cross-model comparison, the interaction contrasts, and the Phase 2 report are produced once all three sweeps are present.

## Experimental setup

Each model runs 9,600 generations: 800 exploratory questions, the gold document placed at every position 0 through 9, plus the closed-book floor and the oracle ceiling for each question.
The question set is identical across models, which is what allows the model comparisons to be paired.

Decoding is greedy with temperature 0, top_p 1, top_k unset, `num_beams` 1, 32 maximum new tokens, seed 240521, bfloat16.
Every record carries the model revision, the resolved execution path, the dataset checksum, and the software versions.

Models and pinned revisions:

| Key | Repository | Revision | Layers |
|---|---|---|---|
| `pythia-2.8b` | `EleutherAI/pythia-2.8b` | `2a259cdd96a4beb1cdf467512e3904197345f6a9` | 32 |
| `mamba-2.8b` | `state-spaces/mamba-2.8b-hf` | `96c48e0292b63f5346b6d30061af2551f7101e26` | 64 |
| `mamba2-2.7b` | `AntonV/mamba2-2.7b-hf` | `ef542707386fa9ec86bbf8a35ed2952af84bf566` | 64 |

`state-spaces/mamba-2.8b` and `state-spaces/mamba2-2.7b` cannot be loaded by transformers 4.57.1, because their configs carry no `model_type`.
The Mamba-1 entry above is the official transformers conversion.
The Mamba-2 entry is a community conversion with no published numerical validation, which is recorded as a limitation in the report.

Both Mamba models run on the CUDA kernel execution path, which the runner requires and records.
A fall back to the numerically different reference path raises instead of running.

## Results so far

### pythia-2.8b

800 complete questions, no excluded questions, no excluded records, no scoring failures.

Accuracy by gold position, with 95 percent paired bootstrap intervals over complete question bundles:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.266 | 0.198 | 0.196 | 0.190 | 0.180 | 0.180 | 0.204 | 0.217 | 0.231 | 0.279 |

Edge contrasts, 10,000 resamples, Holm corrected across the two tests:

| Contrast | Estimate | 95 percent interval | Holm p |
|---|---|---|---|
| Primacy, mean(0,1) minus mean(4,5) | +0.0519 | +0.0319 to +0.0719 | below 0.0001 |
| Recency, mean(8,9) minus mean(4,5) | +0.0750 | +0.0537 to +0.0969 | below 0.0001 |

Floor accuracy 0.092, ceiling accuracy 0.641.
Both edges are positive and their intervals exclude zero, so accuracy at the ends of the document list exceeds accuracy in the middle.
The recency edge is larger than the primacy edge for this model.

Key-value positive control, run against this model before the sweep: accuracy 0.94 at slot 0 and 0.16 at slot 9, edge mean 0.55 against middle mean 0.17, difference 0.38.
These control numbers are identical to the Phase 1 control run, as expected from the same model, seeds, and pinned revision.

Floor 0.092 and ceiling 0.641 here against 0.095 and 0.650 in Phase 1 differ because Phase 1 used the first 200 exploratory questions and Phase 2 uses all 800.

Prompt length is unchanged across positions for 775 questions and shifts by exactly one token for 25 questions, caused by byte-pair merges at document boundaries.
The observed span is recorded on every gold record.
The Phase 2 report will repeat the edge analysis with those 25 questions excluded, as a sensitivity check.

## Files per model

- `sweep.jsonl.gz`: 9,600 records, one per generation.
- `positive-control.jsonl.gz`: 500 key-value control records for that model.
- `environment.json`: package versions, `nvidia-smi` output, and the git commit that produced the run.
- `report/position-curves.png`: accuracy against gold position with bootstrap intervals and floor and ceiling reference lines.
- `report/position-edges.png`: primacy and recency edges with intervals.
- `report/phase2-summary.json`: the numbers behind the figures, including exclusion counts.

The per-model `report/` directories cover one model each.
The combined report across all three models replaces them once the sweeps finish.
