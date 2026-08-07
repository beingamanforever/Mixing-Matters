# Phase 2 artifacts

## Status

`pythia-2.8b`: completed.
`mamba-2.8b`: completed.
`mamba2-2.7b`: in progress.

All three sweeps run on one host, an NVIDIA A40 with 44 GiB.
The combined report and the interaction contrasts in this directory cover the two completed models.
They are regenerated to include `mamba2-2.7b` once its sweep finishes.

## Why one host

Phase 2 compares accuracy curves across architectures, so the only variable that may change between models is the architecture.
A Pythia sweep was first run on an A10G during Phase 1 development.
Comparing that A10G run against the A40 run of the same model, on 1,741 shared records, the primary score matched on every record, while 15 model responses and one normalized-exact-match score differed.
Two GPUs of the same compute capability select different reduction kernels, so greedy decoding diverges on a small number of borderline tokens.
To keep the hardware from acting as a second variable, every model reported here runs on the A40.

## Experimental setup

Each model runs 9,600 generations: 800 exploratory questions, the gold document placed at every position 0 through 9, plus the closed-book floor and the oracle ceiling for each question.
The question set is identical across models, which is what allows the model comparisons to be paired.

Decoding is greedy with temperature 0, top_p 1, top_k unset, `num_beams` 1, 32 maximum new tokens, seed 240521, bfloat16.
Every record carries the model revision, the resolved execution path, the dataset checksum, and the software versions.

Runtime: torch 2.7.1+cu126, transformers 4.57.1, mamba-ssm 2.2.6.post3, causal-conv1d 1.5.3.post1, CUDA 12.6, driver 570.133.20, one NVIDIA A40, compute capability 8.6.

Models and pinned revisions:

| Key | Repository | Revision | Layers | Execution path |
|---|---|---|---|---|
| `pythia-2.8b` | `EleutherAI/pythia-2.8b` | `2a259cdd96a4beb1cdf467512e3904197345f6a9` | 32 | pytorch reference, eager attention |
| `mamba-2.8b` | `state-spaces/mamba-2.8b-hf` | `96c48e0292b63f5346b6d30061af2551f7101e26` | 64 | CUDA kernels |
| `mamba2-2.7b` | `AntonV/mamba2-2.7b-hf` | `ef542707386fa9ec86bbf8a35ed2952af84bf566` | 64 | CUDA kernels |

`state-spaces/mamba-2.8b` and `state-spaces/mamba2-2.7b` cannot be loaded by transformers 4.57.1, because their configs carry no `model_type`.
The Mamba-1 entry is the official transformers conversion.
The Mamba-2 entry is a community conversion.
It was validated against the original checkpoint run through the authors' own `mamba_ssm` implementation: 5 of 5 sampled greedy generations identical, 5 of 5 top-1 next-token agreement, and maximum absolute logit differences of 0.25 to 0.38 against a logit scale of 36 to 52.

Both Mamba models run on the CUDA kernel execution path, which the runner requires and records.
A fall back to the numerically different reference path raises instead of running.

## Finding

Architecture and evidence position interact, and the interaction is in the primacy arm.

Edge contrasts per model, 10,000 paired bootstrap resamples over complete question bundles, Holm corrected across the two edge tests within a model:

| Model | Primacy, mean(0,1) minus mean(4,5) | Recency, mean(8,9) minus mean(4,5) |
|---|---|---|
| `pythia-2.8b` | +0.0519, interval +0.0319 to +0.0719, Holm p below 0.0001 | +0.0750, interval +0.0537 to +0.0969, Holm p below 0.0001 |
| `mamba-2.8b` | -0.0013, interval -0.0162 to +0.0137, Holm p 0.914 | +0.0769, interval +0.0575 to +0.0969, Holm p below 0.0001 |

Pythia has a primacy edge and a recency edge.
Mamba-1 has a recency edge and no measurable primacy edge.

Interaction contrasts between the two models, same paired bootstrap, Holm corrected across the two edge tests:

| Contrast | Mamba-1 minus Pythia | Interval | Holm p |
|---|---|---|---|
| Primacy edge | -0.0531 | -0.0775 to -0.0281 | below 0.0001 |
| Recency edge | +0.0019 | -0.0262 to +0.0294 | 0.920 |

The primacy edge differs by architecture and the difference excludes zero.
The recency edge does not differ by architecture.
Both models recover evidence at the end of the document list to a similar degree, and only the transformer additionally recovers evidence at the start.

Position curves and the numbers behind them are in `report/`.

## Anchors and controls

| Model | Floor, closed book | Ceiling, oracle |
|---|---|---|
| `pythia-2.8b` | 0.091 | 0.640 |
| `mamba-2.8b` | 0.115 | 0.615 |

The two models reach a similar ceiling, so the primacy difference is not explained by one model being unable to use the gold document.

Key-value positive control, run against each model before its sweep:

- `pythia-2.8b`: accuracy 0.94 at slot 0 and 0.16 at slot 9, edge mean 0.55 against middle mean 0.17, difference 0.38. The control passes.
- `mamba-2.8b`: accuracy 0.0 at every slot. The control does not pass.

The key-value control is recorded and does not gate the sweep.
Mamba-1 scoring zero on key-value retrieval is a property of the model: a fixed-size recurrent state cannot store thirty random key-value pairs without loss.
The multi-document QA sweep for the same model is unaffected, with an oracle accuracy of 0.615, so the pipeline is working and the zero is a model result rather than a pipeline failure.

## Files per model

- `sweep.jsonl.gz`: 9,600 records, one per generation.
- `positive-control.jsonl.gz`: 500 key-value control records for that model.
- `environment.json`: package versions, `nvidia-smi` output, and the git commit that produced the run.

Combined report across the completed models, in `report/`:

- `position-curves.png`: accuracy against gold position, one line per model, with bootstrap intervals and floor and ceiling reference lines.
- `position-edges.png`: primacy and recency edges per model with intervals.
- `phase2-summary.json`: the numbers behind the figures, including every pairwise interaction and the exclusion counts.

## Limits

Two models are reported here.
`mamba2-2.7b` is still running and joins the comparison when it finishes.

The comparison holds architecture against a background of other differences that are not fully separated: Pythia has 32 layers and partial rotary positional encoding, while Mamba-1 has 64 layers and no explicit positional encoding.
Depth and positional encoding are therefore confounded with the sequence-mixing architecture in this contrast, which later phases address.
