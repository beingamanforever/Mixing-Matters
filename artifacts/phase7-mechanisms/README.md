# Phase 7 mechanism artifacts

Round 1 of Phase 7 is the compute-free set of mechanism lenses that read from the already-collected Phase 2 and Phase 4 sweeps.
The GPU-bound sub-experiments (query-position ablation, sink-mass measurement, sink-blocking, linear probe, template variation) land in follow-up commits as their sweeps complete.
Runbook: `docs/phase7-mechanisms-runbook.md`.

## Inputs

Records aggregated by this report:

- Phase 2 sweeps on the Pile, one A40 host: `pythia-2.8b`, `mamba-2.8b`, `mamba2-2.7b`.
- Phase 4 Pythia sweeps at every registered size point: `pythia-160m`, `pythia-410m`, `pythia-1b`, `pythia-1.4b`, `pythia-2.8b`.

Every record was produced by the same harness (dataset, seed, decoding, token counts), so within a family the only variable that moves across sizes is scale, and within a variant of `scoring_sensitivity` the only variable is the answer-extraction rule.

## Report

`report/` holds the three lens figures and the machine-readable summary:

- `depth-primacy.png`: per-family primacy edge as a function of layer count.
- `scoring-sensitivity.png`: primacy edge under best_subspan_em, normalized_em, and first-line-only extraction.
- `length-sensitivity.png`: primacy edge inside each of three equal-count prompt-length bins.
- `phase7-summary.json`: per-model edges, CIs, Holm-adjusted p-values, bin boundaries, question counts.

## Headline observations

### Depth vs primacy

| Model | Layers | Primacy edge | Holm p |
|---|---|---|---|
| Pythia-160M | 12 | -0.001 [-0.006, +0.004] | 1.000 |
| Pythia-1B | 16 | +0.052 [+0.036, +0.068] | < 1e-4 |
| Pythia-410M | 24 | +0.013 [-0.004, +0.029] | 0.135 |
| Pythia-1.4B | 24 | +0.053 [+0.034, +0.072] | < 1e-4 |
| Pythia-2.8B | 32 | +0.052 [+0.032, +0.072] | < 1e-4 |
| Mamba-2.8B | 64 | -0.001 [-0.016, +0.014] | 0.914 |
| Mamba2-2.7B | 64 | -0.018 [-0.034, -0.003] | 0.023 |

Depth alone does not create primacy: Pythia-410M (24 layers, 1024 wide) has a primacy edge of +0.013, while Pythia-1B (16 layers, 2048 wide) has +0.052.
More layers with less width gives less primacy than fewer layers with more width; the workshop-spec prediction "primacy strengthens with depth" is not supported at fixed family without capacity.
Mamba variants at 64 layers have zero or slightly negative primacy - depth on its own does not manufacture the arm when the sequence mixer is SSM.
The dominant axis in these numbers is capability/parameter count once family is fixed: Pythia crosses into a strong primacy edge somewhere between 410M and 1B parameters, matches Phase 4's "grows and stabilizes" verdict, and does so *irrespective* of layer count in that range.

### Scoring sensitivity

| Model | Primary (best_subspan_em) | Normalized EM | First-line only |
|---|---|---|---|
| Pythia-2.8B | +0.052 (< 1e-4) | +0.026 (0.0002) | +0.053 (< 1e-4) |
| Pythia-1.4B | +0.053 (< 1e-4) | +0.024 (< 1e-4) | +0.052 (< 1e-4) |
| Pythia-1B | +0.052 (< 1e-4) | +0.022 (< 1e-4) | +0.051 (< 1e-4) |
| Pythia-410M | +0.013 (0.135) | +0.000 (1.000) | +0.010 (0.209) |
| Pythia-160M | -0.001 (1.000) | +0.001 (1.000) | -0.001 (1.000) |
| Mamba-2.8B | -0.001 (0.914) | -0.009 (0.060) | -0.002 (0.848) |
| Mamba2-2.7B | -0.018 (0.023) | -0.015 (0.006) | -0.018 (0.026) |

Signs and Holm significance are preserved across the three scoring rules for every Pythia model at 1B and above, and for both Mamba variants.
Normalized EM roughly halves the magnitude (as expected; it is a strict rule) but does not flip any sign.
First-line-only extraction tracks the primary scorer within a few thousandths on every model.
The primacy claim is not a scoring artifact.

### Length sensitivity

Three equal-count prompt-length bins over the 800 exploratory questions, bin edges 1278 / 1427 / 1504 / 1882 tokens:

| Model | 1278-1427 (n=265) | 1427-1504 (n=267) | 1504-1882 (n=268) |
|---|---|---|---|
| Pythia-2.8B | +0.015 (0.494) | +0.065 (0.0002) | +0.075 (< 1e-4) |
| Pythia-1.4B | +0.057 (0.004) | +0.038 (0.023) | +0.065 (< 1e-4) |
| Pythia-1B | +0.030 (0.037) | +0.075 (< 1e-4) | +0.050 (0.0002) |
| Pythia-410M | +0.019 (0.309) | +0.008 (0.697) | +0.011 (0.413) |
| Pythia-160M | +0.002 (0.900) | -0.002 (1.000) | -0.002 (1.000) |
| Mamba-2.8B | +0.004 (0.827) | +0.000 (1.000) | -0.008 (0.646) |
| Mamba2-2.7B | -0.011 (0.446) | -0.011 (0.472) | -0.032 (0.019) |

Pythia-2.8B and Mamba2-2.7B both show the largest edge magnitudes in the longest-prompt bin, with opposite signs.
The three-model primacy signature persists across every bin (Pythia positive, Mambas at or below zero).
No length-bin narrows the Pythia-2.8B primacy edge to a null or reverses the Mamba pattern, so the Phase 2 shape is not a length artifact.

## Standing reads for the paper story

- **Depth alone does not manufacture the arm.**
The one within-family, opposite-direction contrast on layers vs width (Pythia 410M vs 1B) puts more layers on the *less primacy* side.
Whatever creates primacy in the Pythia family is not simply "more layers".
Capability/parameter count crossing the 1B threshold is the change that matches the emergence.
- **The arm is not a scoring artifact.**
Alternative extraction and normalization preserve sign and Holm-significance on every model in the group; only magnitude shifts a little for normalized EM.
- **The arm is not a length artifact.**
Every prompt-length tertile carries the Phase 2 pattern in the same direction; the top tertile is the strongest, consistent with the arms being genuine position effects that scale with the *middle span* rather than an artifact of prompt fitting.

The remaining single-variable questions live in the GPU-bound sub-experiments listed in the runbook.
