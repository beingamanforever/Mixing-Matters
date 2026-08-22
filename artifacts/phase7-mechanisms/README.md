# Phase 7 mechanism artifacts

Phase 7 tests *why* the primacy and recency arms appear where they do.
Runbook: `docs/phase7-mechanisms-runbook.md`.
Cross-phase synthesis using every phase's evidence: **`CUMULATIVE-FINDINGS.md`**.

## Sub-experiments

| Sub-experiment | Directory | Finding |
|---|---|---|
| 4b depth trend + 4e scoring/length lenses | `report/` | A simple depth-only account is insufficient; the Phase 2 pattern persists under the tested scoring rules and length bins. |
| 4a query-position variants | `4a-query-position/` | Bookend Mamba primacy is +0.103 versus a null baseline; question-first recency is +0.010 versus +0.070; no paired between-condition test was run. |
| 4c attention-sink mass | `4c-sink-scan/`, `4c-sink-scan-variants/` | Late-layer token-0 sink mass is associated with primacy across Pythia scale; pythia-2.8b final-layer sink is 0.455 and pythia-160m is near zero. |
| 4c sink-block ablation | `4c-sink-block/` | Null intervention: Nemotron-H custom attention ignores the token-0 mask; documented follow-up. |
| 4d linear probe | `4d-probe/` | Gold position is more linearly decodable in the measured Mamba state than in Pythia (0.65 versus 0.60), but the probe does not measure answer-content storage or use. |
| 4e instruction-template variation | `4e-template/` | Pythia primacy is positive under two templates, while the instructional-template interval crosses zero; the cells have no paired between-template test. |

The compute-free lenses (below) read the already-collected Phase 2 and Phase 4 sweeps; the GPU sub-experiments were run across one L40S and two T4s.

## Inputs

Records aggregated by this report:

- Phase 2 sweeps on the Pile, one A40 host: `pythia-2.8b`, `mamba-2.8b`, `mamba2-2.7b`.
- Phase 4 Pythia sweeps at every registered size point: `pythia-160m`, `pythia-410m`, `pythia-1b`, `pythia-1.4b`, `pythia-2.8b`.

Every record was produced by the same harness for dataset, seed, decoding, and prompt construction.
Across model sizes, parameter count, depth, width, training progress, and capability can move together, while each `scoring_sensitivity` variant changes only the answer-extraction rule applied to fixed outputs.

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

The observations do not support a simple depth-only account: Pythia-410M (24 layers, 1024 wide) has a primacy edge of +0.013, while Pythia-1B (16 layers, 2048 wide) has +0.052.
More layers with less width gives less primacy than fewer layers with more width; the workshop-spec prediction "primacy strengthens with depth" is not supported at fixed family without capacity.
Mamba variants at 64 layers have zero or slightly negative primacy - depth on its own does not manufacture the arm when the sequence mixer is SSM.
The edge appears in the capable Pythia checkpoints at 1B and above, but these observations do not identify a causal threshold because scale, width, depth, training progress, and task capability move together.

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
The tested alternative scoring rules do not explain the Pythia primacy pattern.

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
No length bin reverses the Phase 2 family pattern, so the result is not explained by the tested prompt-length stratification alone.

## Standing reads for the paper story

- **Depth alone does not manufacture the arm.**
The one within-family, opposite-direction contrast on layers vs width (Pythia 410M vs 1B) puts more layers on the *less primacy* side.
Whatever creates primacy in the Pythia family is not simply "more layers".
The appearance of the edge coincides with higher capability and parameter count in this series, without identifying a causal threshold.
- **The tested scoring rules do not explain the arm.**
Alternative extraction and normalization preserve sign and Holm-significance on every model in the group; only magnitude shifts a little for normalized EM.
- **The tested length bins preserve the family pattern.**
Every prompt-length tertile carries the Phase 2 pattern in the same direction, but the bins are descriptive and can differ in question composition.

The executed GPU sub-experiments generate bounded mechanism hypotheses; a valid causal intervention remains open.
