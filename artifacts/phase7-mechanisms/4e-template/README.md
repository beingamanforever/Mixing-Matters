# Phase 7 sub-experiment 4e: instruction-template variation

Exploratory prompt-template sensitivity check.
Re-runs `pythia-2.8b` and `mamba-2.8b` at 200 questions under two alternative instruction templates and compares the primacy and recency edges to the vendored Liu et al. baseline.
All three templates keep the documents-then-question order; only the instruction wording and the answer cue change (`liu` default, `concise`, `instructional`).

## Results

95 percent bootstrap intervals; Holm-adjusted across the two edges within a (model, template) cell.

### pythia-2.8b

| Template | Primacy edge | Recency edge |
|---|---|---|
| liu (baseline) | +0.070 [+0.030, +0.113], p=0.003 | +0.053, p=0.016 |
| concise | +0.078 [+0.038, +0.118], p<1e-4 | +0.080, p=0.0002 |
| instructional | +0.005 [-0.040, +0.048], p=0.858 | +0.063, p=0.025 |

### mamba-2.8b

| Template | Primacy edge | Recency edge |
|---|---|---|
| liu (baseline) | +0.000 [-0.035, +0.035], p=1.000 | +0.070, p=0.003 |
| concise | -0.003 [-0.035, +0.030], p=0.936 | +0.060, p=0.001 |
| instructional | -0.010 [-0.045, +0.025], p=0.615 | +0.055, p=0.010 |

## Reading the result

The per-template cells show two descriptive patterns.

- Mamba-2.8B has a null primacy edge under every template (0.000, -0.003, -0.010, none distinguishable from zero), while Pythia has a positive primacy edge under the two templates that keep a short instruction head (liu +0.070, concise +0.078).
- The instructional-template Pythia estimate is +0.005 with an interval that crosses zero, so the sign pattern is not template-invariant across all executed cells.
- The Pythia primacy estimates differ by template: the `instructional` template, which prepends a longer multi-sentence instruction and an explicit `Documents:` header before the document block, has primacy +0.005 (not significant) and recency +0.063. Mamba primacy remains near zero in all three cells.

That lower Pythia estimate is consistent with the attention-sink hypothesis, but it does not test that mechanism.
The `instructional` template inserts extra tokens at the very head of the prompt, the region measured by the 4c sink-mass scan, and co-occurs with the lower Pythia primacy estimate.
The `gold_padded` cell in 4a also changes token distances and has a lower Pythia primacy estimate, but neither comparison isolates attention allocation.

No paired between-template contrast was computed, so differences between template point estimates are descriptive rather than tested effects.

Recency is stable across all templates for both models, consistent with recency being anchored to the end of the prompt (unchanged by instruction wording) rather than to the head.

## Caveats

- Absolute primacy numbers should not be quoted without the template, and neither the sign nor the architecture contrast is established as template-robust across all executed cells.
- 200 questions per cell; intervals are wider than the 800-question Phase 2 baselines.

## Artifacts

- `pythia-2.8b-{concise,instructional}.jsonl.gz`, `mamba-2.8b-{concise,instructional}.jsonl.gz`: raw sweeps.
- `report/phase7-variants-summary.json`: per-model per-template curves and edges (the Liu baseline cell is drawn from the Phase 2 sweep and labeled `baseline`).
