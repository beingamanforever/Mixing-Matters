# Phase 7 sub-experiment 4e: instruction-template variation

Measurement-artifact check on the prompt template.
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

Two things move independently.

- The **qualitative architecture contrast is template-invariant**: Mamba-2.8B has a null primacy edge under every template (0.000, -0.003, -0.010, none distinguishable from zero), and Pythia has a positive primacy edge under the two templates that keep a short instruction head (liu +0.070, concise +0.078). The Pythia-vs-Mamba primacy gap is not a creation of the specific Liu wording.
- The **magnitude of the transformer's primacy edge is template-sensitive**: the `instructional` template, which prepends a longer multi-sentence instruction and an explicit `Documents:` header before the document block, collapses Pythia's primacy to +0.005 (not significant) while leaving its recency intact (+0.063). Nothing about Mamba changes.

That collapse is itself evidence for the attention-sink reading rather than against it.
The `instructional` template inserts extra tokens at the very head of the prompt, exactly the region where the token-0 sink and the primacy-relevant early context live (see the 4c sink-mass scan). Pushing the gold document further from the sink-bearing head weakens the primacy arm, mirroring the `gold_padded` result in 4a where inserting filler between documents and question halved Pythia's primacy.

Recency is stable across all templates for both models, consistent with recency being anchored to the end of the prompt (unchanged by instruction wording) rather than to the head.

## Caveats

- Absolute primacy numbers should not be quoted without the template. The safe, template-robust claims are the *signs* and the *architecture contrast*, not the exact edge value.
- 200 questions per cell; intervals are wider than the 800-question Phase 2 baselines.

## Artifacts

- `pythia-2.8b-{concise,instructional}.jsonl.gz`, `mamba-2.8b-{concise,instructional}.jsonl.gz`: raw sweeps.
- `report/phase7-variants-summary.json`: per-model per-template curves and edges (the Liu baseline cell is drawn from the Phase 2 sweep and labeled `baseline`).
