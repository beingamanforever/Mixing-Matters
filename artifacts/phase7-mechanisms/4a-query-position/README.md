# Phase 7 sub-experiment 4a: query-position ablation

Query-position ablation on `pythia-2.8b` and `mamba-2.8b` at 200 exploratory questions per (model, prompt variant).
Every gold record carries the `prompt_variant` field so records join to the Phase 2 baseline (`prompt_variant=None → baseline`) in one call.
The ten distractors and gold-document placement are identical across variants, but the variants jointly change query placement, repetition, and token layout.

## Setup

- Hosts: L40S 46GB (`pythia-2.8b` on eager attention; `mamba-2.8b` on the mamba CUDA-kernel path).
- Questions: 200 per (model, variant).
- Variants: `baseline` (Phase 2, documents then question), `question_first` (question then documents), `bookend` (question before and after documents; Liu et al. `query_aware_contextualization`), `gold_padded` (baseline plus 128 filler tokens inserted between the document block and the closing question).
- Every other knob matches Phase 2: seed 240521, greedy decoding, 32 new tokens, BF16, tokenizer and dataset checksums recorded on every JSONL line.

## Results

95 percent bootstrap intervals shown as `[low, high]`; Holm p-values adjust across the two edges within a model.

### mamba-2.8b

| Variant | Primacy edge | Recency edge |
|---|---|---|
| baseline | +0.000 [-0.035, +0.035], p=1.000 | +0.070 [+0.030, +0.113], p=0.003 |
| question_first | +0.020 [-0.008, +0.048], p=0.366 | +0.010 [-0.020, +0.040], p=0.562 |
| bookend | **+0.103 [+0.055, +0.150], p<1e-4** | +0.095 [+0.058, +0.138], p<1e-4 |
| gold_padded | +0.028 [-0.013, +0.068], p=0.183 | +0.070 [+0.030, +0.113], p<1e-4 |

### pythia-2.8b

| Variant | Primacy edge | Recency edge |
|---|---|---|
| baseline | +0.070 [+0.030, +0.113], p=0.003 | +0.053 [+0.010, +0.095], p=0.016 |
| question_first | +0.088 [+0.053, +0.120], p<1e-4 | +0.025 [-0.005, +0.055], p=0.132 |
| bookend | **+0.150 [+0.113, +0.190], p<1e-4** | +0.045 [+0.015, +0.075], p=0.004 |
| gold_padded | +0.038 [-0.010, +0.083], p=0.123 | +0.055 [+0.013, +0.098], p=0.022 |

## Reading the result

### Mamba edges differ descriptively across prompt variants

Mamba-2.8B has a null primacy edge under the baseline documents-then-question layout (Phase 2 result reproduced at n=200).
Under `bookend`, the per-condition primacy estimate is +0.103, larger than any Mamba primacy edge measured in Phases 2, 4, or 8.
The bookend layout places question tokens before the document block and repeats them after it, so the result is consistent with a fixed-state-compression hypothesis but does not isolate it.
No paired between-condition contrast was computed, so the difference between the baseline and bookend estimates is descriptive.

Under `question_first` (only prefix, no trailing question), primacy is +0.020 (still not clean, wide CI) and recency is +0.010, compared with +0.070 at baseline.
This pattern is consistent with the hypothesis that early query exposure changes what the recurrent state retains, but it does not establish that mechanism.

### Pythia already has a primacy arm, so the effects are additive

Pythia's baseline primacy is +0.070 (Phase 2 replicated on the L40S at n=200).
The `bookend` per-condition estimate is +0.150.
The additional prefix question tokens sit near the beginning of the prompt, which is consistent with an attention-sink hypothesis.
The `question_first` primacy estimate is +0.088, which is descriptively above baseline and consistent with more query-relevant mass at the start of the prompt.

The `gold_padded` cell (128 filler tokens between documents and question) has Pythia primacy +0.038 and recency near baseline.
Filler tokens between the documents and the closing question also change token distances and layout, so the lower estimate is descriptive and does not identify how attention moved.

### Combined direction

Mamba primacy is 0 at baseline and +0.103 under `bookend`.
Pythia primacy is +0.070 at baseline and +0.150 under the same variant.
The Mamba point estimate moves from a null baseline to a positive bookend cell, while Pythia is positive in both cells.
Because the variant changes placement, repetition, and layout and lacks a paired between-condition test, it does not establish that query-first helps one mixer more than another.

## Artifacts

- `pythia-2.8b-{question_first,bookend,gold_padded}/sweep.jsonl.gz`: raw sweep records.
- `mamba-2.8b-{question_first,bookend,gold_padded}/sweep.jsonl.gz`: raw sweep records.
- `report/phase7-variants-summary.json`: per-model per-variant position curves, edges with 95 percent bootstrap intervals, and Holm-adjusted p-values.

`mamba2-2.7b` was attempted on a T4 but the triton mamba-2 kernel failed to compile on sm_75 (`PassManager::run failed`), so mamba2 is deferred to a host with a working triton path (L40S or A10G).
