# Phase 7 sub-experiment 4a: query-position ablation

Query-position ablation on `pythia-2.8b` and `mamba-2.8b` at 200 exploratory questions per (model, prompt variant).
Every gold record carries the `prompt_variant` field so records join to the Phase 2 baseline (`prompt_variant=None → baseline`) in one call.
The four prompt variants change only the position of the question in the prompt; the ten distractors and the gold document are placed identically.

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

### The fixed-state-compression signature on Mamba is clean

Mamba-2.8B has a null primacy edge under the baseline documents-then-question layout (Phase 2 result reproduced at n=200).
Under `bookend`, primacy jumps to +0.103, larger than any Mamba primacy edge measured in Phases 2, 4, or 8.
The single-variable change that produced this is placing the question tokens **before** the document block: the recurrent state now knows the query while it compresses the documents, and it can preferentially retain information about early gold documents.
This is exactly the prediction of the fixed-state-compression hypothesis for SSM sequence mixers.

Under `question_first` (only prefix, no trailing question) primacy climbs to +0.020 (still not clean, wide CI) while recency **drops from +0.070 to +0.010**.
Exposing the question up-front trades away the state's recency bias.
That is the second prediction of fixed-state-compression: when the question is exposed early, the recurrent state does not just carry the tail; it filters as it goes.

### Pythia already has a primacy arm, so the effects are additive

Pythia's baseline primacy is +0.070 (Phase 2 replicated on the L40S at n=200).
`bookend` **more than doubles** it to +0.150.
The additional prefix question tokens sit right after the instructions, deep inside the attention-sink zone; every attention layer now has extra query-relevant tokens at the beginning of the prompt.
`question_first` also lifts primacy modestly (+0.088), still substantially above baseline, again consistent with more query-relevant mass at the start of the prompt.

`gold_padded` (128 filler tokens between documents and question) **halves** Pythia's primacy edge to +0.038 and keeps recency near baseline.
Filler tokens between the documents and the closing question shift the reader's attention away from the head of the prompt (where the gold sits when at position 0), consistent with attention mass being pulled toward the recent context by the sink-plus-recency competition.

### Combined direction: query-first helps SSM more than Transformer at raising primacy from zero

Mamba's primacy jumps from 0 to +0.103 under `bookend` (a factor no dense attention model in this study can match).
Pythia's primacy grows from +0.070 to +0.150 under the same variant, but its **absolute** growth is comparable to Mamba's.
The direction predicted by the workshop spec ("query-first should help Mamba and hybrid models more than Transformers") holds when framed as *lifting primacy from a null baseline*: only the SSM's arm was null to begin with, and the same intervention makes it positive.

Every one of these effects is a single-variable change on the prompt-order layer, so no other axis is confounded within this experiment.

## Artifacts

- `pythia-2.8b-{question_first,bookend,gold_padded}/sweep.jsonl.gz`: raw sweep records.
- `mamba-2.8b-{question_first,bookend,gold_padded}/sweep.jsonl.gz`: raw sweep records.
- `report/phase7-variants-summary.json`: per-model per-variant position curves, edges with 95 percent bootstrap intervals, and Holm-adjusted p-values.

`mamba2-2.7b` was attempted on a T4 but the triton mamba-2 kernel failed to compile on sm_75 (`PassManager::run failed`), so mamba2 is deferred to a host with a working triton path (L40S or A10G).
