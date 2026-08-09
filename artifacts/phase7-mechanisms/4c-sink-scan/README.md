# Phase 7 sub-experiment 4c: attention-sink mass

Token-0 attention share per layer, measured on frozen forward passes over the baseline QA prompts.
The sink hypothesis predicts that sink mass tracks the primacy arm.

## Setup

- One forward pass per (question, gold position) with `output_attentions=True`, last-prompt-position attention row, averaged across heads, one value per layer.
- 200 exploratory questions x 10 gold positions per model.
- Pythia family at every registered size (160m, 410m, 1b, 1.4b, 2.8b) on the sink-scan hosts (L40S and two T4s).
- Attention share is guaranteed in `[0, 1]`; the measurement never generates a token.

## Results

Mean token-0 attention share, averaged over questions and gold positions:

| Model | Layers | Peak sink (layer) | Final-layer sink | Overall mean | Phase 2/4 primacy edge |
|---|---|---|---|---|---|
| pythia-160m | 12 | 0.122 (L7) | 0.003 | 0.029 | -0.001 (null) |
| pythia-410m | 24 | 0.255 (L10) | 0.034 | 0.109 | +0.013 (not sig) |
| pythia-1b | 16 | 0.140 (L7) | 0.089 | 0.082 | +0.052 (Holm < 1e-4) |
| pythia-1.4b | 24 | 0.173 (L17) | 0.139 | 0.103 | +0.053 (Holm < 1e-4) |
| pythia-2.8b | 32 | 0.455 (L31) | 0.455 | 0.110 | +0.052 (Holm < 1e-4) |

Sink mass is flat across gold positions within every model (token 0 is a fixed prompt position, so this is expected and confirms the measurement is reading a prompt-anchored sink, not something that moves with the gold document).

## Reading the result

- The model with a **null** primacy arm, pythia-160m, has by far the **lowest** sink mass (overall mean 0.029, final-layer 0.003).
Sink mass is essentially absent where the primacy arm is absent.
- Every model with a Holm-significant primacy arm (1b, 1.4b, 2.8b) carries a substantial sink, and pythia-2.8b, the largest model, develops an extreme final-layer sink of 0.455.
- The direction the sink hypothesis predicts holds: models that have a primacy arm have a sink, the model without a primacy arm has almost none, and the sink grows with scale in step with the primacy arm becoming significant (Phase 4's "grows and stabilizes" verdict).

The one wrinkle is pythia-410m: its overall-mean sink (0.109) is as high as pythia-2.8b's, yet its primacy edge is only +0.013 (not significant).
Its sink peaks at a mid layer (L10, 0.255) and collapses to 0.034 by the final layer, whereas the models with strong primacy carry the sink through to the last layers.
That suggests it is not raw sink mass but **late-layer** sink mass that couples to the primacy arm: a sink that is gone by the output layer does not steer the answer toward the head of the prompt.

## Hybrid model note

The intended companion measurement on `nemotron-h-8b` could not be produced: the NVIDIA Nemotron-H custom attention implementation does not return attention weights under `output_attentions=True` (the forward pass finishes without emitting any attentions), so the sink-mass scan yields no records for that model.
Measuring the hybrid's sink would require patching the custom attention forward to expose weights; it is left as follow-up work and does not affect the Pythia-family reading above.

## Artifacts

- `pythia-{160m,410m,1b,1.4b}.jsonl.gz`, `pythia-2.8b-baseline.jsonl.gz`: raw per-(question, position, layer) sink-mass records.
- `report/sink-mass-by-layer.png`: mean sink mass against layer index, one line per model.
- `report/sink-mass-summary.json`: per-model per-position per-layer means and question counts.
