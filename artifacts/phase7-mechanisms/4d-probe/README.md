# Phase 7 sub-experiment 4d: utilisation vs storage

A balanced linear probe for gold position (edge vs middle) trained on frozen hidden states.
If a probe recovers where the gold document sits while QA accuracy is not U-shaped for that model, the model *stores* the location but does not *use* it.

## Setup

- Hidden states captured with a single forward pass per (question, gold position), last prompt-token vector, at one layer fixed before the probe was fit.
  - `pythia-2.8b` at layer 16 (of 32 blocks).
  - `mamba-2.8b` at layer 32 (of 64 blocks).
  - Both are the mid-depth layer, chosen by rule (half the block count) before any probe was run, honoring the "choose the probe layer before viewing final results" requirement.
- 200 exploratory questions, all ten gold positions, so 1,200 edge/middle samples per model after dropping the four non-edge non-middle positions.
- Probe: numpy logistic regression, class-balanced, grouped 5-fold by question so a question's edge and middle vectors never straddle the train/test split.
- Control: the same probe fit on shuffled labels.

## Results

| Model | Probe layer | Probe accuracy | Shuffled-label control | QA primacy edge (Phase 2) |
|---|---|---|---|---|
| pythia-2.8b | 16 | 0.603 | 0.484 | +0.052 (Holm < 1e-4) |
| mamba-2.8b | 32 | 0.648 | 0.508 | -0.001 (Holm 0.914) |

The shuffled-label control sits at the 0.5 chance line for both models, so the real-label accuracy is a genuine signal, not a fitting artifact.

## Reading the result

Both models encode gold position in their mid-depth hidden state above chance: 0.60 for Pythia, 0.65 for Mamba.
The decisive contrast is Mamba.
Mamba-2.8B carries the *strongest* linear position signal of the two (0.648 vs Pythia's 0.603) yet has a *null* QA primacy edge (Phase 2: -0.001, not distinguishable from zero).
The location of the gold document is linearly decodable from Mamba's representation, but that knowledge does not translate into a position-dependent accuracy advantage at the edges.

This is the utilisation-not-storage signature the Phase 7 spec predicted: the model knows where the evidence is and still fails to preferentially use edge evidence.
For Pythia the picture is consistent but weaker: it both encodes position (0.60) and shows a primacy arm (+0.052), so its stored location does map onto a utilisation asymmetry.

The clean statement is that storage of position is not the bottleneck.
Both architectures store it; only the transformer converts stored position into an edge-accuracy advantage.
That points the primacy mechanism at *how* position information is used downstream (the attention-sink read on the Transformer side), not at whether the models can locate the gold document at all.

## Limits

- The probe reads one layer per model. A model could encode position more strongly at another depth; the layer was fixed by rule before fitting to avoid selecting on the outcome, so these numbers are honest but not the maximum achievable.
- Probe accuracy near 0.60-0.65 is a modest signal; the finding is the *direction* (Mamba encodes at least as well as Pythia while showing no primacy), not the absolute decodability.

## Artifacts

- `pythia-2.8b-layer16.jsonl.gz`, `mamba-2.8b-layer32.jsonl.gz`: raw hidden-state records, one per (question, gold position).
- `pythia-2.8b-layer16-probe.json`, `mamba-2.8b-layer32-probe.json`: probe accuracy, shuffled control, per-fold accuracy, sample and feature counts.
