# Phase 7 sub-experiment 4d: position decodability probe

A balanced linear probe for gold position (edge vs middle) trained on frozen hidden states.
Above-chance probe accuracy shows that this position label is linearly decodable from the measured representation.
It does not establish that the model stores the answer-bearing content or uses the decoded position during generation.

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

Gold position is linearly decodable from both measured mid-depth representations above chance: 0.603 for Pythia and 0.648 for Mamba.
In these selected layers, Mamba has the higher probe accuracy while its Phase 2 primacy edge is null (-0.001, not distinguishable from zero).
This pattern motivates a hypothesis that position decodability and edge-dependent answer accuracy can dissociate.
The probe does not determine whether answer content is stored, whether the decoded feature is used downstream, or what causes the family difference.

## Limits

- The probe reads one layer per model. A model could encode position more strongly at another depth; the layer was fixed by rule before fitting to avoid selecting on the outcome, so these numbers are honest but not the maximum achievable.
- Probe accuracy near 0.60-0.65 is a modest signal from one selected layer per model.
- Comparing probe accuracies across architectures is descriptive because the representations, dimensions, and selected layers differ.

## Artifacts

- `pythia-2.8b-layer16.jsonl.gz`, `mamba-2.8b-layer32.jsonl.gz`: raw hidden-state records, one per (question, gold position).
- `pythia-2.8b-layer16-probe.json`, `mamba-2.8b-layer32-probe.json`: probe accuracy, shuffled control, per-fold accuracy, sample and feature counts.
