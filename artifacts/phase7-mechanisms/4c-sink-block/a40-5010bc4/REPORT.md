# Nemotron-H sink-block ablation

## Outcome

The repaired intervention ran successfully on one NVIDIA A40 at repository commit `5010bc4b1908a74f90e7cc3345043e39099eeeb6`.
The intervention was not a no-op: it changed 1,094 of 2,400 paired model responses and 330 paired primary scores.
It did not remove the primacy arm in this 200-question experiment.

The baseline primacy edge was `+0.0650` with a 95 percent paired-bootstrap interval of `[+0.0150, +0.1150]` and Holm-adjusted `p = 0.0240`.
The blocked primacy edge was `+0.0500` with an interval of `[+0.0075, +0.0925]` and Holm-adjusted `p = 0.0436`.
The paired blocked-minus-baseline change was `-0.0150`, with an interval of `[-0.0725, +0.0400]` and Holm-adjusted `p = 0.9396`.
These data do not support the claim that blocking token 0 removes Nemotron-H's primacy arm.

## Fixed protocol

- Model: `nvidia/Nemotron-H-8B-Base-8K`.
- Revision: `94ea861e008c2dfced3e8e1302094024077aa04e`.
- Dataset: the repository's fixed 10-document Natural Questions split.
- Seed: `240521`.
- Prompt: Liu documents-then-question template.
- Decoding: greedy, temperature `0`, 32 new tokens.
- Prompt-length gate: maximum span of 8 tokens across gold positions.
- Full comparison: 200 questions per arm, producing 2,400 records per arm.
- Positive control: 500 key-value records.
- Hardware: one NVIDIA A40.

The full baseline and blocked arms use the same 200 source questions in the same condition order.
Question identifiers, source indices, conditions, gold positions, prompts, answers, prompt token counts, template, and model revision match pairwise.
There were no excluded records, incomplete question bundles, scoring failures, or failure sidecars.

## Intervention gate

The fixed one-question pilot produced 12 baseline and 12 blocked records.
Eleven of the 12 paired responses changed, while the oracle score remained `1.0` in both arms.
The blocked context completed without its runtime guard raising.
The implementation is designed to raise when a Nemotron-H attention module is not reached, modifies no legal query row, or produces a non-finite intercepted output.
The exact module, scaled dot-product attention, and modified-row counts were not serialized into the logs or records, so those counts cannot be independently audited from the preserved artifacts.

## Full results

Position accuracies, written as baseline followed by blocked, were:

- Position 0: `0.550`, `0.570`.
- Position 1: `0.445`, `0.495`.
- Position 2: `0.445`, `0.510`.
- Position 3: `0.430`, `0.465`.
- Position 4: `0.425`, `0.485`.
- Position 5: `0.440`, `0.480`.
- Position 6: `0.400`, `0.480`.
- Position 7: `0.405`, `0.480`.
- Position 8: `0.415`, `0.485`.
- Position 9: `0.470`, `0.465`.

The primacy edge was `+0.0650` at baseline and `+0.0500` when blocked, for a change of `-0.0150`.
The recency edge was `+0.0100` at baseline and `-0.0075` when blocked, for a change of `-0.0175`.
The closed-book floor was `0.330` at baseline and `0.310` when blocked, for a change of `-0.020`.
The oracle ceiling was `0.580` at baseline and `0.560` when blocked, for a change of `-0.020`.

The paired recency change was `-0.0175`, with an interval of `[-0.0650, +0.0275]` and Holm-adjusted `p = 0.9396`.
The positive control's overall accuracy was `0.316` across its 500 records.
The control confirms non-zero key-value capability but is not used to estimate the QA position effect.

## Interpretation boundary

The paired output changes establish that blocked execution was behaviorally distinct from the matched baseline.
The missing serialized hook telemetry limits independent audit of the intervention at the per-module level.
The full result does not establish that attention sinks are irrelevant, because it tests one specific intervention on one hybrid architecture with 200 exploratory questions.
It does establish that this token-0 block did not produce a statistically resolved reduction in the primacy edge under the fixed protocol.
The paper is intentionally unchanged in this phase so the result can be incorporated later with the other mechanism evidence.

## Preserved files

- `pilot/baseline.jsonl.gz` and `pilot/sink-blocked.jsonl.gz`: the fixed intervention gate.
- `full/positive-control.jsonl.gz`: 500 key-value control records.
- `full/baseline.jsonl.gz`: 2,400 matched baseline records.
- `full/sink-blocked.jsonl.gz`: 2,400 matched intervention records.
- `report/phase7-variants-summary.json`: the standard 10,000-resample Phase 7 position and edge summary.
- `logs/setup.log`, `logs/pilot.log`, and `logs/full.log`: preserved setup and execution logs.
- `pilot/COMPLETE` and `full/COMPLETE`: successful completion markers from the detached A40 workflows.
