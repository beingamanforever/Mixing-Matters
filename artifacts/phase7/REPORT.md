# Phase 7: cumulative analysis

## The Question

Does the accuracy-versus-evidence-position curve track the architecture, and does that pattern survive changes in scale, training data, and task?

- The phases use a common position-intervention harness, but several model-family and task comparisons change more than one factor.
- Read together, they measure where position curves differ and define which causal explanations remain unresolved.

> This report is the cumulative read of those phases.

## The phases

| Phase | Question                                       | Held fixed                            | Changed                                    | Host    | Status                                        |
| ----- | ---------------------------------------------- | ------------------------------------- | ------------------------------------------ | ------- | --------------------------------------------- |
| 1     | Does the pipeline detect a position effect?    | one model, one task                   | gold position, floor, ceiling              | A10G    | done                                          |
| 2     | Do the model families have different curves?   | corpus, approximate size, context     | mixer, depth, position setup, checkpoint    | A40     | done                                          |
| 4     | Does the architecture gap move with size?      | corpus, tokenizer, host               | parameter count, five matched pairs        | A40     | done                                          |
| 5     | Does training data move the curve?             | architecture, size, host              | pretraining corpus                         | A10G    | done                                          |
| 6     | Does the curve survive a change of task?       | models, host, context length          | multi-document QA to RULER `niah_single_1` | T4      | done                                          |
| 3     | Do released pure and hybrid checkpoints differ? | corpus, tokenizer, scale, depth, host | attention and MLP composition               | A40     | done; clean rerun reproduced summary          |
| 8     | Do production-scale families show the pattern? | none (descriptive)                    | Nemotron-H, Llama-3.1, Qwen                | L40S    | done                                          |

## Setup and phase overview

Every phase uses the same harness: the lost-in-the-middle multi-document QA dataset (10 documents, GPT-NeoX-20B tokenizer), greedy decoding at temperature 0, 32 new tokens, seed 240521, bfloat16, the gold document placed at every position 0 through 9, plus a closed-book floor and an oracle ceiling per question.

The two edges are defined once and reused everywhere: **primacy** is mean accuracy at positions 0,1 minus positions 4,5; **recency** is mean accuracy at positions 8,9 minus positions 4,5. Both are tested with a paired bootstrap over question bundles (10,000 resamples, question = unit of analysis) and Holm correction across the two edges.

Phase 6 swaps the dataset for RULER `niah_single_1` with the needle at ten deterministic depths (mirroring the ten gold positions), the noise count held fixed within a length, and the same floor and ceiling anchors. Two deliberate departures from running RULER's own script: the needle is placed deterministically instead of at random depth, and the key word comes from a vendored list instead of `wonderwords`, so instances stay reproducible.

The models are `pythia-2.8b`, `mamba-2.8b`, `mamba2-2.7b`, the five matched Pythia-Mamba size pairs from 130m to 2.8b, and the three Phase 8 production systems `nvidia/Nemotron-H-8B-Base-8K`, `meta-llama/Llama-3.1-8B`, and `Qwen/Qwen2.5-7B`. Every record carries the model revision, execution path, dataset checksum, and software versions; the Mamba models run on the pinned CUDA kernel path and the runner raises rather than falling back to the reference path.

## Findings

### The primacy arm differs by family and appears at capable scales

- Phase 2 ran three 2.8B models on one A40. Pythia has a primacy edge of +0.0519 (interval +0.0319 to +0.0719, Holm p below 0.0001).

- Mamba-1 has none (−0.0013, Holm p 0.914).
- Mamba-2 has a small negative one (−0.0181, Holm p 0.023).
- This is the contrast to the curve: the Pythia-vs-Mamba interaction excludes zero for both Mamba models (mamba-1 −0.0531, mamba-2 −0.0700, both Holm p < 0.0001), and the two Mamba models do not differ from each other.

![Phase 2 position curves](figures/artifacts/phase2/report/position-curves.png)
![Phase 2 position edges](figures/artifacts/phase2/report/position-edges.png)

> Phase 2 numbers are in `artifacts/phase2/report/phase2-summary.json`; figures in `artifacts/phase2/report/`.

- Phase 4 swept five matched size pairs on one A40 and answered how that primacy gap moves with size.
- Using Pythia minus Mamba as the sign convention, the primacy difference is indistinguishable from zero at the two smallest pairs (−0.0006 at 130m-160m, +0.0025 at 370m-410m) and then appears at 790m-1b (+0.0619) and stays at roughly the same size through 1.4b and 2.8b (+0.0694, +0.0531).

- The endpoint change is +0.054 from the smallest to the largest pair.
  - Pythia's own primacy edge is +0.013 and not significant at 410m, then +0.052 or larger from 1b up, while the measured Mamba edges stay at or below zero.
  - This pattern is descriptive because scale, depth, width, checkpoint identity, and task capability move together.

![Scale primacy gap](figures/artifacts/phase4/report/scale-primacy-gap.png)
![Scale curves](figures/artifacts/phase4/report/scale-curves.png)

- The 130m-160m pair sits at the Pythia capability floor (oracle ceiling 0.009 against mamba-130m's 0.283), so it carries no curve shape; it anchors the small end of the trend without being read as a shape comparison.

The remaining per-pair curves are below.

![130m-160m curves](figures/artifacts/phase4/130m-160m/report/position-curves.png)
![130m-160m edges](figures/artifacts/phase4/130m-160m/report/position-edges.png)
![370m-410m curves](figures/artifacts/phase4/370m-410m/report/position-curves.png)
![370m-410m edges](figures/artifacts/phase4/370m-410m/report/position-edges.png)
![790m-1b curves](figures/artifacts/phase4/790m-1b/report/position-curves.png)
![790m-1b edges](figures/artifacts/phase4/790m-1b/report/position-edges.png)
![1.4b-1.4b curves](figures/artifacts/phase4/1.4b-1.4b/report/position-curves.png)
![1.4b-1.4b edges](figures/artifacts/phase4/1.4b-1.4b/report/position-edges.png)
![2.8b-2.8b curves](figures/artifacts/phase4/2.8b-2.8b/report/position-curves.png)
![2.8b-2.8b edges](figures/artifacts/phase4/2.8b-2.8b/report/position-edges.png)
![Scale curve 130m-160m](figures/artifacts/phase4/report/scale-curve-130m-160m.png)
![Scale curve 370m-410m](figures/artifacts/phase4/report/scale-curve-370m-410m.png)
![Scale curve 790m-1b](figures/artifacts/phase4/report/scale-curve-790m-1b.png)
![Scale curve 1.4b-1.4b](figures/artifacts/phase4/report/scale-curve-1.4b-1.4b.png)
![Scale curve 2.8b-2.8b](figures/artifacts/phase4/report/scale-curve-2.8b-2.8b.png)

> Phase 4 numbers are in `artifacts/phase4/report/phase4-summary.json` (Pythia minus Mamba) and each pair's `report/phase2-summary.json`; figures in `artifacts/phase4/`.

- Phase 6 moved the same three models to a synthetic needle (NIAH: Needle In A Haystack) task at 1K and 2K.
- At 2K the Pythia primacy edge reappears, +0.12 (interval +0.05 to +0.20, Holm p < 0.0001), greater than the QA value of +0.05. At 1K it is +0.04 and does not clear the Holm threshold (p 0.084).
- The Mamba models show zero edges at both lengths, but their depth sweeps are at accuracy of 1.0 for every depth
  - So, this indicates that those zeros are task saturation, not evidence of position invariance.
  - The comparison in Phase 6 is Pythia's curve at 2K, where the middle positions dip (0.74, 0.78) while the edges stay near 1.0.

![Phase 6 position curve 1024](figures/artifacts/phase6/report/position-curve-1024.png)
![Phase 6 position curve 2048](figures/artifacts/phase6/report/position-curve-2048.png)
![Phase 6 position edges 1024](figures/artifacts/phase6/report/position-edges-1024.png)
![Phase 6 position edges 2048](figures/artifacts/phase6/report/position-edges-2048.png)
![Phase 6 task comparison 1024](figures/artifacts/phase6/report/task-comparison-1024.png)
![Phase 6 task comparison 2048](figures/artifacts/phase6/report/task-comparison-2048.png)

> Phase 6 numbers are in `artifacts/phase6/report/phase6-summary.json`; figures in `artifacts/phase6/report/`.

### The recency arm is shared by both architectures

- Phase 2: all three models have a positive recency edge, +0.0750 (Pythia), +0.0769 (Mamba-1), +0.1050 (Mamba-2), all Holm p < 0.0001.

- Phase 4: the recency difference between the two families is stable across size, but "stable" hides a small-model effect. At the two smallest pairs the difference is large and negative (−0.0419 at 130m-160m, −0.0456 at 370m-410m, both Holm p below 0.0001) and coincides with much weaker Pythia task capability (pythia-160m ceiling 0.009; pythia-410m recency +0.033 against mamba-370m +0.078). Once both can answer, from 790m-1b on, the difference sits at +0.0087 and stays near zero through 2.8b (−0.0019). The observed Mamba recency lead is concentrated in the weak small-model pairs and disappears once Pythia can answer.

- Phase 6: the Pythia recency edge reappears at 2K (+0.10, Holm p 0.002), matching the QA sign.

Recency is present in both measured families across the capable comparisons executed here.

### One corpus contrast changes level without a detectable shape change

- Phase 5 holds the model architecture and approximate scale fixed while comparing Pile and SlimPajama checkpoints, so pretraining corpus and checkpoint identity move together.

- The corpus effect on the primacy edge is −0.007 (interval −0.030 to +0.016, Holm p 0.57) and on the recency edge −0.016 (interval −0.044 to +0.012, Holm p 0.50). Both intervals span zero.

The observed overall level is higher for the SlimPajama checkpoint by about 0.05 at every position, the floor changes from 0.11 to 0.27, and the ceiling changes from 0.615 to 0.625.
The floor shift is consistent with different closed-book knowledge, but the contrast does not isolate memorization or rule out a smaller edge-shape effect.

![Phase 5 corpus effect](figures/artifacts/phase5/report/corpus-effect.png)
![Phase 5 position curves](figures/artifacts/phase5/report/position-curves.png)
![Phase 5 position edges](figures/artifacts/phase5/report/position-edges.png)

> Phase 5 numbers are in `artifacts/phase5/report/phase5-summary.json`; figures in `artifacts/phase5/report/`.

> In the executed comparisons, the family contrast has a detectable primacy interaction while the corpus edge contrasts do not clear the prespecified test.

## Consolidated numbers

- 95 percent intervals shown as [low, high]; Holm p-values are adjusted across the two edge tests. Phase 4 rows report Pythia minus Mamba differences, not single-model edges, and are marked "diff".

| Phase | Model / pair                       | Host | Primacy edge                                   | Recency edge                            | Floor / ceiling                  |
| ----- | ---------------------------------- | ---- | ---------------------------------------------- | --------------------------------------- | -------------------------------- |
| 1     | pythia-2.8b, 200 questions         | A10G | gold_first − gold_middle +0.115 [0.055, 0.175] | not run                                 | 0.095 / 0.650                    |
| 2     | pythia-2.8b                        | A40  | +0.0519 [0.0319, 0.0719], Holm < 1e-4          | +0.0750 [0.0537, 0.0969], Holm < 1e-4   | 0.091 / 0.640                    |
| 2     | mamba-2.8b                         | A40  | −0.0013 [−0.0162, +0.0137], Holm 0.914         | +0.0769 [0.0575, 0.0969], Holm < 1e-4   | 0.115 / 0.615                    |
| 2     | mamba2-2.7b                        | A40  | −0.0181 [−0.0338, −0.0025], Holm 0.023         | +0.1050 [0.0856, 0.1244], Holm < 1e-4   | 0.128 / 0.619                    |
| 4     | 130m-160m, diff (P−M)              | A40  | −0.0006 [−0.0094, +0.0075], Holm 0.911         | −0.0419 [−0.0556, −0.0287], Holm < 1e-4 | P 0.000 / 0.009, M 0.030 / 0.283 |
| 4     | 370m-410m, diff (P−M)              | A40  | +0.0025 [−0.0162, +0.0206], Holm 0.793         | −0.0456 [−0.0687, −0.0237], Holm < 1e-4 | P 0.029 / 0.358, M 0.044 / 0.479 |
| 4     | 790m-1b, diff (P−M)                | A40  | +0.0619 [0.0419, 0.0831], Holm < 1e-4          | +0.0087 [−0.0150, +0.0325], Holm 0.481  | P 0.050 / 0.496, M 0.066 / 0.539 |
| 4     | 1.4b-1.4b, diff (P−M)              | A40  | +0.0694 [0.0462, 0.0931], Holm < 1e-4          | −0.0225 [−0.0506, +0.0050], Holm 0.111  | P 0.060 / 0.594, M 0.093 / 0.554 |
| 4     | 2.8b-2.8b, diff (P−M)              | A40  | +0.0531 [0.0281, 0.0775], Holm < 1e-4          | −0.0019 [−0.0294, +0.0262], Holm 0.920  | P 0.091 / 0.640, M 0.115 / 0.615 |
| 5     | corpus effect, mamba-2.8b          | A10G | −0.007 [−0.030, +0.016], Holm 0.57             | −0.016 [−0.044, +0.012], Holm 0.50      | 0.11 → 0.27 / 0.615 → 0.625      |
| 6     | pythia-2.8b at 1K                  | T4   | +0.04 [0.00, 0.09], Holm 0.084                 | +0.04 [0.01, 0.08], Holm 0.058          | 0.00 / 0.98                      |
| 6     | pythia-2.8b at 2K                  | T4   | +0.12 [0.05, 0.20], Holm < 1e-4                | +0.10 [0.04, 0.17], Holm 0.002          | 0.00 / 0.98                      |
| 6     | mamba-2.8b, mamba2-2.7b, 1K and 2K | T4   | 0.00 at both lengths                           | 0.00 at both lengths                    | 0.00 / 1.00 and 0.78 (saturated) |
| 8     | nemotron-h-8b                      | L40S | +0.067 [+0.044, +0.091], Holm < 1e-4           | +0.029 [+0.008, +0.049], Holm 0.006     | 0.311 / 0.588                    |
| 8     | llama-3.1-8b                       | L40S | +0.076 [+0.057, +0.096], Holm < 1e-4           | +0.018 [+0.000, +0.034], Holm 0.051     | 0.315 / 0.782                    |
| 8     | qwen2.5-7b                         | L40S | +0.041 [+0.021, +0.061], Holm 0.0008           | +0.008 [-0.012, +0.027], Holm 0.474     | 0.280 / 0.815                    |

- Phase 2 interactions, 2.8B (Mamba minus Pythia): primacy mamba-1 −0.0531 [−0.0775, −0.0281], mamba2 −0.0700 [−0.0950, −0.0450], both Holm p below 0.0001; mamba-1 minus mamba2 +0.0169 [−0.0044, +0.0381], Holm p 0.125. Recency: mamba-1 − pythia +0.0019 [−0.0262, +0.0294], Holm p 0.920; mamba2 − pythia +0.0300 [0.0038, 0.0562], Holm p 0.026.

- The source of truth for every row is that phase's `*-summary.json` on main (paths listed under the findings above); this table is a copy for reading, not the authoritative store.

## Checks

- The key-value retrieval positive control passes its gate for Pythia (edge +0.38) and Mamba-2 (edge +0.65) and does not pass for Mamba-1 (edge +0.00, consistent with a state size of 16 that cannot store thirty pairs).

- The control is recorded and does not gate any sweep; the Phase 1 calibration demonstrates that the harness can detect the imposed position effect in that executed setting.

![Phase 1 KV position curve](figures/artifacts/phase1/figures/kv-position-curve.png)
![Phase 1 condition accuracy](figures/artifacts/phase1/figures/phase1-condition-accuracy.png)

> Phase 1 numbers are in `artifacts/phase1/summary.json` and `artifacts/phase1/figures/figures-summary.json`.

- All Phase 2 and Phase 4 models run on one A40 so hardware is not a second variable. The A10G sweep used during Phase 1 development diverges from the A40 run on 15 of 1,741 shared records, so a cross-host comparison would be a second changing variable.
- Phase 6 runs on a Tesla T4, the first host below Ampere, where bf16 has no dedicated tensor-core path. The within-phase task and length contrasts are on one execution path, but absolute T4 accuracy is not comparable to the A10G and A40 numbers.
- Prompt length is invariant across positions within every phase; floor and ceiling are attached to every record.
- Phase 6 saturation: the Mamba curves sit at accuracy 1.0 at every depth, so the phase cannot test Mamba position invariance; it tests whether the transformer primacy edge reproduces on a synthetic task, and at 2K it does.

## Limits

- Phase 2 and Phase 4 compare different architectures at matched size, but the families differ in depth, state size, and positional encoding, and capability changes along the size axis.
- The 130m-160m pair is at the Pythia capability floor and carries no curve shape.
- The Phase 4 verdict compares the smallest and largest endpoints, not a fitted slope.
- The phase 6 is underpowered relative to the QA phases and its edges are small and, for Pythia at 1K, not significant after Holm correction.
- The corpus result in Phase 5 is a null; a larger effect could exist below the resolution of 800 questions.

## Phase 3 (done)

Question: do the released pure and hybrid Mamba-2 checkpoints have different position curves?

The 8B pure and hybrid Mamba-2 checkpoints are matched on data, tokenizer, scale, depth, positional setup, host, and execution path at 4K context.
The released checkpoints differ in both attention and MLP composition, so the contrast is composite rather than attention-only.
The hybrid-minus-pure primacy effect is +0.0188 with 95% interval [-0.0056, +0.0444] and Holm p=0.1442.
The original producing manifest records a dirty source tree, and a later clean rerun at commit `33d6bb5` reproduced the summary byte-for-byte.
The held-out confirmatory split remains unopened.

## Phase 8 (done)

Question: do production-scale families show the same position-curve shape?

This is a descriptive comparison, not a matched control.
The three systems - `nvidia/Nemotron-H-8B-Base-8K` (hybrid Mamba-2 plus attention), `meta-llama/Llama-3.1-8B`, and `Qwen/Qwen2.5-7B` - differ from each other on architecture, corpus, token count, tokenizer, alignment status, depth, and positional encoding at once.
The sweep still uses the same 800-question, 10-position harness and the same primacy and recency edges.
All three ran on one rented L40S 46 GB.
Nemotron-H ran the eager attention implementation (its SSM blocks dispatch through `mamba-ssm` and `causal-conv1d`); Llama and Qwen ran the SDPA attention implementation because eager on 8B dense attention is prohibitively slow on a single L40S and Phase 8 is descriptive, not a matched control.
Every raw record carries the actual `execution_path` and `attention_implementation` it used.

### Findings

- Every one of the three systems has a positive, Holm-significant primacy edge at 800 questions: Nemotron-H +0.067 (Holm p < 1e-4), Llama-3.1 +0.076 (Holm p < 1e-4), Qwen2.5 +0.041 (Holm p 0.0008).
The primacy pattern appears in all three production systems on this harness.
- The recency arm splits by system.
Nemotron-H has a clean recency edge (+0.029, Holm p 0.006); Llama-3.1 is marginal (+0.018, Holm p 0.051); Qwen2.5 is a null (+0.008, Holm p 0.474).
Phase 2 measured recency on every 2.8B model.
At production scale the recency arm is clearly non-zero only for Nemotron-H in these per-system tests.
Because every system differs on many factors, this pattern cannot be attributed to its SSM blocks.
- The one pairwise interaction that clears Holm is Llama minus Qwen on primacy, +0.036 (interval +0.009 to +0.063, Holm p 0.016).
Llama's primacy arm sits about 3.6 points above Qwen's despite both being dense attention with RoPE.
Within the dense-attention branch, the Llama-minus-Qwen difference shows that a shared coarse architecture label does not determine the measured primacy magnitude.
The phase does not identify which other changing factor explains the difference.
- Ceilings order Qwen (0.815) > Llama (0.782) > Nemotron-H (0.588); floors sit near 0.28-0.32 for all three.
Nemotron-H's weaker ceiling means its edges are read against a lower headroom, but its primacy edge is still the largest per unit of ceiling headroom.

![Phase 8 position curves](figures/artifacts/phase8/report/position-curves.png)
![Phase 8 position edges](figures/artifacts/phase8/report/position-edges.png)

> Phase 8 numbers are archived at `artifacts/phase7/phase8/phase8-summary.json`, mirrored from `artifacts/phase8/report/phase8-summary.json` on the `phase-8` branch; the two figures above are the report emitted by `phase8-report` from all three raw sweeps.

### How it lines up with Phases 2 and 4

- Phase 2 found a family-by-position difference at 2.8B and no positive primacy arm in either Mamba variant.
Phase 4 found a near-zero Pythia-minus-Mamba gap in the two weakest pairs and a positive gap in the three capable pairs.
Phase 8's three 8B systems all show primacy; two of the three are dense attention with RoPE, matching the Phase 2 and 4 direction, and the third is a hybrid whose primacy edge is comparable in size.
- Phase 8 is not evidence about the Phase 2 confound.
Every axis moves between its systems, so the fact that Nemotron-H's primacy estimate is close to Llama's does not say attention layers cause primacy.
Phase 3 is complete, but its paired effect is statistically uncertain and the released checkpoints differ in both attention and MLP composition.
- Read as a prevalence check: at production scale, on the same 10-document harness, all three full systems have positive primacy estimates.
The unmatched comparison does not isolate what produced them.

### Limits

- Nemotron-H ran eager attention; Llama and Qwen ran SDPA.
Two attention kernels are in scope within Phase 8, so the eager-vs-SDPA path is a second changing variable when Nemotron-H is compared against the others.
That trade-off is documented on every raw record and prevents attributing between-system differences to one factor.
- Nemotron-H fails the key-value positive control (mean kv accuracy ~0.32 across the ten positions, versus Pythia and Mamba-2 near full retrieval in Phases 1 and 2).
Its weaker ceiling on QA co-occurs with weaker retrieval capacity, but the non-gating control cannot distinguish a model limitation from every possible pipeline issue.
- The prompt-token span tolerance was raised from 2 to 8 for Phase 8 because the three tokenizers each merge document boundaries differently.
The recorded per-question span is a few tokens across all runs, well inside the raised limit.

## Next steps

Freeze the analysis before opening the 1,855-question confirmatory split.
If a causal attention claim is retained, test it with an intervention that changes attention without also changing MLP composition and repair the no-op Nemotron-H sink block.
The current standing claim is narrower: the exploratory Pile comparison shows a family-by-position interaction near 2.8B, the interaction is present in the three capable scale pairs, Pythia primacy appears on the 2K synthetic task, and three unmatched production systems also show primacy on the QA harness.
