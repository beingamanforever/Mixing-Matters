# Phase 7: cumulative analysis

## The Question

Does the accuracy-versus-evidence-position curve track the architecture, and does that pattern survive changes in scale, training data, and task?

- The phases 1 through 6 each changed one thing and held everything else fixed. Read together they answer the project's question: at the same size and context, do a transformer and a sequence-mixing architecture place the same value on where the evidence sits?

> This report is the cumulative read of those phases.

## The phases, one variable at a time

| Phase | Question                                       | Held fixed                            | Changed                                    | Host    | Status                                        |
| ----- | ---------------------------------------------- | ------------------------------------- | ------------------------------------------ | ------- | --------------------------------------------- |
| 1     | Does the pipeline detect a position effect?    | one model, one task                   | gold position, floor, ceiling              | A10G    | done                                          |
| 2     | Does the architecture move the curve?          | corpus, size, context, seed, decoding | sequence-mixing architecture               | A40     | done                                          |
| 4     | Does the architecture gap move with size?      | corpus, tokenizer, host               | parameter count, five matched pairs        | A40     | done                                          |
| 5     | Does training data move the curve?             | architecture, size, host              | pretraining corpus                         | A10G    | done                                          |
| 6     | Does the curve survive a change of task?       | models, host, context length          | multi-document QA to RULER `niah_single_1` | T4      | done                                          |
| 3     | Do attention layers cause the primacy edge?    | matched Mamba-2 vs hybrid, 8B, 4K     | attention layers                           | in progress | code on `origin/phase-3`                      |
| 8     | Do production-scale families show the pattern? | none (descriptive)                    | Nemotron-H, Llama-3.1, Qwen                | in progress | 2/3 models run on `origin/phase-8`, Llama running (L40S) |

## Setup and phase overview

Every phase uses the same harness: the lost-in-the-middle multi-document QA dataset (10 documents, GPT-NeoX-20B tokenizer), greedy decoding at temperature 0, 32 new tokens, seed 240521, bfloat16, the gold document placed at every position 0 through 9, plus a closed-book floor and an oracle ceiling per question.

The two edges are defined once and reused everywhere: **primacy** is mean accuracy at positions 0,1 minus positions 4,5; **recency** is mean accuracy at positions 8,9 minus positions 4,5. Both are tested with a paired bootstrap over question bundles (10,000 resamples, question = unit of analysis) and Holm correction across the two edges.

Phase 6 swaps the dataset for RULER `niah_single_1` with the needle at ten deterministic depths (mirroring the ten gold positions), the noise count held fixed within a length, and the same floor and ceiling anchors. Two deliberate departures from running RULER's own script: the needle is placed deterministically instead of at random depth, and the key word comes from a vendored list instead of `wonderwords`, so instances stay reproducible.

The models are `pythia-2.8b`, `mamba-2.8b`, `mamba2-2.7b`, and the five matched Pythia-Mamba size pairs from 130m to 2.8b. Every record carries the model revision, execution path, dataset checksum, and software versions; the Mamba models run on the pinned CUDA kernel path and the runner raises rather than falling back to the reference path.

## Findings

### The primacy arm is architecture-specific and scale-emergent

- Phase 2 ran three 2.8B models on one A40. Pythia has a primacy edge of +0.0519 (interval +0.0319 to +0.0719, Holm p below 0.0001).

- Mamba-1 has none (−0.0013, Holm p 0.914).
- Mamba-2 has a small negative one (−0.0181, Holm p 0.023).
- This is the contrast to the curve: the Pythia-vs-Mamba interaction excludes zero for both Mamba models (mamba-1 −0.0531, mamba-2 −0.0700, both Holm p < 0.0001), and the two Mamba models do not differ from each other.

![Phase 2 position curves](figures/artifacts/phase2/report/position-curves.png)
![Phase 2 position edges](figures/artifacts/phase2/report/position-edges.png)

> Phase 2 numbers are in `artifacts/phase2/report/phase2-summary.json`; figures in `artifacts/phase2/report/`.

- Phase 4 swept five matched size pairs on one A40 and answered how that primacy gap moves with size.
- Using Pythia minus Mamba as the sign convention, the primacy difference is indistinguishable from zero at the two smallest pairs (−0.0006 at 130m-160m, +0.0025 at 370m-410m) and then appears at 790m-1b (+0.0619) and stays at roughly the same size through 1.4b and 2.8b (+0.0694, +0.0531).

- The endpoint verdict is **grows**: a change of +0.054 from the smallest to the largest pair.
  - Reason being Pythia's own primacy edge strengthens with size (from +0.013, not significant at 410m, to +0.052 and above from 1b up) while Mamba's stays at or below zero at every size.

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

- Phase 4: the recency difference between the two families is stable across size, but "stable" hides a small-model effect. At the two smallest pairs the difference is large and negative (−0.0419 at 130m-160m, −0.0456 at 370m-410m, both Holm p below 0.0001), which just means Mamba can answer where Pythia at the same size barely can (pythia-160m ceiling 0.009; pythia-410m recency +0.033 against mamba-370m +0.078). Once both can answer, from 790m-1b on, the difference sits at +0.0087 and stays near zero through 2.8b (−0.0019). So the Mamba recency lead is really a small-model lead — it shows up where Pythia is too weak to read the question and disappears once Pythia catches up.

- Phase 6: the Pythia recency edge reappears at 2K (+0.10, Holm p 0.002), matching the QA sign.

Recency is the arm both architectures share at every size, corpus, and task where the model can answer at all.

### Training data lifts accuracy, not shape

- Phase 5 held the architecture fixed — mamba-2.8b on the Pile against the identical architecture on SlimPajama — and changed only the pretraining corpus.

- The corpus effect on the primacy edge is −0.007 (interval −0.030 to +0.016, Holm p 0.57) and on the recency edge −0.016 (interval −0.044 to +0.012, Holm p 0.50). Both intervals span zero.

What the corpus did change was the overall level. SlimPajama lifts the whole curve by about 0.05 at every position, the floor goes from 0.11 to 0.27, and the ceiling barely moves (0.615 to 0.625). That reads as the corpus contributing more memorized answers rather than changing how the model uses position.

![Phase 5 corpus effect](figures/artifacts/phase5/report/corpus-effect.png)
![Phase 5 position curves](figures/artifacts/phase5/report/position-curves.png)
![Phase 5 position edges](figures/artifacts/phase5/report/position-edges.png)

> Phase 5 numbers are in `artifacts/phase5/report/phase5-summary.json`; figures in `artifacts/phase5/report/`.

> The contrast between the two changes is the point: swapping the architecture moved the primacy edge; swapping the corpus did not.

## Consolidated numbers

- 95 percent intervals shown as [low, high]; Holm p-values are adjusted across the two edge tests. Phase 4 rows report Pythia minus Mamba differences, not single-model edges, and are marked "diff".

| Phase | Model / pair                       | Host | Primacy edge                                   | Recency edge                            | Floor / ceiling                  |
| ----- | ---------------------------------- | ---- | ---------------------------------------------- | --------------------------------------- | -------------------------------- |
| 1     | pythia-2.8b, 200 questions         | A10G | gold_first − gold_middle +0.115 [0.055, 0.175] | —                                       | 0.095 / 0.650                    |
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

- Phase 2 interactions, 2.8B (Mamba minus Pythia): primacy mamba-1 −0.0531 [−0.0775, −0.0281], mamba2 −0.0700 [−0.0950, −0.0450], both Holm p below 0.0001; mamba-1 minus mamba2 +0.0169 [−0.0044, +0.0381], Holm p 0.125. Recency: mamba-1 − pythia +0.0019 [−0.0262, +0.0294], Holm p 0.920; mamba2 − pythia +0.0300 [0.0038, 0.0562], Holm p 0.026.

- The source of truth for every row is that phase's `*-summary.json` on main (paths listed under the findings above); this table is a copy for reading, not the authoritative store.

## Checks

- The key-value retrieval positive control passes its gate for Pythia (edge +0.38) and Mamba-2 (edge +0.65) and does not pass for Mamba-1 (edge +0.00, consistent with a state size of 16 that cannot store thirty pairs).

- The control is recorded and does not gate any sweep; Phase 1 already proved the pipeline detects a real position effect.

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

## Phase 3 (in progress)

Question: does adding attention to a matched Mamba-2 architecture create the primacy edge?

This is the depth-and-attention contrast that Phase 2 names as its unresolved confound: an 8B pure Mamba-2 against an 8B hybrid that adds ~7% attention layers, matched on data, tokenizer, scale, depth, and positional encoding, at 4K context. It was designed to run in parallel with Phase 2, slotting between Phase 2 and Phase 4.

## Phase 8 (in progress)

Question: do production-scale transformer families show the same position-curve shape?

This is a descriptive comparison, not a matched control: Nemotron-H-8B-Base-8K (hybrid Mamba-2 plus attention), Llama-3.1-8B, and Qwen2.5-7B differ from each other on architecture, corpus, token count, tokenizer, alignment, depth, and positional encoding at once. The sweep still uses the same 800-question, 10-position harness and the same edges.

## Next steps

Add Phase 3 and Phase 8 when the runs land, then restitch. Until then, the standing claim the phases isolate is: architecture changes the primacy arm, scale makes that difference appear, training data does not move it, and a second task reproduces it where the task is hard enough to measure.
