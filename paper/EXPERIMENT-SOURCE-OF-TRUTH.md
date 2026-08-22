# Mixing Matters: experiment source of truth

This document is the paper-writing record of what is actually present in the repository.
It separates executed measurements from planned work, descriptive observations from causal claims, and artifact-backed results from prose that needs correction.
It was prepared on 2026-08-11 and reconciled with the committed artifacts on 2026-08-22.

## Repository state

This ledger describes committed experiment evidence rather than a transient local branch or worktree.
The 2026-08-22 reconciliation changed narrative records only and did not overwrite raw result files.

## Authority order

Use the raw JSONL or JSONL.GZ records first, then the machine-readable summary JSON, then the environment JSON, then the phase report and figure.
Treat the current LaTeX draft and older cumulative prose as a narrative draft rather than as the numerical authority.
The phase reports sometimes describe intended controls or old phase status that is not represented by a committed run.

Primary summary paths are:

- `artifacts/phase1/summary.json`.
- `artifacts/phase2/report/phase2-summary.json`.
- `artifacts/phase3/report/phase3-summary.json`.
- `artifacts/phase4/report/phase4-summary.json` and the five pair summaries below `artifacts/phase4/`.
- `artifacts/phase5/report/phase5-summary.json`.
- `artifacts/phase6/report/phase6-summary.json`.
- `artifacts/phase7-mechanisms/report/phase7-summary.json` and its sub-experiment summaries.
- `artifacts/phase8/report/phase8-summary.json`.

## Study design actually executed

The primary task is the Liu et al. Lost-in-the-Middle ten-document multi-document QA release.
The committed dataset contains 2,655 questions and has SHA-256 `192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9`.
Each question has one gold document and nine distractors.

The fixed split seed is `240521`.
The exploratory allocation is 800 questions.
The held-out confirmatory allocation is 1,855 questions.
All committed model results in this repository use the 800-question exploratory allocation.
The 1,855-question confirmatory allocation remains unspent.

For each exploratory question, the gold document is placed at positions 0 through 9 while the question, distractor set, prompt template, and intended prompt length are held fixed.
Each full sweep also contains a closed-book condition and a gold-only oracle condition.
The closed-book condition is the floor anchor for guessing and memorized knowledge.
The oracle condition is the ceiling anchor for answerability when retrieval is perfect.

The prompt follows the official Liu et al. documents-then-question order.
The prompt builders and primary `best_subspan_em` scorer are vendored from upstream commit `29b8a6d042ce29abccee3db1a73171a107d7e6af`.
Normalized exact match and first-line-only extraction are stored as sensitivity scores.

Generation is greedy with `do_sample=false`, temperature 0, `top_p=1`, unset `top_k`, one beam, 32 maximum new tokens, BF16 where supported, and fixed seed `240521`.
Mamba runs use one resolved execution path per phase and the runner refuses an unplanned fallback.
Every raw record stores prompts, generations, score variants, token counts, run ID, model revision, data checksum, execution path, and software metadata.

The primary edge definitions are:

- Primacy = mean accuracy at positions 0 and 1 minus mean accuracy at positions 4 and 5.
- Recency = mean accuracy at positions 8 and 9 minus mean accuracy at positions 4 and 5.

The question is the statistical unit.
Each bootstrap resample draws complete question bundles with all ten positions together.
The standard summaries use 10,000 paired bootstrap resamples and percentile 95% intervals.
Holm correction is applied across the primacy and recency tests within the reported contrast.

## Artifact integrity audit

The committed full QA sweeps contain 9,600 records per model: 800 closed-book records, 800 oracle records, and 8,000 gold-position records.
Every full sweep audited here has 800 complete question bundles and exactly 800 records at each gold position.
The full sweeps have no null primary scores and no excluded questions.
Prompt-token spans are at most one token within a question for the Phase 2, Phase 4, and Phase 5 full sweeps, and are zero or one in the Phase 3 and Phase 8 full sweeps inspected here.

Phase 7 variant sweeps contain 200 questions per model and 2,400 records per variant.
The RULER records contain 50 needle instances per model per length, or 1,200 records per model across the two lengths.
This is smaller than the 100-instance count described in the Phase 6 runbook and must be reported as the executed sample size.

The `artifacts/phase7-mechanisms/inputs/` files reuse raw sweeps from earlier phases.
Their repeated run IDs are copied-artifact provenance, not additional model executions.

The original environment manifests report these producing code revisions: Phase 1 `54dc62b`, Phase 2 `f41ccae`, Phase 3 `279b54c`, Phase 4 `2013199`, Phase 5 `6e3e83c`, and Phase 8 `c4f9f8c`.
Several manifests also record untracked setup or launcher files, and the original Phase 3 manifest records a dirty source tree with the Megatron runner and phase code uncommitted at run time.
A subsequent Phase 3 rerun at commit `33d6bb5`, with both producing repositories clean, reproduced the prior summary byte-for-byte.
The paper preserves both the original provenance caveat and the clean reproduction rather than describing every run as originally produced from a clean checkout.

## Results by phase

### Phase 1: end-to-end measurement calibration

Phase 1 used Pythia-2.8B on 200 exploratory questions, with 800 generation records.
The primary accuracies were 0.095 for closed book, 0.225 for gold at position 4, 0.340 for gold at position 0, and 0.650 for the oracle.
Gold-first minus gold-middle was `+0.115`, with 95% bootstrap interval `[+0.055, +0.175]`.

The key-value positive control used 50 instances and 500 generations.
Its position accuracies were `[0.94, 0.34, 0.20, 0.08, 0.18, 0.16, 0.08, 0.20, 0.42, 0.16]`.
The edge mean at slots 0 and 9 minus the middle mean at slots 4 and 5 was `+0.38`, so the pipeline detected a known position effect.

The ordering of the four Phase 1 conditions was unchanged under `best_subspan_em`, normalized exact match, and first-line-only scoring.
The tracer was replayed in separate processes with matching prompts, generations, scores, and token counts under the same seeds and pinned revisions.

A 50-row blinded audit sample and an audit key were generated.
The human categorization of formatting, extraction, hallucination, and truncation failures is not present in the repository.

### Phase 2: exploratory near-2.8B family comparison

Phase 2 ran Pythia-2.8B, Mamba-2.8B, and Mamba2-2.7B on the same 800 exploratory questions and one NVIDIA A40 execution host.

| Model | Primacy edge | 95% interval | Holm p | Recency edge | 95% interval | Holm p | Floor / ceiling |
|---|---:|---|---:|---:|---|---:|---:|
| Pythia-2.8B | +0.0519 | [+0.0319, +0.0719] | <0.0001 | +0.0750 | [+0.0537, +0.0969] | <0.0001 | 0.091 / 0.640 |
| Mamba-2.8B | -0.0013 | [-0.0162, +0.0137] | 0.914 | +0.0769 | [+0.0575, +0.0969] | <0.0001 | 0.115 / 0.615 |
| Mamba2-2.7B | -0.0181 | [-0.0338, -0.0025] | 0.023 | +0.1050 | [+0.0856, +0.1244] | <0.0001 | 0.128 / 0.619 |

The paired primacy interaction was Mamba-2.8B minus Pythia-2.8B = `-0.0531`, 95% interval `[-0.0775, -0.0281]`, Holm `p<0.0001`.
The paired primacy interaction was Mamba2-2.7B minus Pythia-2.8B = `-0.0700`, 95% interval `[-0.0950, -0.0450]`, Holm `p<0.0001`.
The Mamba-versus-Mamba primacy interaction was `+0.0169`, interval `[-0.0044, +0.0381]`, Holm `p=0.125`.

All three models have a positive recency edge.
The architecture contrast is not fully isolated because Pythia differs from the Mamba models in depth, positional encoding, state size, and exact checkpoint family.

The key-value control passed for Pythia and Mamba2-2.7B and failed for Mamba-2.8B.
The non-gating failure is consistent with the smaller recurrent state and does not invalidate the QA sweep because Phase 1 established that the instrument can detect a known position effect.

### Phase 3: matched pure versus hybrid Mamba-2 at 8B

Phase 3 ran two NVIDIA 8B checkpoints over the same 800 questions through the same bare-metal Megatron-LM backend and CUDA-kernel path.
The checkpoints share training data, tokenizer, approximate scale, depth, and positional-encoding setup, while the hybrid changes both attention and MLP composition.

| Model | Primacy edge | 95% interval | Holm p | Recency edge | 95% interval | Holm p | Floor / ceiling |
|---|---:|---|---:|---:|---|---:|---:|
| Pure Mamba-2 8B | +0.0081 | [-0.0088, +0.0244] | 0.362 | +0.0775 | [+0.0575, +0.0975] | <0.0001 | 0.2575 / 0.6513 |
| Hybrid Mamba-2 8B | +0.0269 | [+0.0075, +0.0469] | 0.007 | +0.0494 | [+0.0313, +0.0681] | <0.0001 | 0.2425 / 0.6163 |

The paired hybrid-minus-pure primacy effect was `+0.0188`, with 95% interval `[-0.0056, +0.0444]` and Holm `p=0.1442`.
The paired hybrid-minus-pure recency effect was `-0.0281`, with 95% interval `[-0.0550, -0.0013]` and Holm `p=0.0892`.

The correct claim is that the hybrid has a significant within-model primacy edge while the pure model does not, and that the paired effect estimate is suggestive but not statistically significant under the prespecified correction.
Phase 3 is a composite released-checkpoint contrast and does not independently prove that adding attention caused the primacy difference.

### Phase 4: scale and family trend

Five Pythia-Mamba size pairs were run on one A40.
The summary reports Pythia minus Mamba interaction edges.

| Pair | Primacy difference | 95% interval | Holm p | Recency difference | Holm p |
|---|---:|---|---:|---:|---:|
| 130M vs 160M | -0.0006 | [-0.0094, +0.0075] | 0.911 | -0.0419 | <0.0001 |
| 370M vs 410M | +0.0025 | [-0.0162, +0.0206] | 0.793 | -0.0456 | <0.0001 |
| 790M vs 1B | +0.0619 | [+0.0419, +0.0831] | <0.0001 | +0.0087 | 0.481 |
| 1.4B vs 1.4B | +0.0694 | [+0.0462, +0.0931] | <0.0001 | -0.0225 | 0.111 |
| 2.8B vs 2.8B | +0.0531 | [+0.0281, +0.0775] | <0.0001 | -0.0019 | 0.920 |

The endpoint trend summary labels primacy as `grows`, with change `+0.0538` from the smallest to largest pair.
The primacy gap is near zero at the two smallest pairs and appears from the 790M versus 1B pair onward.
The smallest Pythia model has an oracle ceiling of 0.009, so its flat curve is a capability-floor limitation rather than evidence of architecture-invariant shape.
Among pairs where both families can answer, the recency difference is near zero.

This phase supports a scale-emergent architecture gap, not a causal statement about depth.
Architecture, depth, positional encoding, and capability remain partly confounded across the size pairs.

### Phase 5: training-data control

Phase 5 held the Mamba-2.8B architecture, tokenizer, depth, host, and execution path fixed while swapping the Pile for SlimPajama.

The Pile primacy edge was `-0.0031`, and the SlimPajama primacy edge was `+0.0038`.
The corpus interaction, Pile minus SlimPajama, was `-0.0069`, 95% interval `[-0.0300, +0.0163]`, Holm `p=0.568`.

The Pile recency edge was `+0.0744`, and the SlimPajama recency edge was `+0.0906`.
The corpus recency interaction was `-0.0163`, 95% interval `[-0.0438, +0.0119]`, Holm `p=0.502`.

SlimPajama raised the overall level of the curve.
The floor changed from 0.1138 to 0.2700 while the ceiling changed from 0.6150 to 0.6250.
The executed result is therefore a null shape change with a clear level change.
It does not isolate which property of the two corpora would explain the level shift.

### Phase 6: RULER `niah_single_1` task check

The committed RULER result uses 50 needle instances per model at each of 1K and 2K target lengths.
At 1K, Pythia-2.8B had primacy `+0.04`, Holm `p=0.084`, and recency `+0.04`, Holm `p=0.058`.
At 2K, Pythia-2.8B had primacy `+0.12`, 95% interval `[+0.05, +0.20]`, Holm `p<0.0001`, and recency `+0.10`, Holm `p=0.0022`.

Both Mamba models scored 1.0 at every needle depth at both lengths.
Their zero edges are task saturation, not evidence that the models are position-invariant.
The useful task-transfer result is that the transformer primacy arm reappears on the synthetic task at 2K.

Absolute RULER accuracy is not directly comparable to QA accuracy because the task, host, and difficulty differ.

### Phase 7: mechanism and artifact checks

These are exploratory mechanism analyses on 200-question variant subsets or on previously collected sweeps, not confirmatory tests on the held-out allocation.
The prompt-variant and template results report per-condition edges without a paired between-condition test, so differences between their point estimates are descriptive.

#### Query-position variants

The Liu baseline on the 200-question subset gave Mamba primacy 0.000 and Pythia primacy `+0.070`.
The bookend prompt gave Mamba primacy `+0.1025`, Holm `p<0.0001`, and Pythia primacy `+0.1500`, Holm `p<0.0001`.
The question-first prompt gave Mamba primacy `+0.0200`, Holm `p=0.366`, and recency `+0.0100`, Holm `p=0.562`.
The gold-padded prompt gave Mamba primacy `+0.0275`, Holm `p=0.183`, and Pythia primacy `+0.0375`, Holm `p=0.123`.

The bookend result is consistent with a fixed-state-compression hypothesis, but the variant comparisons are not a replacement for a preregistered confirmatory contrast.

#### Attention-sink scan

Token-0 attention share was measured across Pythia scale.
The final-layer sink share was approximately 0.003 for Pythia-160M, 0.034 for Pythia-410M, 0.089 for Pythia-1B, 0.139 for Pythia-1.4B, and 0.455 for Pythia-2.8B.
Pythia-410M is a useful caution because it has substantial mid-network sink mass but little final-layer sink and no significant primacy edge.
The data support late-layer sink mass as a correlate of primacy within the measured Pythia scale series.
They do not establish that the sink is causally necessary.

#### Storage versus utilization probe

The frozen mid-depth position probe reached accuracy 0.603 for Pythia-2.8B at layer 16 and 0.648 for Mamba-2.8B at layer 32.
The shuffled-label controls were 0.484 and 0.508 respectively.
Mamba therefore encoded the position label at least as well as Pythia in this probe while retaining a null QA primacy edge.
This is evidence for a utilization difference rather than a storage failure, but it is not a mechanistic proof.

#### Template and length sensitivity

The Phase 2 sign pattern survived the three scoring variants and prompt-length tertiles in the committed analyses.
The instructional-template Pythia primacy interval crosses zero, so the family sign pattern is not template-invariant across all executed cells.
The per-template Pythia primacy estimates were about `+0.070` for the baseline and `+0.005` for the instructional template on the 200-question variant subset, while recency was positive in both cells.
The safe claim is template sensitivity, not robustness of the primacy sign or magnitude.

#### Sink-block intervention

The Nemotron-H sink-block output is identical to its baseline output in the committed summary.
The custom attention path did not honor the generic token-0 mask hook.
This is a null implementation intervention, not evidence that attention sinks are unnecessary.
The direct forward-path intervention remains open.

### Phase 8: descriptive production-system comparison

Phase 8 ran Nemotron-H-8B, Llama-3.1-8B, and Qwen2.5-7B on one L40S using the 800-question ten-position harness.
This is not a matched control because architecture, corpus, token count, tokenizer, alignment, depth, positional encoding, and in some cases attention implementation differ.

| System | Primacy edge | 95% interval | Holm p | Recency edge | 95% interval | Holm p | Floor / ceiling |
|---|---:|---|---:|---:|---|---:|---:|
| Nemotron-H-8B | +0.0669 | [+0.0438, +0.0906] | <0.0001 | +0.0288 | [+0.0081, +0.0494] | 0.006 | 0.3113 / 0.5875 |
| Llama-3.1-8B | +0.0763 | [+0.0569, +0.0956] | <0.0001 | +0.0175 | [0.0000, +0.0344] | 0.051 | 0.3150 / 0.7825 |
| Qwen2.5-7B | +0.0406 | [+0.0206, +0.0606] | 0.0008 | +0.0075 | [-0.0119, +0.0269] | 0.474 | 0.2800 / 0.8150 |

All three production systems have a positive primacy edge.
Only the Llama versus Qwen primacy interaction clears Holm correction, at `+0.0356`, 95% interval `[+0.0088, +0.0625]`, Holm `p=0.0164`.
The production result is a robustness description, not an architecture attribution.

## Controls and unexecuted requirements

The positive key-value control is executed in the committed Phase 1, Phase 2, Phase 3, Phase 5, and Phase 8 artifact directories.
Its result is model-dependent and is non-gating for the QA sweeps after the Phase 1 instrument check.

The committed `artifacts/phase2/controls-09eab5e/` run executes both certification controls on 200 exploratory Pythia questions.
The sham-gold accuracy was `0.1235` against a `0.0950` floor, a `0.0285` difference within its fixed gate; its primacy was `+0.0050` with 95% interval `[-0.0325, +0.0425]`, and its recency was `-0.0175` with 95% interval `[-0.0500, +0.0125]`.
The three distractor-order accuracies were `0.2850`, `0.3000`, and `0.2767`, for a maximum spread of `0.0233` within its fixed gate.
Both controls passed on this exploratory Pythia sample only and were not executed on the 8B Megatron checkpoints.

The manual 50-generation audit sample exists, but its human labels do not.
The held-out 1,855-question confirmatory split is frozen and unused.

The Phase 3 matched contrast is not significant for the paired primacy effect after Holm correction.
The Nemotron-H sink-block result is an unsupported-mask null, not a causal ablation.

## Diagram readout

The Phase 2 curve figure shows a high Pythia start point, a middle trough, and a shared late-position rise, while both Mamba curves lack the same start arm.
The Phase 2 edge figure shows Pythia's positive primacy bar and near-zero or negative Mamba primacy bars alongside positive recency bars for all three models.
The Phase 3 curve figure shows the hybrid curve above the pure curve, but the hybrid-minus-pure effect figure's primacy interval crosses zero.
The Phase 4 scale-gap figure shows a primacy gap emerging at the 790M versus 1B pair and remaining positive thereafter, while recency differences sit near zero at capable scales.
The sink figure shows the largest late-layer token-0 share in Pythia-2.8B and almost no final-layer share in Pythia-160M.
The Phase 8 curve figure shows primacy in all three production systems and a less consistent recency arm.
The Phase 1 key-value figure confirms that the positive-control task has a strong but non-monotone position curve.
The 2K RULER figure shows Pythia's middle dip while both Mamba curves are saturated at one.

## Paper-safe claims

The strongest artifact-backed claim is that the exploratory near-2.8B Pile family comparison has a Pythia primacy arm that is absent or negative in both Mamba variants, while all three share a positive recency arm.
The scale sweep shows that this primacy interaction is near zero at weak small-model pairs and appears from the 790M versus 1B pair onward.
The corpus control shows a level shift without a detectable edge shift on the fixed Mamba architecture.
The synthetic task reproduces the transformer primacy arm at 2K, while the Mamba controls saturate.
The exploratory mechanism evidence identifies associations among late-layer attention-sink mass, linearly decodable position, and prompt-dependent edges, but it does not establish causal attention or fixed-state compression mechanisms.
The matched 8B pure-versus-hybrid result is suggestive but not confirmatory for an attention-causes-primacy claim, and the released checkpoints also differ in MLP composition.

## Open work before a final causal paper claim

Extend the negative and distractor-order controls beyond the exploratory Pythia sample if they are required for the 8B checkpoints.
Complete and archive the blinded human audit labels.
Freeze the statistical and exclusion rules before opening the held-out 1,855-question split.
Re-run the matched Phase 3 contrast on the confirmatory allocation if the causal attention claim is retained.
Implement a direct Nemotron-H attention forward-path sink ablation and verify that unrelated capability is preserved.
Keep all future claims tied to raw records, summary JSON, environment metadata, and the exact producing commit.

## Verification performed for this audit

This section records the 2026-08-11 audit and is retained for historical context.
Current verification results belong in the pull-request audit rather than this scientific evidence ledger.
