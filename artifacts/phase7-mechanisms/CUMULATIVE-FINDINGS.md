# Mixing Matters: cumulative findings across all phases

This report reads every phase's committed artifacts together and states what the program has established, with the statistics attached to each claim.
It supersedes the earlier Phase 7 cumulative note by folding in the Phase 7 mechanism experiments (4a-4e) and the Phase 8 production-system comparison.

The project question: at matched size and context, does changing the sequence-mixing architecture (dense causal attention vs a Mamba state-space model) change the accuracy-versus-evidence-position curve, and if so, why.

The QA phases share the Lost-in-the-Middle 10-document harness: gold document placed at each position 0-9, a closed-book floor and an oracle ceiling per question, greedy decoding at temperature 0, 32 new tokens, seed 240521, BF16, and the vendored `best_subspan_em` scorer.
Phase 6 adapts the position sweep to a synthetic RULER task and therefore changes the prompt and scorer as well as the task.
Two edges are defined once and reused: **primacy** = mean accuracy at positions 0,1 minus positions 4,5; **recency** = mean accuracy at positions 8,9 minus positions 4,5.
Both are tested with a paired bootstrap over question bundles (10,000 resamples, the question as the unit of analysis) and Holm correction across the two edges.

---

## The one-line result

In the exploratory near-2.8B Pile comparison, Pythia has a **primacy arm** that two Mamba variants lack, while all three have a recency arm.
The models are matched on corpus and approximate scale but differ in depth, positional encoding, state size, checkpoint family, and capability, so this is a family-by-position result rather than an isolated architecture effect.
Scale, corpus, synthetic-task, and production-system phases describe where related patterns appear, but none removes all of those confounds.
The sink, probe, and prompt studies motivate hypotheses about evidence use; they do not establish that attention sinks or fixed-state compression cause the measured edges.

---

## Phase-by-phase evidence

### Phase 1 - the pipeline detects a real position effect

Pythia-2.8B, 200 questions. gold-first minus gold-middle = +0.115 (bootstrap CI [0.055, 0.175]); closed-book floor 0.095, oracle ceiling 0.65.
The key-value positive control passes for Pythia (edge +0.38) and Mamba-2 (+0.65) and does not pass for Mamba-1 (+0.00), consistent with the smaller model's 16-state capacity.
Scoring is stable across `best_subspan_em`, normalized EM, and first-line extraction.
The measurement instrument works.

### Phase 2 - the exploratory family gap near 2.8B (Pile, one A40)

| Model | Primacy | Recency |
|---|---|---|
| pythia-2.8b | **+0.052** (Holm < 1e-4) | +0.075 (< 1e-4) |
| mamba-2.8b | -0.001 (0.914) | +0.077 (< 1e-4) |
| mamba2-2.7b | -0.018 (0.023) | +0.105 (< 1e-4) |

Pythia-minus-Mamba primacy interaction: -0.053 (Mamba-1) and -0.070 (Mamba-2), both Holm < 1e-4. The two Mamba variants do not differ from each other on primacy.
Only Pythia has a positive primacy arm in this comparison, while recency is positive for all three models.
Corpus and approximate scale are matched, but depth, positional encoding, state size, checkpoint family, and capability remain confounded.

### Phase 4 - the primacy gap is scale-emergent (five matched pairs, one A40)

Pythia-minus-Mamba primacy difference by size: -0.001 (130m-160m), +0.003 (370m-410m), **+0.062** (790m-1b), **+0.069** (1.4b), **+0.053** (2.8b).
Descriptively, the gap is near zero for the two smallest pairs and positive for the three larger pairs, once both compared models can answer.
This endpoint pattern does not establish a causal scale threshold.
Recency difference is stable/near-zero once both families can answer.

### Phase 5 - no detectable edge-shape change in one corpus contrast

Mamba-2.8B on the Pile vs the same architecture on SlimPajama.
Corpus effect on primacy edge: -0.007 (Holm 0.568); on recency edge: -0.016 (Holm 0.502). Both intervals span zero.
SlimPajama lifts the whole curve (floor 0.11 -> 0.27) while barely moving the ceiling (0.615 -> 0.625).
The substantial floor shift prevents attributing that level change to memorization alone, and the null edge contrasts do not rule out smaller corpus effects.

### Phase 6 - the Pythia arm appears on one synthetic task (RULER niah_single_1, T4)

At 2K tokens Pythia's primacy edge reappears on a synthetic numeric-needle task, +0.12 (Holm < 1e-4), larger than the QA value; recency +0.10 (Holm 0.002). At 1K it is +0.04 (not significant after Holm).
The Mamba models saturate at accuracy 1.0 at every needle depth, so their zeros are task saturation, not position invariance; Phase 6 tests whether the transformer arm reproduces on a different task, and at 2K it does.
Because the task, prompt, scorer, host, and sample size change together and raw generations are unavailable, this phase does not rule out prompt or scoring explanations for other phases.

### Phase 8 - primacy is prevalent across three production 7B to 8B systems (L40S)

Descriptive comparison, not a matched control: the systems differ on architecture, corpus, token count, tokenizer, alignment, depth, and positional encoding at once.

| System | Primacy | Recency | Ceiling |
|---|---|---|---|
| llama-3.1-8b (dense + RoPE) | +0.076 (< 1e-4) | +0.018 (0.051) | 0.782 |
| nemotron-h-8b (hybrid Mamba-2 + attention) | +0.067 (< 1e-4) | +0.029 (0.006) | 0.588 |
| qwen2.5-7b (dense + RoPE) | +0.041 (0.0008) | +0.008 (0.474) | 0.815 |

All three have a Holm-significant primacy arm, establishing prevalence on this harness across the three full systems.
The recency arm differs descriptively by system.
Only the Llama-minus-Qwen primacy contrast clears Holm (+0.036, p=0.016), but every pair changes many factors and the phase cannot isolate architecture, corpus, alignment, or any other one cause.

---

## Phase 7 - mechanism-oriented evidence

### 4b Depth is not sufficient to explain the observed series (compute-free, from Phase 2 + Phase 4 sweeps)

Within the Pythia family, the one contrast where depth and width move in opposite directions is Pythia-410M (24 layers, 1024 wide, primacy +0.013) vs Pythia-1B (16 layers, 2048 wide, primacy +0.052).
This contrast is inconsistent with a simple monotonic depth-only account, but width, capability, and checkpoint identity also change.

### 4e Scoring, length, and template sensitivity (compute-free + GPU)

- Scoring: signs and Holm-significance of every edge are preserved under normalized EM and first-line-only extraction on every model at 1B and above.
- Length: every prompt-length tertile carries the Phase 2 signature (Pythia positive, Mamba at or below zero).
- Template: Mamba primacy is null under liu, concise, and instructional templates, while Pythia is positive under liu and concise but has a +0.005 instructional estimate whose interval crosses zero. The cells have no paired between-template contrast, so neither sign nor magnitude is established as template-invariant.

### 4c Attention-sink mass covaries with the primacy arm (GPU)

Token-0 attention share per layer, Pythia family:

| Model | Overall-mean sink | Final-layer sink | Primacy |
|---|---|---|---|
| pythia-160m | 0.029 | 0.003 | null |
| pythia-410m | 0.109 | 0.034 | +0.013 (ns) |
| pythia-1b | 0.082 | 0.089 | +0.052 |
| pythia-1.4b | 0.103 | 0.139 | +0.053 |
| pythia-2.8b | 0.110 | **0.455** | +0.052 |

The model with a null primacy arm has almost no final-layer sink (0.003), while larger models with a primacy arm have more final-layer sink mass and Pythia-2.8B reaches 0.455.
Pythia-410M has high mid-network sink mass that drops by the output layer and a non-significant primacy estimate.
Scale is shared by both measures, and no valid sink ablation was completed, so this is an association rather than evidence of necessity or sufficiency.
(The companion measurement on the Nemotron-H hybrid could not be produced - its custom attention does not expose weights under `output_attentions`.)

### 4a Prompt variants produce different per-condition edge estimates (GPU)

Prompt-layout variants with fixed questions and document placement:

| Variant | Mamba primacy | Pythia primacy |
|---|---|---|
| baseline (docs then question) | +0.000 (null) | +0.070 |
| bookend (question before and after) | **+0.103** (< 1e-4) | **+0.150** (< 1e-4) |
| question_first | +0.020 (ns) | +0.088 |
| gold_padded (128 tokens after docs) | +0.028 (ns) | +0.038 |

Mamba's bookend primacy estimate is +0.103 versus a null baseline, and its question-first recency estimate is +0.010 versus +0.070 at baseline.
For Pythia, bookend primacy is +0.150 versus +0.070 at baseline, while gold-padded primacy is +0.038.
These patterns are consistent with query-dependent compression and sink hypotheses, but the variants jointly change query placement, repetition, and token layout and have no paired between-condition test.

### 4d Position is linearly decodable in both families (GPU)

Balanced linear probe for gold-at-edge vs gold-at-middle from frozen mid-depth hidden states (layer fixed by rule before fitting), shuffled-label control at chance:

| Model | Probe accuracy | Shuffled control | QA primacy |
|---|---|---|---|
| pythia-2.8b (layer 16) | 0.603 | 0.484 | +0.052 |
| mamba-2.8b (layer 32) | **0.648** | 0.508 | -0.001 (null) |

The position label is modestly more decodable from the measured Mamba states than from the measured Pythia states, even though Mamba has a null primacy arm.
This motivates a utilization hypothesis, but the probe classifies position rather than answer content and does not prove what either model stores or uses for generation.

---

## Mechanism hypotheses and boundaries

The executed evidence narrows some simple explanations but does not identify a causal mechanism:

| Candidate | Verdict | Evidence |
|---|---|---|
| Training-data position patterns | Not detected in one corpus edge contrast | Phase 5 null corpus effect with a large floor shift |
| Prompt / tokenization / scoring artifact | Sensitivity bounded, not ruled out | Phase 6 changes several axes; Phase 7 scoring, length, and template cells |
| Model depth per se | Simple depth-only account is insufficient | Phase 7 4b also changes width and capability |
| Fixed-state compression (Mamba) | Hypothesis consistent with prompt cells | Phase 7 4a changes placement, repetition, and layout without a paired condition contrast |
| Attention sinks (Transformer) | Correlate and hypothesis | Phase 7 4c shares scale as a confound; sink-block attempt was a no-op |
| Storage vs utilization | Hypothesis | Phase 7 4d decodes position, not answer-content storage or use |

One hypothesis consistent with the observations is that prompt layout interacts with how different mixers retain and use evidence, with late-layer sink mass marking one Transformer regime.
The current probes and prompt variants cannot distinguish that account from capability, checkpoint, or other internal differences.
Phase 8 contributes prevalence only, and its unmatched systems cannot assign either edge to attention or state-space blocks.

---

## What remains open

- **Phase 3** completed on the 800-question exploratory split. The hybrid-minus-pure primacy effect is +0.0188 with 95% interval [-0.0056, +0.0444] and Holm p=0.1442. The original producing tree was dirty; a clean rerun at commit `33d6bb5` reproduced the summary byte-for-byte. Because the released checkpoints differ in both attention and MLP composition, the contrast is not an attention-only causal test.
- **Hybrid sink measurement**: Nemotron-H's custom attention must be patched to expose weights before its sink mass can be read.
- **Sink-block ablation**: the token-0 mask hook was a no-op on Nemotron-H's custom attention path, so the existing output is invalid as an ablation; a direct forward patch is needed to test whether removing the sink specifically removes the primacy arm.
- **Confirmatory split**: every number above is on the 800-question exploratory set; the 1,855-question held-out set is still frozen and unspent.

## Where the numbers live

Each claim's source of truth is that phase's `*-summary.json`:
`artifacts/phase1/summary.json`, `artifacts/phase2/report/phase2-summary.json`, `artifacts/phase3/report/phase3-summary.json`, the Phase 4 pair summaries, `artifacts/phase5/report/phase5-summary.json`, `artifacts/phase6/report/phase6-summary.json`, `artifacts/phase8/report/phase8-summary.json`, and the Phase 7 mechanism reports under `artifacts/phase7-mechanisms/`.
