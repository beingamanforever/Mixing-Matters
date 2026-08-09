# Mixing Matters: cumulative findings across all phases

This report reads every phase's committed artifacts together and states what the program has established, with the statistics attached to each claim.
It supersedes the earlier Phase 7 cumulative note by folding in the Phase 7 mechanism experiments (4a-4e) and the Phase 8 production-system comparison.

The project question: at matched size and context, does changing the sequence-mixing architecture (dense causal attention vs a Mamba state-space model) change the accuracy-versus-evidence-position curve, and if so, why.

Every phase shares one harness: the Lost-in-the-Middle 10-document QA set, gold document placed at each position 0-9, a closed-book floor and an oracle ceiling per question, greedy decoding at temperature 0, 32 new tokens, seed 240521, BF16, the vendored `best_subspan_em` scorer.
Two edges are defined once and reused: **primacy** = mean accuracy at positions 0,1 minus positions 4,5; **recency** = mean accuracy at positions 8,9 minus positions 4,5.
Both are tested with a paired bootstrap over question bundles (10,000 resamples, the question as the unit of analysis) and Holm correction across the two edges.

---

## The one-line result

At matched scale, corpus, tokenizer, context, and seed, a dense-attention transformer has a **primacy arm** (higher accuracy when the gold document is at the start) that a Mamba state-space model does not.
The recency arm is shared by both.
The primacy arm emerges with scale, survives a corpus swap, survives a task swap, and reappears across three production 8B families.
The mechanism evidence points at an **attention sink** the transformer learns at the head of the prompt: Mamba stores where the gold document is just as well as the transformer but does not convert that into an edge advantage, and interventions that move query-relevant tokens toward or away from the prompt head move the transformer's primacy arm in the predicted direction.

---

## Phase-by-phase evidence

### Phase 1 - the pipeline detects a real position effect

Pythia-2.8B, 200 questions. gold-first minus gold-middle = +0.115 (bootstrap CI [0.055, 0.175]); closed-book floor 0.095, oracle ceiling 0.65.
The key-value positive control passes for Pythia (edge +0.38) and Mamba-2 (+0.65) and fails for Mamba-1 (+0.00, its 16-state cannot hold 30 pairs).
Scoring is stable across `best_subspan_em`, normalized EM, and first-line extraction.
The measurement instrument works.

### Phase 2 - the primacy arm is architecture-specific (2.8B, Pile, one A40)

| Model | Primacy | Recency |
|---|---|---|
| pythia-2.8b | **+0.052** (Holm < 1e-4) | +0.075 (< 1e-4) |
| mamba-2.8b | -0.001 (0.914) | +0.077 (< 1e-4) |
| mamba2-2.7b | -0.018 (0.023) | +0.105 (< 1e-4) |

Pythia-minus-Mamba primacy interaction: -0.053 (Mamba-1) and -0.070 (Mamba-2), both Holm < 1e-4. The two Mamba variants do not differ from each other on primacy.
At matched everything-but-architecture, only the transformer has a primacy arm. Recency is universal.

### Phase 4 - the primacy gap is scale-emergent (five matched pairs, one A40)

Pythia-minus-Mamba primacy difference by size: -0.001 (130m-160m), +0.003 (370m-410m), **+0.062** (790m-1b), **+0.069** (1.4b), **+0.053** (2.8b).
Verdict: **grows** then stabilizes; the gap appears once the models cross roughly 1B parameters and both can actually answer.
Recency difference is stable/near-zero once both families can answer.

### Phase 5 - training data does not move the shape (architecture fixed)

Mamba-2.8B on the Pile vs the same architecture on SlimPajama.
Corpus effect on primacy edge: -0.007 (Holm 0.568); on recency edge: -0.016 (Holm 0.502). Both intervals span zero.
SlimPajama lifts the whole curve (floor 0.11 -> 0.27) while barely moving the ceiling (0.615 -> 0.625): the corpus adds memorized answers, not a change in how position is used.
Contrast with Phase 2: swapping the architecture moved primacy; swapping the corpus did not.

### Phase 6 - the shape survives a change of task (RULER niah_single_1, T4)

At 2K tokens Pythia's primacy edge reappears on a synthetic numeric-needle task, +0.12 (Holm < 1e-4), larger than the QA value; recency +0.10 (Holm 0.002). At 1K it is +0.04 (not significant after Holm).
The Mamba models saturate at accuracy 1.0 at every needle depth, so their zeros are task saturation, not position invariance; Phase 6 tests whether the transformer arm reproduces on a different task, and at 2K it does.
Rules out prompt-template and scoring format as the driver of the shape.

### Phase 8 - the pattern holds across three production 8B systems (L40S)

Descriptive comparison, not a matched control: the systems differ on architecture, corpus, token count, tokenizer, alignment, depth, and positional encoding at once.

| System | Primacy | Recency | Ceiling |
|---|---|---|---|
| llama-3.1-8b (dense + RoPE) | +0.076 (< 1e-4) | +0.018 (0.051) | 0.782 |
| nemotron-h-8b (hybrid Mamba-2 + attention) | +0.067 (< 1e-4) | +0.029 (0.006) | 0.588 |
| qwen2.5-7b (dense + RoPE) | +0.041 (0.0008) | +0.008 (0.474) | 0.815 |

All three have a Holm-significant primacy arm: the phenomenon survives replacing the entire training pipeline.
The recency arm splits by system and is cleanest on the one system with SSM blocks (Nemotron-H), echoing Phase 2 where Mamba-2 carried the largest recency edge.
Only the Llama-minus-Qwen primacy contrast clears Holm (+0.036, p=0.016) - two dense-RoPE models still differ, so corpus/alignment still move primacy magnitude at fixed architecture family.

---

## Phase 7 - why it happens

### 4b Depth is not the cause (compute-free, from Phase 2 + Phase 4 sweeps)

Within the Pythia family, the one contrast where depth and width move in opposite directions is Pythia-410M (24 layers, 1024 wide, primacy +0.013) vs Pythia-1B (16 layers, 2048 wide, primacy +0.052).
**More layers with less width gives less primacy.** Depth alone does not manufacture the arm; capability crossing ~1B does.

### 4e Not a scoring or length or template artifact (compute-free + GPU)

- Scoring: signs and Holm-significance of every edge are preserved under normalized EM and first-line-only extraction on every model at 1B and above.
- Length: every prompt-length tertile carries the Phase 2 signature (Pythia positive, Mamba at or below zero).
- Template: the architecture contrast is template-invariant (Mamba primacy null under liu/concise/instructional; Pythia positive under liu/concise). But the transformer's primacy *magnitude* is template-sensitive - the `instructional` template, which prepends a longer instruction and a `Documents:` header, collapses Pythia primacy from +0.070 to +0.005 while leaving recency intact. The safe claims are signs and the architecture contrast, not absolute edge values.

### 4c Attention-sink mass tracks the primacy arm (GPU)

Token-0 attention share per layer, Pythia family:

| Model | Overall-mean sink | Final-layer sink | Primacy |
|---|---|---|---|
| pythia-160m | 0.029 | 0.003 | null |
| pythia-410m | 0.109 | 0.034 | +0.013 (ns) |
| pythia-1b | 0.082 | 0.089 | +0.052 |
| pythia-1.4b | 0.103 | 0.139 | +0.053 |
| pythia-2.8b | 0.110 | **0.455** | +0.052 |

The model with a null primacy arm has almost no sink (final-layer 0.003); models with a primacy arm carry a substantial sink and the largest model develops an extreme final-layer sink of 0.455.
Pythia-410m is the informative wrinkle: high mid-network sink but it collapses by the output layer and its primacy is not significant, so it is **late-layer** sink mass, not raw sink mass, that couples to the arm.
(The companion measurement on the Nemotron-H hybrid could not be produced - its custom attention does not expose weights under `output_attentions`.)

### 4a Query position moves the arms as fixed-state compression predicts (GPU)

Prompt-order ablation, single-variable change on where the question sits:

| Variant | Mamba primacy | Pythia primacy |
|---|---|---|
| baseline (docs then question) | +0.000 (null) | +0.070 |
| bookend (question before and after) | **+0.103** (< 1e-4) | **+0.150** (< 1e-4) |
| question_first | +0.020 (ns) | +0.088 |
| gold_padded (128 tokens after docs) | +0.028 (ns) | +0.038 |

Placing the question before the documents lets the Mamba recurrent state filter documents as it compresses them: Mamba's primacy jumps from a null baseline to +0.103, the largest Mamba primacy edge anywhere in this study, exactly the fixed-state-compression prediction.
Under `question_first` Mamba's recency drops from +0.070 to +0.010 (the state stops merely carrying the tail).
For Pythia, `bookend` more than doubles primacy (extra query tokens in the sink-bearing head) and `gold_padded` halves it (filler pushes gold away from the head) - both consistent with the sink reading.

### 4d Utilisation, not storage (GPU)

Balanced linear probe for gold-at-edge vs gold-at-middle from frozen mid-depth hidden states (layer fixed by rule before fitting), shuffled-label control at chance:

| Model | Probe accuracy | Shuffled control | QA primacy |
|---|---|---|---|
| pythia-2.8b (layer 16) | 0.603 | 0.484 | +0.052 |
| mamba-2.8b (layer 32) | **0.648** | 0.508 | -0.001 (null) |

Mamba encodes gold position **more** strongly than Pythia yet has a null primacy arm.
Storage of position is not the bottleneck - both architectures locate the gold document; only the transformer turns stored position into an edge-accuracy advantage.
This is the utilisation-not-storage signature.

---

## The mechanism, assembled

Every candidate cause from the project brief now has a verdict:

| Candidate | Verdict | Evidence |
|---|---|---|
| Training-data position patterns | Ruled out as shape driver | Phase 5 null corpus effect |
| Prompt / tokenization / scoring artifact | Ruled out | Phase 6 task swap; Phase 7 4e scoring, length, template |
| Model depth per se | Ruled out as sufficient cause | Phase 7 4b (410m < 1b on primacy) |
| Fixed-state compression (Mamba) | Supported | Phase 2/8 recency on SSM models; Phase 7 4a bookend lifts Mamba primacy 0 -> +0.103, question_first kills Mamba recency |
| Attention sinks (Transformer) | Leading mechanism for primacy | Phase 7 4c late-layer sink tracks primacy; 4a gold_padded halves it, bookend doubles it; 4e instructional header collapses it |
| Storage vs utilisation | Utilisation | Phase 7 4d: Mamba stores position better yet shows no arm |

The picture that fits every result without contradiction:

1. Dense causal attention learns an attention sink at the prompt head; late-layer sink mass steers the answer toward early context, producing the primacy arm. It scales with capability (needs ~1B parameters) and with the amount of sink mass surviving to the output layer.
2. Mamba has no attention and no sink, so no primacy arm - even though its recurrent state clearly encodes where the gold document is. Its state-compression instead produces a recency arm, and exposing the question early (bookend/question_first) lets that same state manufacture a primacy arm on demand.
3. A hybrid (Nemotron-H) carries both: attention layers restore primacy, SSM layers keep the cleanest recency of the three 8B systems.
4. Corpus and task change the level, not the shape.

---

## What remains open

- **Phase 3** (matched pure Mamba-2 8B vs hybrid Mamba-2 8B, same data/depth/tokenizer) is the one clean single-variable test that attention layers *cause* primacy. Code is staged; the run is the remaining confirmatory piece.
- **Hybrid sink measurement**: Nemotron-H's custom attention must be patched to expose weights before its sink mass can be read.
- **Sink-block ablation**: the token-0 mask hook is a null op on Nemotron-H's custom attention path; a direct forward patch is needed to test whether removing the sink specifically removes the primacy arm.
- **Confirmatory split**: every number above is on the 800-question exploratory set; the 1,855-question held-out set is still frozen and unspent.

## Where the numbers live

Each claim's source of truth is that phase's `*-summary.json`:
`artifacts/phase1/summary.json`, `artifacts/phase2/report/phase2-summary.json`, the Phase 4 pair summaries on the `phase-4` branch, `artifacts/phase5/report/phase5-summary.json`, `artifacts/phase6/report/phase6-summary.json`, `artifacts/phase8/report/phase8-summary.json`, and the Phase 7 mechanism reports under `artifacts/phase7-mechanisms/`.
