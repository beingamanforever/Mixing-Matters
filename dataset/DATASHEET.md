# Mixing Matters position-bias dataset

Every model generation and attention measurement produced by the Mixing Matters study, flattened into one schema.

The study moves a single answer-bearing passage through ten positions in a fixed multi-document context and records what each model answers.
This release is the record-level output of that intervention across 17 pinned checkpoints, 4 prompt variants, 3 instruction templates, and 2 tasks.

Rebuild it from the committed artifacts with one command:

```bash
uv run mixing-matters build-dataset --output dataset
```

## Files

| File | Rows | Description |
|---|---:|---|
| `generations.jsonl.gz` | 229,700 | One row per model generation, across every phase, model, position, and prompt variant. |
| `attention_sink.jsonl.gz` | 280,000 | Token-0 attention mass per layer, per question, per gold position, for five Pythia scales. |
| `runs.csv` | 60 | One row per executed sweep file, with the model pin and record counts it contributed. |
| `position_accuracy.csv` | 522 | Accuracy at each gold position for each run, recomputed from the released rows. |

Total size is under 10 MB compressed.

## Composition

The `generations.jsonl.gz` stream covers two tasks.

`multidoc_qa` (219,200 rows) is the Liu et al. ten-document question answering release.
Each question contributes twelve rows per model: ten gold positions, one closed-book condition, and one gold-only oracle condition.
`kv_retrieval` (10,500 rows) is the synthetic key-value positive control used to show that each execution path can detect a position effect at all.

Prompt variants are `liu_baseline`, `bookend`, `question_first`, and `gold_padded`.
Prompt templates are `liu_baseline`, `concise`, and `instructional`.
The two axes are independent columns because Phase 7 varied them separately.

Models span three mixer classes, recorded in the `mixer` column: `attention` (Pythia 160M to 2.8B, Llama 3.1 8B, Qwen2.5 7B), `state-space` (Mamba 130M to 2.8B, Mamba-2 2.7B and 8B), and `hybrid` (Mamba-2 hybrid 8B, Nemotron-H 8B).

## Schema: `generations.jsonl.gz`

| Field | Type | Notes |
|---|---|---|
| `run_key` | string | Joins to `runs.csv` and `position_accuracy.csv`. |
| `phase` | string | `phase1` through `phase8`. |
| `experiment` | string | The contrast the run belongs to, for example `matched-2.8b-architecture`. |
| `task` | string | `multidoc_qa` or `kv_retrieval`. |
| `model_key` | string | Registry key, for example `mamba2-hybrid-8b`. |
| `mixer` | string | `attention`, `state-space`, or `hybrid`. |
| `family` | string | Finer-grained family that also selects the execution path. |
| `model_repo`, `model_revision` | string | Exact pinned checkpoint. |
| `params_millions` | int or null | Null for checkpoints outside the scale sweep. |
| `training_corpus` | string or null | Set only for the Phase 5 corpus pair. |
| `prompt_variant` | string | Where the query sits relative to the documents. |
| `prompt_template` | string | Instruction wording. |
| `question_id`, `source_index` | string, int | Stable identifiers into the upstream release. |
| `condition` | string | `gold`, `closed_book`, `oracle`, or `kv_retrieval`. |
| `gold_position` | int or null | 0-indexed. Null for closed-book and oracle rows. |
| `answers` | list of string | Accepted gold answers. |
| `model_response` | string | Raw greedy decode, capped at 32 new tokens. |
| `score` | float | Primary `best_subspan_em`. |
| `score_normalized_em`, `score_first_line` | float or null | Sensitivity scores. Null on key-value rows. |
| `prompt_token_count`, `generated_token_count` | int | Tokenizer counts at run time. |
| `gpu`, `execution_path`, `dtype` | string | Execution provenance. |
| `run_id` | string or null | Groups rows produced by one process. |

## Schema: `attention_sink.jsonl.gz`

One row per (model, question, gold position, layer): `run_key`, `phase`, `model_key`, `mixer`, `family`, `question_id`, `source_index`, `prompt_variant`, `condition`, `gold_position`, `layer`, `sink_mass`, `prompt_token_count`, `run_id`.
`sink_mass` is the share of attention mass that layer sends to token 0.

## What is deliberately not here

Prompt text is excluded.
A ten-document prompt averages 5.5 KB, so carrying it would grow the release roughly twentyfold.
The harness rebuilds any prompt deterministically from `question_id`, `gold_position`, `prompt_variant`, and `prompt_template`; the builders are vendored in `src/lost_in_the_middle/` at upstream commit `29b8a6d042ce29abccee3db1a73171a107d7e6af`.

Phase 6 RULER `niah_single_1` generations are not included.
Those raw records were never committed to the public tree, so only the Phase 6 aggregate summary is public.

Phase 7 probe rows are not included.
They are 2,560-dimensional hidden-state vectors whose only released use is the fitted probe accuracy in `artifacts/phase7-mechanisms/4d-probe/`.

## Collection process

The dataset was fixed at 2,655 questions with SHA-256 `192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9` and split with seed `240521` into 800 exploratory and 1,855 confirmatory questions.
Every row in this release comes from the 800-question exploratory allocation.
The 1,855-question confirmatory allocation has never been opened.

Decoding was greedy and identical across models: `do_sample=false`, temperature 0, `top_p=1`, one beam, 32 maximum new tokens, BF16 where supported, seed `240521`.
Within a question, prompt token counts vary by at most one token across the ten positions.

## Known limitations

Phase 1 ran on a different host from the later phases, so its `execution_path` is null and its GPU differs.
The Phase 3 environment manifest records a dirty source tree, with the Megatron runner uncommitted at run time.
The `phase7/nemotron-h-8b-sink-blocked` run returned output identical to its baseline because the generic token-0 mask hook was not honoured by that model's custom attention path; it is a null implementation, not a successful ablation.
Key-value control rows carry no `run_id` and no sensitivity scores because that harness predates those fields.

Accuracy figures in this release describe fixed checkpoints under fixed prompts and decoding settings.
They do not describe variation across training seeds.

## Licence and citation

Released under the Apache License 2.0, the same licence as the repository.
The underlying questions come from the Lost in the Middle release of Liu et al. (2024); their terms apply to the question and passage content.

Cite by title until the archival record is public: *Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering*, New in ML workshop submission, 2026.
