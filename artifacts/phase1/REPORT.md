# Phase 1 report

## Status

Completed.
The pipeline runs end to end, the positive control passes its gate, and the recorded output supports the Phase 2 analysis without rerunning the model.
The manual generation audit is prepared but not yet performed.

## Experimental setup

Model: `EleutherAI/pythia-2.8b` at revision `2a259cdd96a4beb1cdf467512e3904197345f6a9`.

Dataset: the Lost-in-the-Middle multi-document QA release from Liu et al., 10-document setting, file `nq-open-10_total_documents_gold_at_0.jsonl.gz`, SHA-256 `192a05b2...37aab9`, upstream commit `29b8a6d042ce29abccee3db1a73171a107d7e6af`.
The dataset contains 2,655 question instances, each with one gold document and nine distractors.

Split: one split with seed 240521 produces 800 exploratory and 1,855 confirmatory instances.
Phase 1 used the first 200 exploratory indices.
A test asserts that the selected indices are disjoint from the confirmatory set.
The confirmatory set was not read.

Conditions, four per question, 800 generations total:

- `closed_book`: the question with no documents, which measures the floor from guessing and memorised knowledge.
- `oracle`: the gold document alone, which measures the ceiling when retrieval is perfect.
- `gold_first`: ten documents with the gold document at position 0.
- `gold_middle`: ten documents with the gold document at position 4.

Prompt construction and scoring use the vendored upstream code without semantic changes.
The primary score is `best_subspan_em`.
The prompt is the official zero-shot QA prompt in documents-then-question order.

Decoding: greedy, temperature 0, top_p 1, top_k unset, `num_beams` 1, 32 maximum new tokens, seed 240521, bfloat16.

The call to `generate` passes `do_sample=False`, `num_beams=1`, and `max_new_tokens=32`.
The recorded `temperature`, `top_p`, and `top_k` fields state the greedy configuration that those arguments select rather than arguments that are passed through, because sampling parameters have no effect once `do_sample` is false.
Verifying those three fields against the code therefore requires reading `Generator.__call__`, not only the records.

Runtime: Python 3.12.10, torch 2.7.1+cu126, transformers 4.57.1, CUDA runtime 12.6, driver 570.133.20, one NVIDIA A10G.
Every one of these values is recorded in each JSONL record under `software_versions`, including the compute dtype read back from the loaded model.

Positive control: the released key-value retrieval task, 50 examples, 30 keys, the target key swept across ten slots, 500 generations.

## Results

### Positive control

Accuracy by key-value slot, with 95 percent paired bootstrap intervals over complete example bundles:

| Slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.94 | 0.34 | 0.20 | 0.08 | 0.18 | 0.16 | 0.08 | 0.20 | 0.42 | 0.16 |

Mean of the edge slots 0 and 9 is 0.55.
Mean of the middle slots 4 and 5 is 0.17.
The difference is 0.38 against a gate threshold of 0.05, so the gate passes.
The pipeline detects a known position effect.

### Phase 1 conditions

Primary score with 95 percent paired bootstrap intervals over complete question bundles, 200 questions:

| Condition | Accuracy | 95 percent interval |
|---|---|---|
| `closed_book` (floor) | 0.095 | 0.055 to 0.140 |
| `gold_middle` | 0.225 | 0.170 to 0.285 |
| `gold_first` | 0.340 | 0.275 to 0.405 |
| `oracle` (ceiling) | 0.650 | 0.585 to 0.715 |

Paired contrast `gold_first` minus `gold_middle` is 0.115 with a 95 percent interval of 0.055 to 0.175.
The interval excludes zero.
Of the 200 paired questions, 157 agree, 33 favour position 0, and 10 favour position 4, so the discordance is 0.215.
The preregistered planning calibration maps that discordance to 1,248 questions for a five-point architecture interaction.
This is a planning calibration, not an independently identified power calculation.

Prompt length is 1,476 tokens on average in both gold conditions and is identical within each question, which the runner enforces during generation.
Measured prompt lengths across the selected 200 questions range from 1,294 to 1,746 tokens, so the longest prompt plus 32 new tokens stays inside the 2,048-token context.

### Scoring sensitivity

| Condition | `best_subspan_em` | First line only | Normalized exact match |
|---|---|---|---|
| `closed_book` | 0.095 | 0.080 | 0.030 |
| `gold_middle` | 0.225 | 0.170 | 0.095 |
| `gold_first` | 0.340 | 0.290 | 0.160 |
| `oracle` | 0.650 | 0.605 | 0.385 |

The ordering of the four conditions and the gap between the two gold positions hold under all three scoring variants.

Normalized exact match is computed on the first non-empty line rather than the whole generation.
Whole-generation normalized exact match was 0.000 in all 800 generations, including the oracle condition that scored 0.650 under the primary metric, because a base model continues generating past its answer.
That variant carries no sensitivity information, so the extracted first line is used instead.

### Reproducibility check

The tracer was run twice in separate processes with the same seeds and the same pinned revisions.
Across all 800 records, the prompts, model responses, primary scores, first-line scores, prompt token counts, generated token counts, and attached floor and ceiling values are identical.
The `run_id` values differ, as intended.

There were no scoring failures, no exclusions, and no context overflows.
The failures sidecar was not created because no generation failed.

## Artifacts

- `tracer.jsonl.gz`: 800 Phase 1 records, one per generation, 28 fields each.
- `positive-control.jsonl.gz`: 500 key-value control records.
- `summary.json`: condition accuracies, paired contrast, discordance, scoring variants, exclusion count.
- `figures/kv-position-curve.png`: key-value accuracy by slot with bootstrap intervals.
- `figures/phase1-condition-accuracy.png`: Phase 1 condition accuracies with bootstrap intervals.
- `figures/figures-summary.json`: the numbers behind both figures, including the paired contrast interval.
- `audit/audit-sample.jsonl`: 50 blinded generations, 13 `closed_book`, 13 `gold_first`, 12 `gold_middle`, 12 `oracle`.
- `audit/audit-key.jsonl`: the unblinding key, held separately.
- `audit/audit-form.md`: the four failure categories to record per item.
- `environment.json`: package versions, `nvidia-smi` output, and the git commit of the code that produced the run.

Raw generations were produced at commit `54dc62b`.
The two figures were regenerated at commit `5f0a5e4`, which only changed figure titles and axis labels, from the same JSONL records and without invoking the model.

The blinded rows contain the question, the model response, the correct answer, and the documents sorted by title.
Sorting by title makes the gold slot unrecoverable from the blinded file.
The model name, the gold position, and all scores appear only in the key file.

The number of documents in a blinded row identifies the two anchor conditions: 13 rows carry no documents and are `closed_book`, and 12 rows carry one document and are `oracle`.
The remaining 25 rows carry ten documents and are either `gold_first` or `gold_middle`, which the blinded file does not distinguish.
Issue #4 requires blinding the reviewer to the model and the gold position, and that holds.
Hiding the document count would require showing the auditor documents that were not in the prompt, which would make the extraction and hallucination categories unanswerable.

## Limits

Phase 1 measures two gold positions, 0 and 4.
The dense sweep across all ten positions belongs to Phase 2.

One model, one seed, one execution path, and one prompt template were used, so nothing here separates architecture from data or from prompt format.

The manual audit sample is generated but the human categorisation of formatting, extraction, hallucination, and truncation failures has not been carried out.

The floor of 0.095 and the ceiling of 0.650 bound what the position sweep can show on this model.
The ceiling means that on 35 percent of questions the model fails even when the gold document is the only document present.

The key-value control shows accuracy at slot 8 of 0.42 against 0.16 at slot 9, so the recency arm of that curve is not monotone.
Cause unidentified.

## Next step

Phase 2 replaces the two gold positions with all ten, adds `state-spaces/mamba-2.8b` and `state-spaces/mamba2-2.7b`, and applies the paired bootstrap with Holm correction across the primacy and recency contrasts.
The record schema and the analysis code in this phase already carry the fields that comparison needs.
