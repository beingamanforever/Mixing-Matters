# Phase 7 4c: sink mass under query-position variants (supplementary)

Token-0 attention share per layer for `pythia-2.8b` under two of the 4a query-position variants, as a supplement to the baseline sink-mass scan in `../4c-sink-scan/`.

- `pythia-2.8b-question_first.jsonl.gz`: question placed before the documents.
- `pythia-2.8b-bookend.jsonl.gz`: question placed before and after the documents.

The `gold_padded` variant sink-scan was attempted but ran out of memory on the 16GB T4 (the 128 pad tokens lengthen the prompt and `output_attentions` materializes the full attention matrix), so it is not included. The baseline sink-scan and the Pythia-scale sweep in `../4c-sink-scan/` carry the main sink-mass finding; these two variants are provided as raw data for anyone extending the analysis to how the sink relocates when query-relevant tokens are moved to the head of the prompt.
