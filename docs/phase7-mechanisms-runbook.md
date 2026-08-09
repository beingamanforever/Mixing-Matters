# Phase 7 mechanisms runbook

Phase 7 tests *why* the primacy and recency arms appear where they do.
This runbook covers the compute-free lenses that already sit on top of the collected Phase 2 and Phase 4 sweeps.
The GPU-bound experiments (query-position ablation, sink-mass measurement, sink-blocking, linear probe) each have their own runbook and command.

## Compute-free analyses

Three lenses live in `src/mixing_matters/phase7.py` and share the Phase 2 paired-bootstrap machinery:

- `depth_trend` reads per-model primacy and recency edges from any set of gold-condition records, joins each `model_key` to its layer count (Pythia and Mamba families defined at the top of the module), and returns an ordered per-family list.
Answers the "primacy strengthens with depth" prediction in the workshop spec, in particular the Pythia 410m (24 layers, 1024 wide) vs Pythia 1B (16 layers, 2048 wide) depth-width contrast.
- `scoring_sensitivity` recomputes the same edges under three scoring variants that every sweep record already carries: `score` (best_subspan_em, primary), `score_normalized_em` (Liu et al. normalized exact match), and `score_first_line` (first-line extraction).
The output makes clear whether the sign or magnitude of the primary claim depends on the scorer.
- `length_sensitivity` bins questions into three equal-count prompt-length groups (tertile of median gold-prompt token count) and reruns the edge bootstrap inside each group.
Bounds the confound between prompt length and edge magnitude.

Every analysis returns 95 percent bootstrap intervals and Holm-adjusted p-values on the same two edges (primacy and recency), so the numbers are directly comparable to Phase 2/4/5/6/8.

## Build the report

Point `phase7-report` at any set of raw sweep files whose gold-condition records should join the mechanism lenses.
The three-variant scoring analysis needs the three score fields to be present on every record (Phase 2 and Phase 4 sweeps write them; Phase 8 sweeps also).

```bash
PYTHONPATH=src .venv/bin/python -m mixing_matters.cli phase7-report \
  --results \
    artifacts/phase7-mechanisms/inputs/pythia-160m-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/pythia-410m-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/pythia-1b-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/pythia-1.4b-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/pythia-2.8b-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/mamba-2.8b-sweep.jsonl.gz \
    artifacts/phase7-mechanisms/inputs/mamba2-2.7b-sweep.jsonl.gz \
  --output artifacts/phase7-mechanisms/report
```

The seven input sweeps are the Phase 2 pythia-2.8b/mamba-2.8b/mamba2-2.7b runs on the Pile, plus the Phase 4 Pythia sweeps at every registered size point.
They are bundled in `artifacts/phase7-mechanisms/inputs/` so this command reproduces the report against the tree, without needing to pull the Phase 4 branch or unzip anything by hand.

`read_jsonl` handles gzipped files transparently (see `src/mixing_matters/io.py`).
The command writes:

- `depth-primacy.png`: per-family primacy edge vs layer count.
- `scoring-sensitivity.png`: primacy edge under each of the three scoring variants.
- `length-sensitivity.png`: primacy edge per model per prompt-length bin.
- `phase7-summary.json`: full machine-readable summary including edges, CIs, and bin boundaries.

Overwriting an existing output raises `FileExistsError`; delete or move the target directory first if a rerun is intended.

## GPU-bound experiments

These do not live in `phase7.py` because they change the harness (prompt builder, forward-pass hook, or downstream analysis pipeline).
Each gets its own module and CLI subcommand once its sweep is planned; see the tracking issue in the PR opened for this branch.

- 7-4a query-position ablation: three additional prompt orders (question-first, bookend, gold-padded-after) run through the standard `sweep` command with a `--prompt-variant` selector.
- 7-4c sink-mass measurement: attention-hook capture of the token-0 attention share per layer per question.
- 7-4c sink-blocking ablation: mask-token-0 sweep on Nemotron-H hybrid attention layers.
- 7-4d linear probe: hidden-state extractor plus a balanced probe with a shuffled-label control.
- 7-4e prompt-template variation: alternative Liu et al. templates re-swept on Pythia-2.8B and Mamba-2.8B.
