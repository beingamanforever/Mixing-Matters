<div align="center">

<a href="https://beingamanforever.github.io/Mixing-Matters/">
  <img src="web/assets/og.png" width="100%" alt="Mixing Matters? Evidence and Its Limits for Position Bias Across Sequence Mixers. Accepted at New in ML, NeurIPS 2026.">
</a>

# Mixing Matters?

**Evidence and Its Limits for Position Bias Across Sequence Mixers**

Aman Behera<sup>1</sup>, Namit Solanki<sup>2</sup>, Mehul Anand<sup>3</sup>

<sup>1</sup>IIT Roorkee &nbsp; <sup>2</sup>AISSMS IOIT, Pune &nbsp; <sup>3</sup>Independent Researcher

[![Accepted at New in ML @ NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-New%20in%20ML%20·%20Accepted-009e73?style=flat-square)](https://beingamanforever.github.io/Mixing-Matters/)
[![Project page](https://img.shields.io/badge/Project-page-0072b2?style=flat-square)](https://beingamanforever.github.io/Mixing-Matters/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](paper/mixing-matters-newinml-2026.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-229%2C700%20generations-e69f00?style=flat-square)](dataset/)
[![License](https://img.shields.io/badge/License-Apache%202.0-4f5459?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](pyproject.toml)

[**Project page**](https://beingamanforever.github.io/Mixing-Matters/) ·
[**Paper**](paper/mixing-matters-newinml-2026.pdf) ·
[**Dataset**](dataset/) ·
[**Evidence ledger**](paper/EXPERIMENT-SOURCE-OF-TRUTH.md) ·
[**Artifacts**](artifacts/) ·
[**Runbooks**](docs/)

</div>

> **TL;DR.**
> "Lost in the middle" was measured on Transformers, but it is often quoted as a fact about long context itself.
> We move one answer-bearing passage through ten positions of a fixed document set and compare Transformer, state-space, and hybrid models on identical questions.
> Pythia favours the opening of the context while two Mamba checkpoints do not, and all of them favour the end.
> The measured curves differ by model family, but the study does not isolate architecture as the cause.

## Study design

```text
            1     2     3     4     5     6     7     8     9     10
Position 1  [■]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   Question
Position 5  [ ]   [ ]   [ ]   [ ]   [■]   [ ]   [ ]   [ ]   [ ]   [ ]   Question
Position 10 [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [ ]   [■]   Question
            └─ start ─┘             └─ middle ─┘             └── end ──┘
```

The released evaluation contains 2,655 multi-document questions, each with one answer-bearing passage (■) and nine fixed distractors.
We use 800 questions for exploratory analysis and leave 1,855 questions untouched for confirmation.
For each question, we move the answer-bearing passage through positions 1 to 10 while holding the question, distractors, prompt template, and decoding configuration fixed.
The protocol is derived from the [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) evaluation.

- **Primacy** is mean accuracy at positions 1 and 2 minus mean accuracy at positions 5 and 6.
- **Recency** is mean accuracy at positions 9 and 10 minus mean accuracy at positions 5 and 6.
- **Inference** uses 10,000 bootstrap resamples of complete question bundles and Holm correction across the two edge tests.

## Main results

<table>
  <tr>
    <td width="50%"><img src="paper/figures/paper-phase2-position.svg" width="100%" alt="Accuracy by evidence position for Pythia 2.8B, Mamba 2.8B, and Mamba-2 2.7B"></td>
    <td width="50%"><img src="paper/figures/paper-phase3-position.svg" width="100%" alt="Accuracy by evidence position for matched pure and hybrid Mamba-2 8B models"></td>
  </tr>
  <tr>
    <td align="center"><b>Primary comparison</b> at 2.8B scale</td>
    <td align="center"><b>Matched 8B</b> pure and hybrid checkpoints</td>
  </tr>
</table>

**Primary comparison.**
Pythia-2.8B has a `+5.19` percentage-point primacy edge, while Mamba-2.8B and Mamba-2 2.7B have edges of `-0.13` and `-1.81` points.
The paired Pythia-minus-Mamba primacy differences are `+5.31` and `+7.00` points, both with Holm `p < 0.0001`.
All three models show positive recency edges.
See the [Phase 2 summary](artifacts/phase2/report/phase2-summary.json).

**Matched 8B comparison.**
The hybrid model has a larger primacy estimate than pure Mamba-2, but the paired hybrid-minus-pure effect is `+1.88` points with a 95% confidence interval of `[-0.56, +4.44]` and Holm `p = 0.1442`.
The direction is consistent with the primary comparison, but the paired effect is statistically uncertain and does not establish that attention caused the difference.
This released-checkpoint contrast also changes the models' MLP composition, so it is not an attention-only intervention.
The original run recorded a dirty producing tree, while a later clean rerun at commit `33d6bb5` reproduced the summary byte-for-byte.
See the [Phase 3 summary](artifacts/phase3/report/phase3-summary.json), [clean-rerun report](artifacts/phase3/rerun-33d6bb5/REPORT.md), and the matched model release of [Waleffe et al. (2024)](https://arxiv.org/abs/2406.07887).

## Controls and scope

<table>
  <tr>
    <td width="50%"><img src="paper/figures/paper-phase1-calibration.svg" width="100%" alt="End-to-end calibration and key-value positive-control results"></td>
    <td width="50%"><img src="paper/figures/paper-phase4-scale.svg" width="100%" alt="Primacy effects and paired Pythia-minus-Mamba differences across model scales"></td>
  </tr>
  <tr>
    <td align="center"><b>Calibration</b> and positive control</td>
    <td align="center"><b>Scale</b> across five size pairs</td>
  </tr>
</table>

**Calibration and scale.**
The end-to-end calibration and key-value control show that the harness can detect a known position effect.
Across five approximate size pairs, the family gap is near zero at the two smallest scales and appears in the three larger pairs, but capability and architecture remain confounded.
See the [Phase 1 summary](artifacts/phase1/summary.json), [Phase 4 summary](artifacts/phase4/report/phase4-summary.json), and the original [Pythia](https://proceedings.mlr.press/v202/biderman23a.html), [Mamba](https://openreview.net/forum?id=tEYskw1VY2), and [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html) papers.

<table>
  <tr>
    <td width="50%"><img src="paper/figures/paper-phase5-corpus.svg" width="100%" alt="Mamba 2.8B position curves after changing the pretraining corpus"></td>
    <td width="50%"><img src="paper/figures/paper-phase6-task.svg" width="100%" alt="Pythia and Mamba primacy and recency effects on multi-document QA and 2K-token synthetic needle retrieval"></td>
  </tr>
  <tr>
    <td align="center"><b>Corpus</b> control</td>
    <td align="center"><b>Task</b> check on synthetic retrieval</td>
  </tr>
</table>

**Corpus and task checks.**
Changing the Mamba-2.8B pretraining corpus changes overall accuracy but produces no detectable primacy or recency shape change in this executed contrast.
On [RULER](https://openreview.net/forum?id=kIoBbc76Sy) at 2K tokens, Pythia reproduces a primacy edge, while both Mamba models saturate at perfect accuracy and therefore do not support a mixer comparison on that task.
See the [Phase 5 summary](artifacts/phase5/report/phase5-summary.json) and [Phase 6 summary](artifacts/phase6/report/phase6-summary.json).

<details>
<summary><b>Mechanistic evidence and production systems</b></summary>
<br>

<p align="center">
  <img src="paper/figures/paper-phase7-mechanisms.svg" width="100%" alt="Attention-sink, position-probe, and prompt-sensitivity analyses">
</p>

**Mechanistic evidence.**
Late-layer [attention-sink](https://openreview.net/forum?id=NG7sS51zVF) mass tracks Pythia primacy across scale, position remains linearly decodable in both model families, and prompt variants show substantially different per-condition edges.
These results are correlational and diagnostic, not evidence of a causal mechanism.
See the [Phase 7 summary](artifacts/phase7-mechanisms/report/phase7-summary.json).

<p align="center">
  <img src="paper/figures/paper-phase8-production.svg" width="75%" alt="Position curves for Nemotron-H 8B, Llama 3.1 8B, and Qwen 2.5 7B">
</p>

**Production systems.**
Nemotron-H-8B, Llama-3.1-8B, and Qwen2.5-7B all show positive primacy edges, but their many architectural and training differences make this a descriptive prevalence check rather than an architecture test.
See the [Phase 8 summary](artifacts/phase8/report/phase8-summary.json).

</details>

## Evidence boundary

> [!IMPORTANT]
> - All reported ten-document QA results are exploratory, and the 1,855-question confirmatory split remains unopened.
> - The committed sham-gold and distractor-order controls passed on a 200-question exploratory Pythia sample, but they were not run on the 8B Megatron checkpoints, and the prepared manual audit has no human labels.
> - The matched 8B paired effect is statistically uncertain, and the mechanism analyses do not support a causal attention claim.
> - Confidence intervals quantify variation across questions for fixed checkpoints, prompts, and decoding settings, not variation across training seeds or model checkpoints.

## Quick start

The committed summaries regenerate every canonical SVG and PDF figure deterministically.

```bash
uv sync --extra test
uv run pytest -q
uv run python paper/generate_figures.py
```

GPU execution, checkpoint validation, and phase-specific analysis commands are documented in the [runbooks](docs/).
The [figure tests](tests/test_paper_figures.py) check deterministic regeneration, expected labels, and source-summary provenance.

## Released dataset

The released dataset flattens 229,700 selected committed model generations across 17 pinned checkpoints, 10 evidence positions, 4 prompt variants, and 2 tasks into one schema in [`dataset/`](dataset/), together with 280,000 per-layer attention-sink measurements.
It excludes the uncommitted Phase 6 synthetic-retrieval generations, the later Pythia certification-control runs, and the duplicate clean Phase 3 rerun.
Field documentation and collection details are in the [datasheet](dataset/DATASHEET.md).

| File | Contents |
| :-- | :-- |
| [`generations.jsonl.gz`](dataset/generations.jsonl.gz) | One row per model generation |
| [`position_accuracy.csv`](dataset/position_accuracy.csv) | Accuracy by model, condition, and evidence position |
| [`attention_sink.jsonl.gz`](dataset/attention_sink.jsonl.gz) | Per-layer attention-sink measurements |
| [`runs.csv`](dataset/runs.csv) | Pinned checkpoints and run metadata |

```bash
uv run mixing-matters build-dataset --output dataset
```

The builder reads only committed artifacts, so any clone reproduces the same files.

## Project page

The [project page](https://beingamanforever.github.io/Mixing-Matters/) presents these results interactively.
Its source is in [`web/`](web/), and every number it renders is generated from the committed phase summaries.

```bash
uv run mixing-matters build-site-data --output web/data/results.json
python3 -m http.server 8123 --directory web
```

Pushing to `main` deploys it through [the Pages workflow](.github/workflows/pages.yml), which regenerates the page data and stages the paper PDF and the dataset alongside it.

## Repository layout

```text
src/mixing_matters/   evaluation harness, phase analyses, dataset and site builders
artifacts/            committed per-phase summaries and reports
dataset/              released generations, accuracies, and datasheet
paper/                paper source, figures, and evidence ledger
web/                  project page
docs/                 GPU runbooks for each phase
tests/                end-to-end and figure regeneration tests
```

## Citation

The paper builds on the position-intervention protocol of [Liu et al. (2024)](https://aclanthology.org/2024.tacl-1.9/) and evaluates models introduced by [Biderman et al. (2023)](https://proceedings.mlr.press/v202/biderman23a.html), [Gu and Dao (2024)](https://openreview.net/forum?id=tEYskw1VY2), [Dao and Gu (2024)](https://proceedings.mlr.press/v235/dao24a.html), and [Waleffe et al. (2024)](https://arxiv.org/abs/2406.07887).
The complete scholarly bibliography is included in the [paper](paper/mixing-matters-newinml-2026.pdf).

If you use the study, the released dataset, or the evaluation harness, please cite the paper.

```bibtex
@inproceedings{behera2026mixing,
  title     = {Mixing Matters? Evidence and Its Limits for Position Bias
               Across Sequence Mixers},
  author    = {Behera, Aman and Solanki, Namit and Anand, Mehul},
  booktitle = {New in ML Workshop at NeurIPS},
  year      = {2026},
  url       = {https://beingamanforever.github.io/Mixing-Matters/}
}
```

## Acknowledgments

We thank the Indian Institute of Technology Roorkee (IIT Roorkee) for providing the computational resources that supported this research.

## License

This project is released under the [Apache License 2.0](LICENSE).
