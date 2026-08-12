# Paper: Mixing Matters (NewInML @ NeurIPS 2026)

Source for the workshop submission.

## Files

- `mixing_matters.tex` - the paper, anonymized for double-blind review.
- `figures/` - figures pulled from the phase artifacts, self-contained for the build.
- `REFERENCE-RESEARCH.md` - the primary-source reference audit and insertion rationale.

## Build

The paper uses the NeurIPS 2026 workshop style. Download `neurips_2026.sty` (and its `.bst` if you prefer the natbib style) from the official template and place it beside `mixing_matters.tex`:

https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc

Then build from the `paper/` directory:

```bash
pdflatex mixing_matters
pdflatex mixing_matters
```

The source keeps exactly one bibliography environment and includes `checklist.tex` exactly once after the references.
The appendix is part of the same build and uses only copied experiment artifacts.

The default (no `[final]` option on the style) keeps the author block anonymized, as required by the NewInML double-blind process on OpenReview.

## Figure provenance

| Figure file | Source artifact |
|---|---|
| `phase2-curves.png` | `artifacts/phase2/report/position-curves.png` |
| `phase2-edges.png` | `artifacts/phase2/report/position-edges.png` |
| `phase3-attention-effect.png`, `phase3-curves.png` | `artifacts/phase3/report/` (main) |
| `phase4-scale-gap.png` | `artifacts/phase4/report/scale-primacy-gap.png` (phase-4) |
| `phase7-sink.png` | `artifacts/phase7-mechanisms/4c-sink-scan/report/sink-mass-by-layer.png` |
| `phase8-curves.png` | `artifacts/phase8/report/position-curves.png` |
| `phase5-corpus-effect.png` | `artifacts/phase5/report/corpus-effect.png` |
| `phase6-task-comparison-2048.png` | `artifacts/phase6/report/task-comparison-2048.png` |
| `phase7-attention-sink.png` | `artifacts/phase7-mechanisms/4c-sink-scan/report/sink-mass-by-layer.png` |
| `phase8-production-curves.png` | `artifacts/phase8/report/position-curves.png` |

Every number in the paper traces to a `*-summary.json` under `artifacts/`; see `artifacts/phase7-mechanisms/CUMULATIVE-FINDINGS.md` for the consolidated table with sources.

## Content map (phases to sections)

- Section 3 (measurement and statistics) - the ten-position harness, paired bootstrap, anchors, and executed versus pending controls.
- Section 4 (architecture comparisons) - the exploratory 2.8B interaction and the statistically uncertain matched 8B interaction.
- Section 5 (scope checks) - scale, corpus, RULER, and descriptive system comparisons.
- Sections 6 and 7 (mechanistic evidence and limitations) - sinks, probes, prompt interventions, and the remaining confirmatory gates.
- Appendix A (experiment artifacts) - exact copied figures for scale, corpus, synthetic retrieval, sink, and production-system experiments, with exploratory and causal boundaries in the captions.
