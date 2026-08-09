# Paper: Mixing Matters (NewInML @ NeurIPS 2026)

Source for the workshop submission.

## Files

- `mixing_matters.tex` - the paper, anonymized for double-blind review.
- `figures/` - figures pulled from the phase artifacts, self-contained for the build.

## Build

The paper uses the NeurIPS 2026 workshop style. Download `neurips_2026.sty` (and its `.bst` if you prefer the natbib style) from the official template and place it beside `mixing_matters.tex`:

https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc

Then:

```bash
pdflatex mixing_matters
pdflatex mixing_matters
```

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

Every number in the paper traces to a `*-summary.json` under `artifacts/`; see `artifacts/phase7-mechanisms/CUMULATIVE-FINDINGS.md` for the consolidated table with sources.

## Content map (phases to sections)

- Section 2 (measurement/stats/controls) - the harness, the paired bootstrap, Phase 1 anchors and the three controls.
- Section 3 (architecture contrast) - Phase 2 (matched 2.8B) and Phase 3 (matched 8B attention isolation).
- Section 4 (robustness) - Phase 4 scale, Phase 5 corpus, Phase 6 task, Phase 8 production systems.
- Section 5 (mechanism) - Phase 7: depth (4b), artifacts (4e), sinks (4c), probe (4d), query-position/training-free intervention (4a).
