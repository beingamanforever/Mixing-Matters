# Paper reconstruction audit

This report records how the NeurIPS 2026 workshop paper was reconstructed from the repository evidence on 2026-08-14.

## Evidence policy

The audit treated raw generations and machine-readable summaries as authoritative, followed by environment manifests, phase reports, and figures.
Narrative drafts were not used as numerical authority.
All reported ten-document QA sweeps use the 800-question exploratory split or a documented subset of it.
The frozen 1,855-question confirmatory split remains unopened.

## Phase conclusions

- Phase 1 establishes calibration and an executed positive control.
  The reported control is the edge-versus-middle key-value effect, not a first-slot-only contrast.
- Phase 2 supports a strong exploratory family-by-position interaction near 2.8B parameters.
  It does not isolate attention because model depth, positional encoding, state size, and checkpoint family differ.
- Phase 3 is the tightest architecture comparison.
  The hybrid-minus-pure primacy estimate is directionally consistent with Phase 2 but crosses zero after the prespecified correction.
- Phase 4 shows a descriptive family gap once the compared checkpoints can answer the task.
  The smallest Pythia checkpoint is at an answerability ceiling near zero, so the phase does not establish a causal scale threshold.
- Phase 5 finds no detectable edge-shape change after a corpus swap under a fixed Mamba architecture.
  The large floor shift prevents the stronger claim that training data do not matter.
- Phase 6 reproduces Pythia primacy on 2K-token synthetic retrieval.
  Both Mamba models saturate the task, and only summaries and plots are committed for this phase.
- Phase 7 provides prompt, sink, and probe evidence for hypotheses.
  The sink result is correlational, the probe does not prove content use, the bookend prompt partly lowers middle accuracy, and the invalid sink-block attempt is excluded.
- Phase 8 shows positive primacy in three additional 7B to 8B systems.
  It is a prevalence check rather than a matched architecture comparison.

## Paper decisions

The abstract and introduction lead with the scientific question, paired intervention, main finding, and confirmatory boundary instead of listing phase numbers.
The paper consistently distinguishes family interactions, matched but uncertain contrasts, descriptive scope checks, and mechanism hypotheses.
References were checked against primary proceedings, publisher, dataset-card, or preprint records and expanded only where they sharpen the cross-architecture boundary.
The official double-blind NeurIPS 2026 workshop style and checklist are retained.
References precede the appendix, and the checklist remains last.

## Figure system

All paper figures are regenerated from committed JSON summaries by `paper/generate_figures.py`.
SVG is the canonical output, and a PDF companion is produced by the same plotting call for LaTeX inclusion.
The entire paper uses vivid blue for attention-based models and orange for state-space models, with marker shape, fill, and line style encoding model or condition.
No figure uses an embedded raster image or an additional categorical color.

## Remaining boundaries

The matched Phase 3 producing manifest records a dirty source tree.
Phase 6 lacks committed raw generations and an environment manifest.
Negative and distractor-order control outputs are absent.
The blinded audit sample has no completed human labels.
Some converted checkpoint mirrors do not declare complete license metadata in their model cards.
Elapsed time and aggregate compute were not logged consistently enough to report a defensible total.
