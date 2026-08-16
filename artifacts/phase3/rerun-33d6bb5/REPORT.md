# Clean Phase 3 rerun

## Status

The clean A40 rerun completed successfully at repository commit `33d6bb5211d02d5ef5040086b18298fdacf31778`.
The immutable remote run directory was `/root/outputs/phase3-clean-33d6bb5`.
The completion log ended with `Phase 3 run complete: /root/outputs/phase3-clean-33d6bb5`.
The preserved [combined completion log](phase3-combined.log.gz) expands to 130,338 bytes and has SHA-256 `092987365111cc960320dc7295220db4d6ac152b43d68238b68fd115df0f4b93`.
The [post-run receipt](post-run-receipt.txt) records that the detached process exited, the A40 returned to 0 MiB in use, both remote repositories remained clean, and no failure sidecars were present.

## Provenance

The Mixing-Matters commit was `33d6bb5211d02d5ef5040086b18298fdacf31778`.
The Megatron-LM commit was `df61e60bf5670b1196fcae2264311401d3bb82db`.
The NQ dataset SHA-256 was `192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9`.
The PIQA validation dataset SHA-256 was `61533005e22f175534909b1ec8eacb6da03c233933558d1bafb15787453b1f55`.
The full package, GPU, checkpoint, launcher, and command provenance is recorded in [environment.json](environment.json).

## Validation gates

The pure Mamba-2 8B checkpoint scored 79.27 on all 1,838 PIQA validation examples against a target of 79.82, for a -0.55 point delta and a pass.
The hybrid Mamba-2 8B checkpoint scored 79.76 against a target of 79.65, for a +0.11 point delta and a pass.
Each key-value control contains exactly 500 records.
Each position sweep contains exactly 9,600 records, consisting of 800 complete question bundles with 12 conditions each.
The key-value control is a non-gating runtime diagnostic and is not used as evidence for the QA position result.

## Results

The regenerated `phase3-summary.json` is byte-identical to the prior report summary.
The pure model's primacy edge was 0.0081 with 95% CI [-0.0088, 0.0244] and Holm-adjusted p = 0.3624.
The pure model's recency edge was 0.0775 with 95% CI [0.0575, 0.0975] and Holm-adjusted p < 0.0001.
The hybrid model's primacy edge was 0.0269 with 95% CI [0.0075, 0.0469] and Holm-adjusted p = 0.0070.
The hybrid model's recency edge was 0.0494 with 95% CI [0.0312, 0.0681] and Holm-adjusted p < 0.0001.
The hybrid-minus-pure primacy contrast was 0.0188 with 95% CI [-0.0056, 0.0444] and Holm-adjusted p = 0.1442.
The hybrid-minus-pure recency contrast was -0.0281 with 95% CI [-0.0550, -0.0012], raw p = 0.0446, and Holm-adjusted p = 0.0892.
The pure model's floor and ceiling accuracies were 0.2575 and 0.6512.
The hybrid model's floor and ceiling accuracies were 0.2425 and 0.6162.

## Interpretation

The comparison is a composite architecture contrast between the released pure and hybrid checkpoints.
The released checkpoints use different attention and MLP composition, so the comparison does not isolate an attention-only causal effect.
The exploratory 800-question result exactly reproduces the prior summary, but the held-out confirmatory evaluation remains unrun.

## Preserved outputs

The exact output hashes are recorded in [SHA256SUMS](SHA256SUMS).
The analysis outputs are [phase3-summary.json](report/phase3-summary.json), [position-curves.png](report/position-curves.png), [position-edges.png](report/position-edges.png), and [attention-effect.png](report/attention-effect.png).
The per-stage validation, control, and sweep logs are preserved losslessly as deterministic gzip archives beside the JSONL outputs.
