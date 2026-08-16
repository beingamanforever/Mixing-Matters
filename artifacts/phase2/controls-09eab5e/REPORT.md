# Phase 2 certification controls

## Status

Both pre-registered certification controls passed on the single NVIDIA A40 host.
The run used the immutable repository commit `09eab5e62136a0a9b828853866690acae84c5ea6` and completed without a failure sidecar or failed marker.
The post-run receipt confirms that no Phase 2 process remained, GPU model-process memory returned to 0 MiB, and the remote repository stayed clean.

## Protocol

The controls use `EleutherAI/pythia-2.8b` at revision `2a259cdd96a4beb1cdf467512e3904197345f6a9`.
They use the pinned NQ dataset SHA-256 `192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9` and only the 200-question exploratory sample selected with seed `240521`.
Decoding is greedy with temperature 0, top-p 1, no top-k cutoff, one beam, and 32 maximum new tokens.
The runtime was Python 3.12.3, torch 2.7.1+cu126, transformers 4.57.1, CUDA 12.6, bfloat16, and eager attention.
The full package, GPU, repository, dataset, and protocol provenance is preserved in [environment.json](environment.json).

## Positive control

The fresh key-value control contains exactly 500 records and passed before either certification control was accepted.
Accuracy was 0.94 at slot 0 and 0.16 at slot 9.
The mean edge accuracy minus mean middle accuracy was 0.38.

## Sham-gold negative control

The negative control contains exactly 2,000 records from 200 complete ten-position question bundles.
Sham-gold accuracy was 0.1235 against a closed-book floor of 0.0950, a difference of 0.0285 within the fixed 0.05 gate.
The primacy estimate was 0.0050 with bootstrap interval [-0.0325, 0.0425].
The recency estimate was -0.0175 with bootstrap interval [-0.0500, 0.0125].
Both intervals contain zero, so the fixed flatness gates passed.

## Distractor-order control

The order control contains exactly 1,800 records from 200 questions, positions 0, 4, and 9, and permutations 0, 1, and 2.
Permutation accuracies were 0.2850, 0.3000, and 0.2767.
The maximum permutation spread was 0.0233, below the fixed 0.10 gate.
Accuracy by gold position was 0.3217 at position 0, 0.2350 at position 4, and 0.3050 at position 9.

## Interpretation

The results certify that removing the real gold document eliminates the measured position edge and that distractor permutation does not materially change pooled accuracy under the committed gates.
These are Pythia pipeline certification controls and do not test the Megatron 8B checkpoints.
The 200 questions belong to the exploratory split, so no held-out confirmation was consumed.

## Preserved evidence

The authoritative machine-readable result is [summary.json](summary.json).
The completion marker, combined log, per-stage logs, exact records, and [post-run receipt](phase2-controls-09eab5e-post-run.txt) are preserved in this directory.
JSONL files and logs are stored as deterministic gzip archives without changing their decompressed bytes.
All committed artifact hashes are recorded in [SHA256SUMS](SHA256SUMS).
