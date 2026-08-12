# Reference research and citation audit

This note records the primary-source checks used to expand and correct the paper bibliography on 2026-08-12.
The additions were selected only when they support a claim that appears in the paper.

## Added primary references and insertion rationale

- `gu2022s4` cites the [ICLR paper](https://openreview.net/forum?id=uYLFoz1vlAC) as the structured state-space precursor in the sequence-mixer discussion.
- `peng2023rwkv` cites the [ACL Anthology record](https://aclanthology.org/2023.findings-emnlp.936/) as a recurrent alternative to dense attention.
- `poli2023hyena` cites the [PMLR paper](https://proceedings.mlr.press/v202/poli23a.html) as a long-convolution alternative.
- `de2024griffin` cites the [primary preprint](https://arxiv.org/abs/2402.19427) as an example of mixing gated recurrences with local attention.
- `su2024roformer` cites the [publisher record](https://doi.org/10.1016/j.neucom.2023.127063) to identify rotary position embeddings as a model-family confound.
- `press2022alibi` cites the [ICLR paper](https://openreview.net/forum?id=R8sQPpGCv0) to identify linear attention biases as another positional design.
- `kwiatkowski2019nq` cites the [ACL Anthology record](https://aclanthology.org/Q19-1026/) for the Natural Questions lineage of the evaluation data.
- `rajpurkar2016squad` cites the [ACL Anthology record](https://aclanthology.org/D16-1264/) for the normalized exact-match convention.
- `efron1979bootstrap` cites the [journal record](https://doi.org/10.1214/aos/1176344552) for bootstrap resampling.
- `holm1979multiple` cites the [journal record](https://www.jstor.org/stable/4615733) for sequential multiple-test correction.
- `bai2024longbench` cites the [ACL Anthology record](https://aclanthology.org/2024.acl-long.172/) as a broad long-context benchmark.
- `shaham2023zeroscrolls` cites the [ACL Anthology record](https://aclanthology.org/2023.findings-emnlp.536/) as a zero-shot long-text benchmark.
- `li2024loogle` cites the [ACL Anthology record](https://aclanthology.org/2024.acl-long.859/) as a benchmark of long-range dependency understanding.
- `jiang2024longllmlingua` cites the [ACL Anthology record](https://aclanthology.org/2024.acl-long.91/) for prompt compression and position-aware context use.
- `xu2024recomp` cites the [ICLR paper](https://openreview.net/forum?id=mlJLVigNHp) for selective retrieval compression.
- `agrawal2024rr` cites the [ACL Anthology record](https://aclanthology.org/2024.findings-emnlp.742/) for retrieval and rereading over long documents.
- `shi2023distracted` cites the [PMLR paper](https://proceedings.mlr.press/v202/shi23a.html) to motivate holding irrelevant documents fixed across positions.
- `hewitt2019probes` cites the [ACL Anthology record](https://aclanthology.org/D19-1275/) to justify shuffled-label controls for diagnostic probes.
- `sun2024massive` cites the [primary preprint](https://arxiv.org/abs/2402.17762) to place attention-sink measurements beside related activation phenomena.

## Corrected venue metadata

- `an2024make` is recorded as a NeurIPS 2024 paper rather than only as an arXiv preprint.
- `gu2023mamba` is recorded as a COLM 2024 paper rather than only as an arXiv preprint.
- `hsieh2024ruler` is recorded as a COLM 2024 paper rather than only as an arXiv preprint.
- `waleffe2024empirical` remains an arXiv report because the audited source did not establish a proceedings venue.
- `nvidia2025nemotronh` remains a technical report because the audited source did not establish a proceedings venue.

## Thirty-six-reference audit

The bibliography defines exactly 36 unique keys after adding 19 references to the original 17.
The defined keys are `agrawal2024rr`, `an2024make`, `bai2024longbench`, `biderman2023pythia`, `dao2024mamba2`, `de2024griffin`, `dubey2024llama`, `efron1979bootstrap`, `gao2020pile`, `gu2022s4`, `gu2023mamba`, `hewitt2019probes`, `holm1979multiple`, `hsieh2024found`, `hsieh2024ruler`, `jiang2024longllmlingua`, `kwiatkowski2019nq`, `li2024loogle`, `lieber2024jamba`, `liu2024lost`, `nvidia2025nemotronh`, `peng2023rwkv`, `peysakhovich2023attention`, `poli2023hyena`, `press2022alibi`, `rajpurkar2016squad`, `shaham2023zeroscrolls`, `shi2023distracted`, `soboleva2023slimpajama`, `su2024roformer`, `sun2024massive`, `vaswani2017attention`, `waleffe2024empirical`, `xiao2024efficient`, `xu2024recomp`, and `yang2024qwen2`.
Every defined key appears in a `\citep` or `\citet` command outside the bibliography.
No citation key used in the prose is undefined.
The citations support dataset lineage, evaluation practice, sequence mixers, positional mechanisms, long-context benchmarks, context interventions, distractor controls, probes, or the exact model families studied.
No reference was added solely to increase the bibliography count.
