"""Declarative registry of the models compared in the Phase 2 architecture sweep.

Loading logic (tokenizer, weights, execution path) lives in ``run.py``; this
module only records which models exist and how to pin them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    family: str


MODELS: dict[str, ModelSpec] = {
    "pythia-2.8b": ModelSpec(
        key="pythia-2.8b",
        repo="EleutherAI/pythia-2.8b",
        revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
        family="pythia",
    ),
    "mamba-2.8b": ModelSpec(
        key="mamba-2.8b",
        repo="state-spaces/mamba-2.8b-hf",
        revision="96c48e0292b63f5346b6d30061af2551f7101e26",
        family="mamba",
    ),
    "mamba2-2.7b": ModelSpec(
        key="mamba2-2.7b",
        repo="AntonV/mamba2-2.7b-hf",
        revision="ef542707386fa9ec86bbf8a35ed2952af84bf566",
        family="mamba2",
    ),
}


def spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model key {key!r}; valid keys: {valid}") from None
