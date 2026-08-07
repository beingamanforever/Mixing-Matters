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
    "mamba2-8b": ModelSpec(
        key="mamba2-8b",
        repo="nvidia/mamba2-8b-3t-4k",
        revision="main",
        family="mamba2",
    ),
    "mamba2-hybrid-8b": ModelSpec(
        key="mamba2-hybrid-8b",
        repo="nvidia/mamba2-hybrid-8b-3t-4k",
        revision="main",
        family="mamba2",
    ),
    # Phase 4: Scale and Family Trend
    "pythia-160m": ModelSpec(
        key="pythia-160m", repo="EleutherAI/pythia-160m", revision="main", family="pythia"
    ),
    "pythia-410m": ModelSpec(
        key="pythia-410m", repo="EleutherAI/pythia-410m", revision="main", family="pythia"
    ),
    "pythia-1b": ModelSpec(
        key="pythia-1b", repo="EleutherAI/pythia-1b", revision="main", family="pythia"
    ),
    "pythia-1.4b": ModelSpec(
        key="pythia-1.4b", repo="EleutherAI/pythia-1.4b", revision="main", family="pythia"
    ),
    "mamba-130m": ModelSpec(
        key="mamba-130m", repo="state-spaces/mamba-130m-hf", revision="main", family="mamba"
    ),
    "mamba-370m": ModelSpec(
        key="mamba-370m", repo="state-spaces/mamba-370m-hf", revision="main", family="mamba"
    ),
    "mamba-790m": ModelSpec(
        key="mamba-790m", repo="state-spaces/mamba-790m-hf", revision="main", family="mamba"
    ),
    "mamba-1.4b": ModelSpec(
        key="mamba-1.4b", repo="state-spaces/mamba-1.4b-hf", revision="main", family="mamba"
    ),
    # Phase 5: Training Data Control
    "mamba-2.8b-slimpj": ModelSpec(
        key="mamba-2.8b-slimpj",
        repo="state-spaces/mamba-2.8b-slimpj",
        revision="main",
        family="mamba",
    ),
}


def spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model key {key!r}; valid keys: {valid}") from None
