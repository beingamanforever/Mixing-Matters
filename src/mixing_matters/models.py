"""Declarative registry of the models compared in the position sweeps.

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
    # Phase 4 groups models into matched size points. ``scale_pair`` labels the
    # Mamba and Pythia model that sit at the same point on the scale axis, and
    # ``params_millions`` is the nominal parameter count used to order that axis.
    # Both are None for models that do not take part in the scale sweep.
    scale_pair: str | None = None
    params_millions: int | None = None
    # Phase 5 holds the architecture fixed and changes the pretraining corpus.
    # ``training_corpus`` records that corpus, and ``data_pair`` labels the two
    # models that share an architecture and differ only in corpus. Both are None
    # for models outside the data-control sweep.
    training_corpus: str | None = None
    data_pair: str | None = None
    # Phase 3 holds training data, scale, tokenizer, depth, and positional
    # encoding fixed. ``arch_pair`` labels the released pure and hybrid Mamba-2
    # checkpoints, which differ in both attention and MLP composition. The
    # field name is retained for artifact compatibility.
    arch_pair: str | None = None
    # Phase 8 groups a set of complete production 7-8B systems into a
    # descriptive comparison. ``phase8_system`` labels a model as part of that
    # sweep so callers can iterate the group without hardcoding keys.
    # Multiple variables move together inside this group by design, so it is
    # not a matched control, unlike ``data_pair``/``scale_pair``.
    phase8_system: str | None = None
    # ``format`` is "hf" for checkpoints transformers can load directly and
    # "mamba_ssm" for original state-spaces checkpoints whose config carries no
    # model_type. A "mamba_ssm" checkpoint must be converted to an HF-format
    # directory before ``run.Generator`` can load it (see ``convert.py``); this
    # keeps every Mamba run on the one transformers CUDA-kernel path so the only
    # variable that changes across the Phase 5 contrast is the training corpus,
    # and "megatron" for checkpoints published only as Megatron-LM distributed
    # checkpoints, loaded through a separate Megatron backend and never by
    # transformers.
    format: str = "hf"


MODELS: dict[str, ModelSpec] = {
    "pythia-160m": ModelSpec(
        key="pythia-160m",
        repo="EleutherAI/pythia-160m",
        revision="50f5173d932e8e61f858120bcb800b97af589f46",
        family="pythia",
        scale_pair="130m-160m",
        params_millions=160,
    ),
    "mamba-130m": ModelSpec(
        key="mamba-130m",
        repo="state-spaces/mamba-130m-hf",
        revision="1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
        family="mamba",
        scale_pair="130m-160m",
        params_millions=130,
    ),
    "pythia-410m": ModelSpec(
        key="pythia-410m",
        repo="EleutherAI/pythia-410m",
        revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="pythia",
        scale_pair="370m-410m",
        params_millions=410,
    ),
    "mamba-370m": ModelSpec(
        key="mamba-370m",
        repo="state-spaces/mamba-370m-hf",
        revision="b519127f5bfaaa1c27dd938dad051ec360972b23",
        family="mamba",
        scale_pair="370m-410m",
        params_millions=370,
    ),
    "pythia-1b": ModelSpec(
        key="pythia-1b",
        repo="EleutherAI/pythia-1b",
        revision="f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
        family="pythia",
        scale_pair="790m-1b",
        params_millions=1000,
    ),
    "mamba-790m": ModelSpec(
        key="mamba-790m",
        repo="state-spaces/mamba-790m-hf",
        revision="9822dd4b76af2bd9099b6ce2f19efd8329189a7e",
        family="mamba",
        scale_pair="790m-1b",
        params_millions=790,
    ),
    "pythia-1.4b": ModelSpec(
        key="pythia-1.4b",
        repo="EleutherAI/pythia-1.4b",
        revision="fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
        family="pythia",
        scale_pair="1.4b-1.4b",
        params_millions=1400,
    ),
    "mamba-1.4b": ModelSpec(
        key="mamba-1.4b",
        repo="state-spaces/mamba-1.4b-hf",
        revision="6e46eae61c27280517feef46f536d16b91076f08",
        family="mamba",
        scale_pair="1.4b-1.4b",
        params_millions=1400,
    ),
    "pythia-2.8b": ModelSpec(
        key="pythia-2.8b",
        repo="EleutherAI/pythia-2.8b",
        revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
        family="pythia",
        scale_pair="2.8b-2.8b",
        params_millions=2800,
    ),
    "mamba-2.8b": ModelSpec(
        key="mamba-2.8b",
        repo="state-spaces/mamba-2.8b-hf",
        revision="96c48e0292b63f5346b6d30061af2551f7101e26",
        family="mamba",
        scale_pair="2.8b-2.8b",
        params_millions=2800,
        training_corpus="pile",
        data_pair="mamba-2.8b-corpus",
    ),
    # Phase 5 data control: same Mamba architecture as ``mamba-2.8b`` above,
    # trained on SlimPajama instead of the Pile. Published only in the original
    # state-spaces format (config.json + pytorch_model.bin, no model_type), so
    # it is converted to an HF-format directory before loading; see convert.py.
    "mamba-2.8b-slimpj": ModelSpec(
        key="mamba-2.8b-slimpj",
        repo="state-spaces/mamba-2.8b-slimpj",
        revision="a7bdd41af90ca0cc4ecfbd967e2ec28f1954b915",
        family="mamba",
        params_millions=2800,
        training_corpus="slimpajama",
        data_pair="mamba-2.8b-corpus",
        format="mamba_ssm",
    ),
    "mamba2-2.7b": ModelSpec(
        key="mamba2-2.7b",
        repo="AntonV/mamba2-2.7b-hf",
        revision="ef542707386fa9ec86bbf8a35ed2952af84bf566",
        family="mamba2",
    ),
    # Phase 3 released-checkpoint contrast: pure Mamba-2 versus the same NVIDIA
    # checkpoint family with a different attention and MLP composition. Both are
    # published only as Megatron-LM distributed checkpoints that transformers
    # cannot load, so both run through the Megatron backend (see
    # scripts/megatron_sweep.py) on one shared execution path, which is the
    # the most tightly matched within-phase contrast in the study. The pure
    # model is gated on a published zero-shot benchmark (PIQA 79.82, Waleffe
    # et al. 2024, Table 3) and the hybrid on its own published number (PIQA
    # 79.65, Table 7) via scripts/megatron_validate.py before its sweep,
    # matching how the Phase 2 and Phase 5 conversions were validated.
    "mamba2-8b": ModelSpec(
        key="mamba2-8b",
        repo="nvidia/mamba2-8b-3t-4k",
        revision="b915550c63ba9359f88f44d1f6a600d85af27302",
        family="mamba2",
        format="megatron",
        arch_pair="mamba2-8b-arch",
    ),
    "mamba2-hybrid-8b": ModelSpec(
        key="mamba2-hybrid-8b",
        repo="nvidia/mamba2-hybrid-8b-3t-4k",
        revision="35e8852e2240b350ac2fe2a3b8aa341b5930018e",
        family="mamba2-hybrid",
        format="megatron",
        arch_pair="mamba2-8b-arch",
    ),
    # Phase 8 descriptive system comparison. Three complete 7-8B production
    # systems that differ from each other on many axes at once - architecture,
    # pretraining corpus, token count, tokenizer, alignment status, depth,
    # and positional encoding - so the sweep is not a matched control: it is
    # a description of how full-system curves look side by side. Nemotron-H
    # is a hybrid Mamba-2 + attention model; Llama-3.1 and Qwen2.5 are dense
    # attention transformers with RoPE. All three run through the standard
    # transformers backend; their execution paths are resolved by family in
    # ``run.py``.
    "nemotron-h-8b": ModelSpec(
        key="nemotron-h-8b",
        repo="nvidia/Nemotron-H-8B-Base-8K",
        revision="94ea861e008c2dfced3e8e1302094024077aa04e",
        family="nemotron-h",
        phase8_system="phase8-systems",
    ),
    "llama-3.1-8b": ModelSpec(
        key="llama-3.1-8b",
        repo="meta-llama/Llama-3.1-8B",
        revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        family="llama",
        phase8_system="phase8-systems",
    ),
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        repo="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796",
        family="qwen2",
        phase8_system="phase8-systems",
    ),
}


SCALE_PAIRS: tuple[str, ...] = ("130m-160m", "370m-410m", "790m-1b", "1.4b-1.4b", "2.8b-2.8b")

# Phase 5 data-control pair: the Pile and SlimPajama models that share the
# 2.8B Mamba architecture. Ordered (Pile, SlimPajama) so contrasts read as the
# Pile-minus-SlimPajama corpus effect.
DATA_PAIR: tuple[str, str] = ("mamba-2.8b", "mamba-2.8b-slimpj")

# Phase 3 released-checkpoint pair: the hybrid and pure Mamba-2 8B models share
# training data, scale, tokenizer, depth, and positional encoding, but differ in
# both attention and MLP composition. Ordered (hybrid, pure) so contrasts read
# as hybrid minus pure. The constant name is retained for artifact compatibility.
ARCH_PAIR: tuple[str, str] = ("mamba2-hybrid-8b", "mamba2-8b")

# Phase 8 descriptive system comparison group. Ordered so figures read
# consistently across summary and plots.
PHASE8_SYSTEMS: tuple[str, ...] = ("nemotron-h-8b", "llama-3.1-8b", "qwen2.5-7b")


def spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model key {key!r}; valid keys: {valid}") from None
