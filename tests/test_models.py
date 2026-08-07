import dataclasses

import pytest

from mixing_matters.models import MODELS, ModelSpec, spec


def test_registry_has_the_three_pinned_models():
    assert set(MODELS).issuperset({"pythia-2.8b", "mamba-2.8b", "mamba2-2.7b"})
    assert MODELS["pythia-2.8b"] == ModelSpec(
        key="pythia-2.8b",
        repo="EleutherAI/pythia-2.8b",
        revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
        family="pythia",
    )
    assert MODELS["mamba-2.8b"] == ModelSpec(
        key="mamba-2.8b",
        repo="state-spaces/mamba-2.8b-hf",
        revision="96c48e0292b63f5346b6d30061af2551f7101e26",
        family="mamba",
    )
    assert MODELS["mamba2-2.7b"] == ModelSpec(
        key="mamba2-2.7b",
        repo="AntonV/mamba2-2.7b-hf",
        revision="ef542707386fa9ec86bbf8a35ed2952af84bf566",
        family="mamba2",
    )


def test_registry_keys_match_their_spec_key_field():
    for key, model_spec in MODELS.items():
        assert model_spec.key == key


def test_spec_looks_up_by_key():
    assert spec("mamba2-2.7b") is MODELS["mamba2-2.7b"]


def test_spec_raises_clear_error_listing_valid_keys():
    with pytest.raises(ValueError, match="mamba-2.8b"):
        spec("mamba-3b")


def test_model_spec_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODELS["pythia-2.8b"].key = "changed"
