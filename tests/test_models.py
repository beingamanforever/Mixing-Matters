import dataclasses

import pytest

from mixing_matters.models import DATA_PAIR, MODELS, SCALE_PAIRS, ModelSpec, spec


def test_registry_pins_the_phase2_and_phase4_models():
    assert MODELS["pythia-2.8b"] == ModelSpec(
        key="pythia-2.8b",
        repo="EleutherAI/pythia-2.8b",
        revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
        family="pythia",
        scale_pair="2.8b-2.8b",
        params_millions=2800,
    )
    assert MODELS["mamba2-2.7b"] == ModelSpec(
        key="mamba2-2.7b",
        repo="AntonV/mamba2-2.7b-hf",
        revision="ef542707386fa9ec86bbf8a35ed2952af84bf566",
        family="mamba2",
    )


def test_registry_pins_the_phase5_data_control_models():
    assert MODELS["mamba-2.8b-slimpj"] == ModelSpec(
        key="mamba-2.8b-slimpj",
        repo="state-spaces/mamba-2.8b-slimpj",
        revision="a7bdd41af90ca0cc4ecfbd967e2ec28f1954b915",
        family="mamba",
        params_millions=2800,
        training_corpus="slimpajama",
        data_pair="mamba-2.8b-corpus",
        format="mamba_ssm",
    )


def test_data_pair_shares_an_architecture_and_differs_only_in_corpus():
    pile, slimpj = DATA_PAIR
    assert MODELS[pile].training_corpus == "pile"
    assert MODELS[slimpj].training_corpus == "slimpajama"
    assert MODELS[pile].data_pair == MODELS[slimpj].data_pair == "mamba-2.8b-corpus"
    assert MODELS[pile].family == MODELS[slimpj].family == "mamba"
    assert MODELS[pile].params_millions == MODELS[slimpj].params_millions
    # The Pile checkpoint loads directly; the SlimPajama one is converted first.
    assert MODELS[pile].format == "hf"
    assert MODELS[slimpj].format == "mamba_ssm"


def test_every_scale_pair_has_one_mamba_and_one_pythia():
    for pair in SCALE_PAIRS:
        members = [model for model in MODELS.values() if model.scale_pair == pair]
        families = sorted(model.family for model in members)
        assert families == ["mamba", "pythia"], f"{pair}: {families}"
        assert all(model.params_millions for model in members)


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
