import random

import pytest

from mixing_matters.convert import _hf_config, _rename_state_dict
from mixing_matters.models import DATA_PAIR
from mixing_matters.phase2 import interaction
from mixing_matters.phase5 import (
    _relative_size,
    compare_to_architecture,
    corpus_interaction,
    data_control,
)

FLOOR = 0.1
CEILING = 0.9
PILE, SLIMPJ = DATA_PAIR

# state-spaces/mamba-2.8b-slimpj config.json, verbatim.
SLIMPJ_CONFIG = {
    "d_model": 2560,
    "n_layer": 64,
    "vocab_size": 50277,
    "ssm_cfg": {},
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 8,
}


def _uniform_model_records(
    model_key: str,
    question_count: int,
    primacy_score: float,
    center_score: float,
    recency_score: float,
) -> list[dict]:
    positions = {
        0: primacy_score,
        1: primacy_score,
        2: center_score,
        3: center_score,
        4: center_score,
        5: center_score,
        6: center_score,
        7: center_score,
        8: recency_score,
        9: recency_score,
    }
    records = []
    for question in range(question_count):
        question_id = f"q{question}"
        for position, score in positions.items():
            records.append(
                {
                    "model_key": model_key,
                    "question_id": question_id,
                    "condition": "gold",
                    "gold_position": position,
                    "score": score,
                    "floor_accuracy": FLOOR,
                    "ceiling_accuracy": CEILING,
                }
            )
    return records


def _corpus_records(question_count: int = 20) -> list[dict]:
    """Pile has a primacy edge of 0.3; SlimPajama is flat. No recency edge."""
    records = _uniform_model_records(PILE, question_count, 0.6, 0.3, 0.3)
    records += _uniform_model_records(SLIMPJ, question_count, 0.3, 0.3, 0.3)
    return records


# --- conversion helpers -----------------------------------------------------


def test_hf_config_matches_the_published_pile_conversion():
    pytest.importorskip("transformers")
    config = _hf_config(SLIMPJ_CONFIG)
    # Values read from state-spaces/mamba-2.8b-hf config.json: same architecture.
    assert config.hidden_size == 2560
    assert config.num_hidden_layers == 64
    assert config.intermediate_size == 5120
    assert config.time_step_rank == 160
    # 50277 padded up to the next multiple of 8.
    assert config.vocab_size == 50280
    assert config.state_size == 16
    assert config.conv_kernel == 4


def test_rename_state_dict_only_renames_the_embedding():
    original = {
        "backbone.embedding.weight": "E",
        "backbone.layers.0.mixer.in_proj.weight": "W",
        "backbone.norm_f.weight": "N",
    }
    renamed = _rename_state_dict(original)
    assert renamed == {
        "backbone.embeddings.weight": "E",
        "backbone.layers.0.mixer.in_proj.weight": "W",
        "backbone.norm_f.weight": "N",
    }


# --- corpus contrast --------------------------------------------------------


def test_corpus_interaction_recovers_the_pile_minus_slimpajama_primacy_gap():
    result = corpus_interaction(_corpus_records(), n_resamples=50)
    assert result["primacy"]["estimate"] == pytest.approx(0.3)
    assert result["recency"]["estimate"] == pytest.approx(0.0)


def test_data_control_reports_pair_members_and_per_model_edges():
    result = data_control(_corpus_records(), n_resamples=50)
    assert result["pile_model"] == PILE
    assert result["slimpajama_model"] == SLIMPJ
    assert result["primacy_diff"]["estimate"] == pytest.approx(0.3)
    assert result["pile_edges"]["primacy"]["estimate"] == pytest.approx(0.3)
    assert result["slimpajama_edges"]["primacy"]["estimate"] == pytest.approx(0.0)
    assert result["question_count"] == 20


def test_data_control_is_deterministic_across_shuffled_order():
    records = _corpus_records()
    first = data_control(records, n_resamples=50)
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    second = data_control(shuffled, n_resamples=50)
    assert first == second


# --- cross-phase comparison -------------------------------------------------


def _interaction(primacy_estimate: float, ci_half: float) -> dict:
    return {
        "primacy": {
            "estimate": primacy_estimate,
            "ci_low": primacy_estimate - ci_half,
            "ci_high": primacy_estimate + ci_half,
            "p_value": 0.0,
            "p_value_holm": 0.0,
        },
        "recency": {
            "estimate": 0.0,
            "ci_low": -ci_half,
            "ci_high": ci_half,
            "p_value": 1.0,
            "p_value_holm": 1.0,
        },
    }


def test_compare_to_architecture_calls_overlapping_effects_comparable():
    control = data_control(_corpus_records(), n_resamples=50)
    # Architecture primacy effect at 0.3 with a wide interval overlapping the
    # corpus effect at 0.3.
    architecture = _interaction(0.3, 0.2)
    comparison = compare_to_architecture(control, architecture)
    assert comparison["primacy"]["corpus_effect_vs_architecture"] == "comparable"


def test_compare_to_architecture_calls_a_clearly_larger_corpus_effect_larger():
    control = data_control(_corpus_records(), n_resamples=50)
    # Architecture primacy effect near zero with a tight interval; corpus effect
    # at 0.3 sits well outside it.
    architecture = _interaction(0.0, 0.02)
    comparison = compare_to_architecture(control, architecture)
    assert comparison["primacy"]["corpus_effect_vs_architecture"] == "larger"


def test_relative_size_calls_equal_magnitude_opposite_sign_comparable():
    corpus = {"estimate": 0.30, "ci_low": 0.25, "ci_high": 0.35}
    architecture = {"estimate": -0.30, "ci_low": -0.35, "ci_high": -0.25}
    assert _relative_size(corpus, architecture) == "comparable"


def test_relative_size_labels_disjoint_magnitudes():
    small = {"estimate": 0.05, "ci_low": 0.02, "ci_high": 0.08}
    large = {"estimate": -0.40, "ci_low": -0.45, "ci_high": -0.35}
    assert _relative_size(large, small) == "larger"
    assert _relative_size(small, large) == "smaller"


def test_corpus_interaction_matches_phase2_interaction_on_the_data_pair():
    records = _corpus_records()
    assert corpus_interaction(records, n_resamples=50) == interaction(
        records, PILE, SLIMPJ, n_resamples=50
    )
