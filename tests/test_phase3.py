import inspect
import random

import pytest

from mixing_matters.models import ARCH_PAIR
from mixing_matters.phase2 import interaction
from mixing_matters.phase3 import attention_control, attention_interaction

FLOOR = 0.1
CEILING = 0.9
HYBRID, PURE = ARCH_PAIR


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


def _checkpoint_records(question_count: int = 20) -> list[dict]:
    """The hybrid has a primacy edge of 0.3; the pure model is flat. No recency edge."""
    records = _uniform_model_records(HYBRID, question_count, 0.6, 0.3, 0.3)
    records += _uniform_model_records(PURE, question_count, 0.3, 0.3, 0.3)
    return records


# --- checkpoint contrast ----------------------------------------------------


def test_attention_interaction_recovers_the_hybrid_minus_pure_primacy_gap():
    result = attention_interaction(_checkpoint_records(), n_resamples=50)
    assert result["primacy"]["estimate"] == pytest.approx(0.3)
    assert result["recency"]["estimate"] == pytest.approx(0.0)


def test_attention_control_reports_pair_members_and_per_model_edges():
    result = attention_control(_checkpoint_records(), n_resamples=50)
    assert result["hybrid_model"] == HYBRID
    assert result["pure_model"] == PURE
    assert result["pair"] == "mamba2-8b-arch"
    assert result["primacy_diff"]["estimate"] == pytest.approx(0.3)
    assert result["hybrid_edges"]["primacy"]["estimate"] == pytest.approx(0.3)
    assert result["pure_edges"]["primacy"]["estimate"] == pytest.approx(0.0)
    assert result["question_count"] == 20


def test_attention_control_edges_use_only_the_two_phase3_models():
    # A third model that is not in the pair must not enter the per-model edges.
    records = _checkpoint_records()
    records += _uniform_model_records("pythia-2.8b", 20, 0.9, 0.9, 0.9)
    result = attention_control(records, n_resamples=50)
    assert result["hybrid_edges"]["primacy"]["estimate"] == pytest.approx(0.3)
    assert result["pure_edges"]["primacy"]["estimate"] == pytest.approx(0.0)


def test_attention_control_is_deterministic_across_shuffled_order():
    records = _checkpoint_records()
    first = attention_control(records, n_resamples=50)
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    second = attention_control(shuffled, n_resamples=50)
    assert first == second


def test_attention_interaction_matches_phase2_interaction_on_the_arch_pair():
    records = _checkpoint_records()
    assert attention_interaction(records, n_resamples=50) == interaction(
        records, HYBRID, PURE, n_resamples=50
    )


# --- figures ----------------------------------------------------------------


def test_write_phase3_figures_writes_all_outputs(tmp_path):
    from mixing_matters.figures import write_phase3_figures

    paths = write_phase3_figures(_checkpoint_records(), tmp_path, n_resamples=50)
    names = {path.name for path in paths}
    assert names == {
        "position-curves.png",
        "position-edges.png",
        "attention-effect.png",
        "phase3-summary.json",
    }
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_write_phase3_figures_refuses_to_overwrite(tmp_path):
    from mixing_matters.figures import write_phase3_figures

    write_phase3_figures(_checkpoint_records(), tmp_path, n_resamples=50)
    with pytest.raises(FileExistsError):
        write_phase3_figures(_checkpoint_records(), tmp_path, n_resamples=50)


# --- injected-generator seam ------------------------------------------------


class _FakeGenerator:
    """A stand-in backend honoring the Generator interface used by run.py."""

    def __init__(self):
        self.metadata = {
            "seed": 42,
            "model": "nvidia/mamba2-8b-3t-4k",
            "model_key": "mamba2-8b",
            "family": "mamba2",
            "model_revision": "deadbeef",
            "python": "3.12",
            "torch": "x",
            "transformers": "x",
            "cuda": "12.6",
            "driver": "570",
            "gpu": "L40S",
            "attention_implementation": None,
            "dtype": "torch.bfloat16",
            "execution_path": "megatron",
            "compute_capability": "8.9",
            "mamba_ssm": "2.0.3",
            "causal_conv1d": "1.2.2.post1",
        }

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        return "answer", 100, 4


def test_run_sweep_and_kv_control_accept_a_generator_seam():
    from mixing_matters.run import run_kv_control, run_sweep

    assert inspect.signature(run_sweep).parameters["generator"].default is None
    assert inspect.signature(run_kv_control).parameters["generator"].default is None


def test_run_kv_control_uses_injected_generator_and_never_builds_a_real_one(tmp_path, monkeypatch):
    from mixing_matters import run
    from mixing_matters.io import read_jsonl
    from mixing_matters.models import MODELS
    from mixing_matters.positive_control import EXAMPLES, POSITIONS

    def _boom(*args, **kwargs):
        raise AssertionError("run.Generator must not be constructed when one is injected")

    monkeypatch.setattr(run, "Generator", _boom)

    output = tmp_path / "kv.jsonl"
    run.run_kv_control(MODELS["mamba2-8b"], output, "deadbeef", generator=_FakeGenerator())

    records = read_jsonl(output)
    assert len(records) == EXAMPLES * len(POSITIONS)
    record = records[0]
    for field in ("control_id", "condition", "prompt", "generation", "gold", "score", "model_key"):
        assert field in record
    assert record["model_key"] == "mamba2-8b"
