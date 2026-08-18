import csv
import gzip
import json
from pathlib import Path

import pytest

from mixing_matters.models import MODELS
from mixing_matters.release import (
    GENERATION_FIELDS,
    GENERATION_RUNS,
    SINK_FIELDS,
    SINK_RUNS,
    build_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    output = tmp_path_factory.mktemp("dataset")
    return output, build_dataset(ROOT, output)


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def test_every_declared_run_points_at_a_committed_sweep():
    for run in GENERATION_RUNS + SINK_RUNS:
        assert (ROOT / run.path).is_file(), run.path
        assert run.model_key in MODELS


def test_run_keys_are_unique():
    keys = [run.key for run in GENERATION_RUNS + SINK_RUNS]
    assert len(keys) == len(set(keys))


def test_builds_expected_record_counts(dataset):
    _, counts = dataset
    # 800 questions x 12 conditions across 22 full QA sweeps, plus the
    # 200-question Phase 7 variants, the Phase 1 tracer, and 21 key-value
    # control sweeps of 500 records each.
    assert counts["generations"] == 229_700
    assert counts["attention_sink"] == 280_000
    assert counts["runs"] == len(GENERATION_RUNS) + len(SINK_RUNS)
    assert counts["positions"] == 522


def test_generation_rows_carry_the_full_schema(dataset):
    output, _ = dataset
    rows = _rows(output / "generations.jsonl.gz")
    assert all(tuple(row) == GENERATION_FIELDS for row in rows[:200])

    gold = [row for row in rows if row["condition"] == "gold"]
    assert {row["gold_position"] for row in gold} == set(range(10))
    assert {row["mixer"] for row in rows} == {"attention", "state-space", "hybrid"}
    assert {row["task"] for row in rows} == {"multidoc_qa", "kv_retrieval"}
    assert all(row["score"] is not None for row in rows)
    assert all("prompt" not in row for row in rows)


def test_prompt_variant_names_are_normalized(dataset):
    output, _ = dataset
    rows = _rows(output / "generations.jsonl.gz")
    assert {row["prompt_variant"] for row in rows} == {
        "liu_baseline",
        "bookend",
        "question_first",
        "gold_padded",
    }
    assert {row["prompt_template"] for row in rows} == {
        "liu_baseline",
        "concise",
        "instructional",
    }


def test_sink_rows_cover_every_layer(dataset):
    output, _ = dataset
    rows = _rows(output / "attention_sink.jsonl.gz")
    assert all(tuple(row) == SINK_FIELDS for row in rows[:200])
    assert all(isinstance(row["sink_mass"], float) for row in rows[:200])

    layers = {row["layer"] for row in rows if row["model_key"] == "pythia-2.8b"}
    # Pythia 2.8B has 32 layers and the scan records token-0 mass in each.
    assert layers == set(range(32))


def test_position_accuracy_matches_the_committed_phase2_summary(dataset):
    output, _ = dataset
    summary = json.loads((ROOT / "artifacts/phase2/report/phase2-summary.json").read_text())
    with (output / "position_accuracy.csv").open() as stream:
        rows = {
            (row["run_key"], int(row["gold_position"])): float(row["accuracy"])
            for row in csv.DictReader(stream)
        }
    for model, curve in summary["position_curve"].items():
        for position, point in curve["positions"].items():
            assert rows[(f"phase2/{model}", int(position))] == pytest.approx(point["accuracy"])


def test_run_table_reports_pinned_checkpoints(dataset):
    output, _ = dataset
    with (output / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(GENERATION_RUNS) + len(SINK_RUNS)
    for row in rows:
        spec = MODELS[row["model_key"]]
        assert row["model_repo"] == spec.repo
        assert row["model_revision"] == spec.revision
        assert int(row["record_count"]) > 0


def test_rebuilding_is_deterministic(dataset, tmp_path):
    first, _ = dataset
    second = tmp_path / "rebuild"
    build_dataset(ROOT, second)
    for name in (
        "generations.jsonl.gz",
        "attention_sink.jsonl.gz",
        "runs.csv",
        "position_accuracy.csv",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
