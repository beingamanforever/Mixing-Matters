import json
from unittest.mock import MagicMock, patch

import pytest

from mixing_matters.run import run_certify_negative, run_sweep, run_tracer

TRACER_EXPECTED_KEYS = {
    "run_id",
    "question_id",
    "source_index",
    "condition",
    "gold_position",
    "prompt",
    "model_response",
    "correct_answer",
    "answers",
    "score",
    "score_normalized_em",
    "score_first_line",
    "floor_accuracy",
    "ceiling_accuracy",
    "prompt_token_count",
    "generated_token_count",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "random_seed",
    "manual_seed",
    "model_name",
    "model_revision",
    "software_versions",
    "data_revision",
    "data_sha256",
    "positive_control_sha256",
}


def _tracer_generator_metadata():
    return {
        "model": "EleutherAI/pythia-2.8b",
        "model_revision": "rev",
        "python": "py",
        "torch": "pt",
        "transformers": "tr",
        "cuda": "cu",
        "driver": "550.54",
        "gpu": "mock-gpu",
        "attention_implementation": "eager",
        "dtype": "torch.bfloat16",
    }


SWEEP_EXPECTED_KEYS = TRACER_EXPECTED_KEYS | {"prompt_token_span", "prompt_variant", "gold_padded_tokens"}


def _sweep_generator_metadata():
    return {
        "model": "state-spaces/mamba-2.8b-hf",
        "model_revision": "rev",
        "model_key": "mamba-2.8b",
        "family": "mamba",
        "execution_path": "cuda_kernels",
        "compute_capability": "8.6",
        "mamba_ssm": "2.2.0",
        "causal_conv1d": "1.4.0",
        "python": "py",
        "torch": "pt",
        "transformers": "tr",
        "cuda": "cu",
        "driver": "550.54",
        "gpu": "mock-gpu",
        "attention_implementation": None,
        "dtype": "torch.bfloat16",
    }


def _run_mocked_sweep(tmp_path, row, side_effect, questions=2):
    data_path = tmp_path / "data.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [row for _ in range(200)]

    with (
        patch("mixing_matters.run.file_sha256", return_value="hash_match"),
        patch("mixing_matters.run.SHA256", "hash_match"),
        patch("mixing_matters.run.UPSTREAM_COMMIT", "commit"),
        patch("mixing_matters.run.read_rows", return_value=rows),
        patch("mixing_matters.run.Generator") as generator_class,
    ):
        generator = MagicMock()
        generator.metadata = _sweep_generator_metadata()
        generator.side_effect = side_effect
        generator_class.return_value = generator

        run_sweep(data_path, out_path, "mamba-2.8b", "rev", questions=questions)

    return [json.loads(line) for line in out_path.read_text().strip().split("\n")]


def _run_mocked_tracer(tmp_path, row, side_effect):
    data_path = tmp_path / "data.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [row for _ in range(200)]

    with (
        patch("mixing_matters.run.file_sha256", return_value="hash_match"),
        patch("mixing_matters.run.SHA256", "hash_match"),
        patch("mixing_matters.run.UPSTREAM_COMMIT", "commit"),
        patch("mixing_matters.run.read_rows", return_value=rows),
        patch("mixing_matters.positive_control.validate_control", return_value="control_hash"),
        patch("mixing_matters.run.Generator") as generator_class,
    ):
        generator = MagicMock()
        generator.metadata = _tracer_generator_metadata()
        generator.side_effect = side_effect
        generator_class.return_value = generator

        run_tracer(data_path, out_path, "rev", tmp_path / "control.jsonl")

    return [json.loads(line) for line in out_path.read_text().strip().split("\n")]


def test_record_schema_and_invariance(tmp_path, row):
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w") as stream:
        # just 2 rows
        stream.write(json.dumps(row) + "\n")
        stream.write(json.dumps(row) + "\n")

    out_path = tmp_path / "out.jsonl"

    # We need to patch file_sha256, read_rows, and Generator
    with (
        patch("mixing_matters.run.file_sha256", return_value="hash_match"),
        patch("mixing_matters.run.SHA256", "hash_match"),
        patch("mixing_matters.run.UPSTREAM_COMMIT", "commit"),
        patch("mixing_matters.run.read_rows", return_value=[row, row]),
        patch("mixing_matters.run.Generator") as generator_class,
    ):
        generator = MagicMock()
        generator.metadata = {
            "model": "EleutherAI/pythia-2.8b",
            "model_revision": "rev",
            "python": "py",
            "torch": "pt",
            "transformers": "tr",
            "cuda": "cu",
            "attention_implementation": "eager",
        }

        # return generation, prompt_tokens, generated_tokens
        # we must return a constant prompt_tokens per prompt!
        def side_effect(prompt):
            return ("ans", len(prompt), 10)

        generator.side_effect = side_effect
        generator_class.return_value = generator

        run_certify_negative(data_path, out_path, "rev", None, n=2)

    # Now read the output and check schema
    records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]

    assert len(records) == 20  # 2 questions * 10 positions

    expected_keys = {
        "run_id",
        "condition",
        "gold_present",
        "fake_source_index",
        "question_id",
        "gold_position",
        "permutation_id",
        "permutation_seed",
        "ceiling_accuracy",
        "floor_accuracy",
        "score",
        "prompt",
        "model_response",
        "correct_answer",
        "prompt_token_count",
        "generated_token_count",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "random_seed",
        "manual_seed",
        "model_name",
        "model_revision",
        "software_versions",
        "data_revision",
        "data_sha256",
        "positive_control_sha256",
    }

    ceilings = {}
    floors = {}

    for record in records:
        assert set(record.keys()) == expected_keys

        qid = record["question_id"]
        if qid not in ceilings:
            ceilings[qid] = record["ceiling_accuracy"]
            floors[qid] = record["floor_accuracy"]
        else:
            assert ceilings[qid] == record["ceiling_accuracy"]
            assert floors[qid] == record["floor_accuracy"]


def test_failures_sidecar_blocks_rerun(tmp_path, row):
    (tmp_path / "out.failures.jsonl").write_text("")
    with (
        patch("mixing_matters.run.read_rows", return_value=[row]),
        pytest.raises(FileExistsError),
    ):
        run_certify_negative(tmp_path / "data.jsonl", tmp_path / "out.jsonl", "rev", None, n=1)


def test_tracer_record_schema_and_conditions(tmp_path, row):
    def side_effect(prompt):
        return ("Gold", len(prompt), 5)

    records = _run_mocked_tracer(tmp_path, row, side_effect)
    assert len(records) == 800

    for record in records:
        assert set(record.keys()) == TRACER_EXPECTED_KEYS
        assert record["answers"] == row["answers"]
        assert record["correct_answer"] == row["answers"][0]

    assert len({record["run_id"] for record in records}) == 1

    gold_positions_by_condition: dict[str, set] = {}
    for record in records:
        gold_positions_by_condition.setdefault(record["condition"], set()).add(
            record["gold_position"]
        )
    assert gold_positions_by_condition["closed_book"] == {None}
    assert gold_positions_by_condition["oracle"] == {None}
    assert gold_positions_by_condition["gold_first"] == {0}
    assert gold_positions_by_condition["gold_middle"] == {4}


def test_tracer_rejects_reordered_plan_before_generating(tmp_path, row):
    def side_effect(prompt):
        return ("Gold", len(prompt), 5)

    data_path = tmp_path / "data.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [row for _ in range(200)]

    reordered_work = [
        (index, row, condition)
        for index in range(200)
        for condition in ("oracle", "closed_book", "gold_first", "gold_middle")
    ]

    with (
        patch("mixing_matters.run.file_sha256", return_value="hash_match"),
        patch("mixing_matters.run.SHA256", "hash_match"),
        patch("mixing_matters.run.UPSTREAM_COMMIT", "commit"),
        patch("mixing_matters.run.read_rows", return_value=rows),
        patch("mixing_matters.positive_control.validate_control", return_value="control_hash"),
        patch("mixing_matters.run.plan", return_value=reordered_work),
        patch("mixing_matters.run.Generator") as generator_class,
    ):
        generator = MagicMock()
        generator.metadata = _tracer_generator_metadata()
        generator.side_effect = side_effect
        generator_class.return_value = generator

        with pytest.raises(ValueError, match="unexpected condition order"):
            run_tracer(data_path, out_path, "rev", tmp_path / "control.jsonl")

    assert generator.call_count == 0


def test_tracer_floor_and_ceiling_attach_to_every_condition(tmp_path, row):
    def side_effect(prompt):
        if "Document [" in prompt:
            return ("Gold", len(prompt), 5)
        return ("definitely wrong", len(prompt), 5)

    records = _run_mocked_tracer(tmp_path, row, side_effect)

    by_question: dict[str, list[dict]] = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    assert len(by_question) == 200
    for question_records in by_question.values():
        assert len(question_records) == 4
        assert {record["floor_accuracy"] for record in question_records} == {0.0}
        assert {record["ceiling_accuracy"] for record in question_records} == {1.0}


def test_tracer_scoring_failure_nulls_scores_and_logs_sidecar(tmp_path, row):
    def side_effect(prompt):
        return ("Gold", len(prompt), 5)

    with patch("mixing_matters.run.score_variants", side_effect=RuntimeError("boom")):
        records = _run_mocked_tracer(tmp_path, row, side_effect)

    assert len(records) == 800
    for record in records:
        assert record["score"] is None
        assert record["score_normalized_em"] is None
        assert record["score_first_line"] is None
        assert record["prompt"]
        assert record["model_response"] == "Gold"

    failure_lines = (tmp_path / "out.failures.jsonl").read_text().strip().split("\n")
    assert len(failure_lines) == 800
    conditions_seen = set()
    for line in failure_lines:
        failure = json.loads(line)
        assert failure["error"] == "boom"
        assert isinstance(failure["source_index"], int)
        conditions_seen.add(failure["condition"])
    assert conditions_seen == {"closed_book", "oracle", "gold_first", "gold_middle"}


def test_sweep_record_schema_and_conditions(tmp_path, row):
    def side_effect(prompt):
        return ("Gold", len(prompt), 5)

    records = _run_mocked_sweep(tmp_path, row, side_effect, questions=2)
    assert len(records) == 24  # 2 questions * 12 items

    for record in records:
        assert set(record.keys()) == SWEEP_EXPECTED_KEYS
        assert record["condition"] in {"closed_book", "oracle", "gold"}
        assert record["software_versions"]["model_key"] == "mamba-2.8b"
        assert record["software_versions"]["family"] == "mamba"
        assert record["software_versions"]["execution_path"] == "cuda_kernels"

    gold_positions_by_condition: dict[str, set] = {}
    for record in records:
        gold_positions_by_condition.setdefault(record["condition"], set()).add(
            record["gold_position"]
        )
    assert gold_positions_by_condition["closed_book"] == {None}
    assert gold_positions_by_condition["oracle"] == {None}
    assert gold_positions_by_condition["gold"] == set(range(10))

    for record in records:
        if record["condition"] == "gold":
            assert isinstance(record["prompt_token_span"], int)
        else:
            assert record["prompt_token_span"] is None


def test_sweep_floor_and_ceiling_attach_to_every_condition(tmp_path, row):
    def side_effect(prompt):
        if "Document [" in prompt:
            return ("Gold", len(prompt), 5)
        return ("definitely wrong", len(prompt), 5)

    records = _run_mocked_sweep(tmp_path, row, side_effect, questions=2)

    by_question: dict[str, list[dict]] = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    assert len(by_question) == 2
    for question_records in by_question.values():
        assert len(question_records) == 12
        assert {record["floor_accuracy"] for record in question_records} == {0.0}
        assert {record["ceiling_accuracy"] for record in question_records} == {1.0}


def test_sweep_allows_a_one_token_span_across_gold_positions(tmp_path, row):
    # closed_book, oracle, then ten gold generations; one position drifts by
    # a single token from a byte-pair merge at the document boundary.
    token_counts = [100, 100] + [500] * 5 + [501] + [500] * 4
    calls = iter(token_counts)

    def side_effect(prompt):
        return ("Gold", next(calls), 5)

    records = _run_mocked_sweep(tmp_path, row, side_effect, questions=1)
    gold_records = [record for record in records if record["condition"] == "gold"]
    assert len(gold_records) == 10
    assert {record["prompt_token_span"] for record in gold_records} == {1}


def test_sweep_rejects_a_three_token_span_across_gold_positions(tmp_path, row):
    token_counts = [100, 100] + [500] * 5 + [503] + [500] * 4
    calls = iter(token_counts)

    def side_effect(prompt):
        return ("Gold", next(calls), 5)

    with pytest.raises(ValueError, match="span too large"):
        _run_mocked_sweep(tmp_path, row, side_effect, questions=1)
