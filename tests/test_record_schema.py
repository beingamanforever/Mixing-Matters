import json
from unittest.mock import MagicMock, patch

from mixing_matters.run import run_certify_negative


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
    ):
        with patch("mixing_matters.run.Generator") as generator_class:
            generator = MagicMock()
            generator.metadata = {
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
