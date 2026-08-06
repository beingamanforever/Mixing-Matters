from collections import Counter

import pytest

from mixing_matters.audit import FAILURE_CATEGORIES, audit_sample, write_audit_sample

CONDITIONS = ("closed_book", "gold_first", "gold_middle", "oracle")


def _records(n_questions: int = 60, n_anchors: int = 20) -> list[dict]:
    records = []
    for qid in range(n_questions):
        for condition in CONDITIONS:
            if condition in ("closed_book", "oracle") and qid >= n_anchors:
                continue
            if condition == "closed_book":
                prompt = f"Question: What is item {qid}?\nAnswer:"
            else:
                prompt = (
                    f"Document [1](Title: d{qid}) some text about item {qid}\n\n"
                    f"Question: What is item {qid}?\nAnswer:"
                )
            records.append(
                {
                    "run_id": "run-1",
                    "question_id": str(qid),
                    "condition": condition,
                    "gold_position": {"gold_first": 0, "gold_middle": 4}.get(condition),
                    "prompt": prompt,
                    "model_response": f"response-{qid}",
                    "correct_answer": f"gold-{qid}",
                    "score": 1.0,
                    "score_normalized_em": 1.0,
                    "score_first_line": 1.0,
                    "model_name": "test-model",
                }
            )
    return records


def test_audit_sample_is_deterministic():
    records = _records()
    first = audit_sample(records)
    second = audit_sample(records)
    assert first == second


def test_audit_sample_blinds_condition_and_model():
    records = _records()
    blinded_rows, key_rows = audit_sample(records)
    assert len(blinded_rows) == 50
    for row in blinded_rows:
        assert set(row) == {
            "audit_id",
            "question",
            "documents",
            "model_response",
            "correct_answer",
        }
        assert row["question"] == row["question"]  # recovered without raising
    for row in key_rows:
        assert set(row) == {
            "audit_id",
            "run_id",
            "question_id",
            "condition",
            "gold_position",
            "score",
            "score_normalized_em",
            "score_first_line",
            "model_name",
        }
    blinded_ids = {row["audit_id"] for row in blinded_rows}
    key_ids = {row["audit_id"] for row in key_rows}
    assert blinded_ids == key_ids


def test_audit_sample_spreads_evenly_across_conditions():
    records = _records()
    _, key_rows = audit_sample(records, size=50)
    counts = Counter(row["condition"] for row in key_rows)
    assert set(counts) == set(CONDITIONS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_audit_sample_redistributes_when_a_condition_is_short():
    records = [
        record
        for record in _records()
        if not (record["condition"] == "oracle" and int(record["question_id"]) >= 5)
    ]
    _, key_rows = audit_sample(records, size=40)
    counts = Counter(row["condition"] for row in key_rows)
    assert counts["oracle"] == 5
    assert sum(counts.values()) == 40


def test_audit_sample_fails_loudly_without_question_marker():
    records = [dict(record, prompt="no question marker in this prompt") for record in _records()]
    with pytest.raises(ValueError, match="question"):
        audit_sample(records)


def test_audit_sample_documents_are_blind_to_gold_position():
    same_titles_different_order = [
        {
            "run_id": "run-1",
            "question_id": "0",
            "condition": "gold_first",
            "gold_position": 0,
            "prompt": (
                "Document [1](Title: gold-doc) golden text\n"
                "Document [2](Title: zeta-doc) other text\n\n"
                "Question: q?\nAnswer:"
            ),
            "model_response": "response-a",
            "correct_answer": "gold-a",
            "score": 1.0,
            "score_normalized_em": 1.0,
            "score_first_line": 1.0,
            "model_name": "test-model",
        },
        {
            "run_id": "run-1",
            "question_id": "1",
            "condition": "gold_middle",
            "gold_position": 1,
            "prompt": (
                "Document [1](Title: zeta-doc) other text\n"
                "Document [2](Title: gold-doc) golden text\n\n"
                "Question: q?\nAnswer:"
            ),
            "model_response": "response-b",
            "correct_answer": "gold-b",
            "score": 1.0,
            "score_normalized_em": 1.0,
            "score_first_line": 1.0,
            "model_name": "test-model",
        },
    ]
    blinded_rows, _ = audit_sample(same_titles_different_order, size=2)
    document_orders = [
        tuple(document["title"] for document in row["documents"]) for row in blinded_rows
    ]
    assert document_orders[0] == document_orders[1] == ("gold-doc", "zeta-doc")


def test_audit_sample_reads_missing_variant_scores_defensively():
    # Certify-control outputs (e.g. certify-negative, certify-order) do not
    # compute score_normalized_em or score_first_line at all.
    records = [
        {key: value for key, value in record.items() if key != "score_normalized_em"}
        for record in _records()
    ]
    records = [
        {key: value for key, value in record.items() if key != "score_first_line"}
        for record in records
    ]
    _, key_rows = audit_sample(records)
    for row in key_rows:
        assert row["score_normalized_em"] is None
        assert row["score_first_line"] is None


def test_write_audit_sample_refuses_to_overwrite(tmp_path):
    records = _records()
    write_audit_sample(records, tmp_path)
    with pytest.raises(FileExistsError):
        write_audit_sample(records, tmp_path)


def test_write_audit_sample_writes_form_with_four_categories(tmp_path):
    records = _records()
    paths = write_audit_sample(records, tmp_path)
    assert {path.name for path in paths} == {
        "audit-sample.jsonl",
        "audit-key.jsonl",
        "audit-form.md",
    }
    form_path = next(path for path in paths if path.name == "audit-form.md")
    text = form_path.read_text()
    for category in FAILURE_CATEGORIES:
        assert category in text
