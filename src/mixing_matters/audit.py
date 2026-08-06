import random
import re
from pathlib import Path

from .io import write_jsonl

FAILURE_CATEGORIES = ("formatting", "extraction", "hallucination", "truncation")

FORM_TEXT = (
    "# Audit form\n"
    "\n"
    "Record at most one failure category per item, or none if the response is correct.\n"
    "\n"
    "- formatting: the answer is correct but wrapped in extra text or punctuation the grader could not parse.\n"
    "- extraction: the model copied the wrong span or fact from the supplied context.\n"
    "- hallucination: the model invented an answer unsupported by the context or closed-book knowledge.\n"
    "- truncation: the response was cut off before a complete answer was produced.\n"
)

_DOCUMENT_LINE = re.compile(r"^Document \[\d+\]\(Title: (?P<title>.*?)\) (?P<text>.*)$")


def _question_from_prompt(prompt: str) -> str:
    marker = "Question: "
    index = prompt.rfind(marker)
    if index == -1:
        raise ValueError("could not recover a question from the prompt")
    question = prompt[index + len(marker) :]
    end = question.find("\nAnswer:")
    if end != -1:
        question = question[:end]
    question = question.strip()
    if not question:
        raise ValueError("recovered question text is empty")
    return question


def _documents_from_prompt(prompt: str) -> list[dict]:
    """Recover the retrieved documents from a prompt, sorted by title.

    Sorting by title yields a canonical order that cannot reveal which
    document held the gold slot, regardless of how the documents were
    originally arranged in the prompt.
    """
    documents = []
    for line in prompt.splitlines():
        match = _DOCUMENT_LINE.match(line)
        if match:
            documents.append({"title": match["title"], "text": match["text"]})
    documents.sort(key=lambda document: document["title"])
    return documents


def _allocate_quota(available: dict[str, int], size: int) -> dict[str, int]:
    conditions = sorted(available)
    quota = dict.fromkeys(conditions, 0)
    remaining = min(size, sum(available.values()))
    active = [condition for condition in conditions if available[condition] > quota[condition]]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        allocated = 0
        for index, condition in enumerate(active):
            want = share + (1 if index < extra else 0)
            room = available[condition] - quota[condition]
            take = min(want, room)
            quota[condition] += take
            allocated += take
        remaining -= allocated
        active = [condition for condition in conditions if available[condition] > quota[condition]]
        if allocated == 0:
            break
    return quota


def audit_sample(
    records: list[dict], size: int = 50, seed: int = 240521
) -> tuple[list[dict], list[dict]]:
    """Sample records for a blinded manual audit, spread evenly across conditions.

    Returns (blinded_rows, key_rows). Blinded rows carry the question, the
    retrieved documents (in title order, so the order cannot reveal the gold
    slot), the model response, and the correct answer; key rows carry the
    identifying and scoring fields needed to unblind an item after the audit
    is complete. The scoring-variant fields are read defensively, since some
    record sources (for example certify-control outputs) do not compute them.
    """
    by_condition: dict[str, list[dict]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    if not by_condition:
        raise ValueError("no records to sample from")

    for bucket in by_condition.values():
        bucket.sort(key=lambda record: record["question_id"])

    available = {condition: len(bucket) for condition, bucket in by_condition.items()}
    quota = _allocate_quota(available, size)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for condition in sorted(by_condition):
        chosen.extend(rng.sample(by_condition[condition], quota[condition]))
    rng.shuffle(chosen)

    blinded_rows = []
    key_rows = []
    for index, record in enumerate(chosen, start=1):
        audit_id = f"audit-{index:04d}"
        blinded_rows.append(
            {
                "audit_id": audit_id,
                "question": _question_from_prompt(record["prompt"]),
                "documents": _documents_from_prompt(record["prompt"]),
                "model_response": record["model_response"],
                "correct_answer": record["correct_answer"],
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "run_id": record["run_id"],
                "question_id": record["question_id"],
                "condition": record["condition"],
                "gold_position": record["gold_position"],
                "score": record["score"],
                "score_normalized_em": record.get("score_normalized_em"),
                "score_first_line": record.get("score_first_line"),
                "model_name": record["model_name"],
            }
        )
    return blinded_rows, key_rows


def write_audit_sample(records: list[dict], directory: Path) -> list[Path]:
    sample_path = directory / "audit-sample.jsonl"
    key_path = directory / "audit-key.jsonl"
    form_path = directory / "audit-form.md"
    for path in (sample_path, key_path, form_path):
        if path.exists():
            raise FileExistsError(path)

    blinded_rows, key_rows = audit_sample(records)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(sample_path, blinded_rows)
    write_jsonl(key_path, key_rows)
    form_path.write_text(FORM_TEXT)
    return [sample_path, key_path, form_path]
