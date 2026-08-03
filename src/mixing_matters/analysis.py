import math
from collections import Counter

EXPECTED_COUNTS = {"gold_first": 200, "gold_middle": 200, "closed_book": 50, "oracle": 50}


def validate_phase1(records: list[dict]) -> None:
    counts = Counter(record["condition"] for record in records)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"incomplete Phase 1 results: {dict(counts)}")
    ids = {
        condition: {record["question_id"] for record in records if record["condition"] == condition}
        for condition in EXPECTED_COUNTS
    }
    if ids["gold_first"] != ids["gold_middle"]:
        raise ValueError("gold-first and gold-middle IDs differ")
    if ids["closed_book"] != ids["oracle"] or not ids["oracle"] <= ids["gold_first"]:
        raise ValueError("anchor IDs must be the same subset of position IDs")
    invariant_fields = (
        "model",
        "model_revision",
        "data_revision",
        "data_sha256",
        "positive_control_sha256",
        "seed",
        "torch",
        "transformers",
        "cuda",
        "gpu",
        "attention_implementation",
    )
    for field in invariant_fields:
        values = {record.get(field) for record in records}
        if None in values or len(values) != 1:
            raise ValueError(f"mixed or missing {field}: {values}")
    source_ids = {
        condition: {
            record["source_index"] for record in records if record["condition"] == condition
        }
        for condition in EXPECTED_COUNTS
    }
    if source_ids["gold_first"] != source_ids["gold_middle"]:
        raise ValueError("position source indices differ")
    if (
        source_ids["closed_book"] != source_ids["oracle"]
        or not source_ids["oracle"] <= source_ids["gold_first"]
    ):
        raise ValueError("anchor source indices must be the same subset of position indices")


def summarize(records: list[dict]) -> dict:
    keyed: dict[tuple[str, str], float] = {}
    for record in records:
        key = record["question_id"], record["condition"]
        if key in keyed:
            raise ValueError(f"duplicate result: {key}")
        keyed[key] = float(record["score"])

    accuracies = {}
    for condition in {condition for _, condition in keyed}:
        scores = [score for (_, name), score in keyed.items() if name == condition]
        accuracies[condition] = sum(scores) / len(scores)

    first_ids = {qid for qid, condition in keyed if condition == "gold_first"}
    middle_ids = {qid for qid, condition in keyed if condition == "gold_middle"}
    if first_ids != middle_ids or not first_ids:
        raise ValueError("gold-first and gold-middle results must form complete pairs")
    differences = [keyed[qid, "gold_first"] - keyed[qid, "gold_middle"] for qid in first_ids]
    counts = Counter(str(int(value)) for value in differences)
    discordance = sum(value != 0 for value in differences) / len(differences)
    return {
        "accuracy": accuracies,
        "paired_count": len(differences),
        "difference_counts": {key: counts.get(key, 0) for key in ("-1", "0", "1")},
        "mean_first_minus_middle": sum(differences) / len(differences),
        "discordance": discordance,
        "planned_interaction_n": planning_sample_size(discordance),
    }


def planning_sample_size(discordance: float) -> int:
    """Apply the preregistered affine calibration in the research plan.

    This reproduces its 0.20 -> 1156, 0.25 -> 1460, and 0.35 -> 2068 table.
    It is a planning calibration for a five-point architecture interaction, not
    an independently identified power calculation from one model.
    """
    if not 0 <= discordance <= 1:
        raise ValueError("discordance must be between zero and one")
    return max(1, math.ceil(6080 * discordance - 60))
