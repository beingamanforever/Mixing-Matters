import math
import random
from collections import Counter

EXPECTED_COUNTS = {"gold_first": 200, "gold_middle": 200, "closed_book": 50, "oracle": 50}
BOOTSTRAP_SEED = 240521
FLOOR_TOLERANCE = 0.05
ORDER_TOLERANCE = 0.10


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


def bootstrap_paired_edges(
    scores_by_question: dict[str, dict[int, float]], rng: random.Random, n_resamples: int = 10000
) -> tuple[tuple[float, float], tuple[float, float]]:
    primacy = []
    recency = []
    questions = list(scores_by_question.keys())
    size = len(questions)

    for _ in range(n_resamples):
        sampled = rng.choices(questions, k=size)

        def mean_accuracy(positions):
            scores = [
                scores_by_question[question][position]
                for question in sampled
                for position in positions
            ]
            return sum(scores) / len(scores) if scores else 0.0

        primacy.append(mean_accuracy([0, 1]) - mean_accuracy([4, 5]))
        recency.append(mean_accuracy([8, 9]) - mean_accuracy([4, 5]))

    primacy.sort()
    recency.sort()

    ci_primacy = (primacy[int(0.025 * n_resamples)], primacy[int(0.975 * n_resamples)])
    ci_recency = (recency[int(0.025 * n_resamples)], recency[int(0.975 * n_resamples)])
    return ci_primacy, ci_recency


def validate_negative(records: list[dict]) -> None:
    lengths_by_question: dict[str, set[int]] = {}
    for record in records:
        lengths = lengths_by_question.setdefault(record["question_id"], set())
        lengths.add(record["prompt_token_count"])
    for question, lengths in lengths_by_question.items():
        if len(lengths) > 1:
            raise ValueError(
                f"per-question length non-invariance detected for {question}: {lengths}"
            )

    scored_by_question: dict[str, dict[int, dict]] = {}
    for record in records:
        if record["score"] is None:
            continue
        positions = scored_by_question.setdefault(record["question_id"], {})
        if record["gold_position"] in positions:
            raise ValueError(
                f"duplicate result: {record['question_id']} at {record['gold_position']}"
            )
        positions[record["gold_position"]] = record
    if not scored_by_question:
        raise ValueError("no valid records found")
    unscored = set(lengths_by_question) - set(scored_by_question)
    if unscored:
        raise ValueError(f"{len(unscored)} questions have no scored position: {sorted(unscored)}")

    scores_by_question: dict[str, dict[int, float]] = {}
    floors = []
    for question, positions in scored_by_question.items():
        if len(positions) != 10:
            raise ValueError(f"question {question} lacks ten scored positions: {len(positions)}")
        floor_values = {record["floor_accuracy"] for record in positions.values()}
        if len(floor_values) != 1:
            raise ValueError(f"question {question} carries mixed floor accuracies: {floor_values}")
        floors.append(floor_values.pop())
        scores_by_question[question] = {
            position: record["score"] for position, record in positions.items()
        }

    # Single questions score 0 or 1, so only the pooled accuracy can sit near the
    # closed-book floor. A per-question gate would reject any correct guess.
    scored_count = 10 * len(scores_by_question)
    accuracy = sum(sum(scores.values()) for scores in scores_by_question.values()) / scored_count
    floor = sum(floors) / len(floors)
    if abs(accuracy - floor) > FLOOR_TOLERANCE:
        raise ValueError(
            f"negative control accuracy differs from floor by more than {FLOOR_TOLERANCE}: "
            f"{accuracy} vs {floor}"
        )

    ci_primacy, ci_recency = bootstrap_paired_edges(
        scores_by_question, random.Random(BOOTSTRAP_SEED)
    )
    if not ci_primacy[0] <= 0 <= ci_primacy[1]:
        raise ValueError(f"flatness CI for primacy does not contain 0: {ci_primacy}")
    if not ci_recency[0] <= 0 <= ci_recency[1]:
        raise ValueError(f"flatness CI for recency does not contain 0: {ci_recency}")


def validate_order(records: list[dict]) -> None:
    lengths_by_group: dict[tuple[str, int], set[int]] = {}
    scores_by_permutation: dict[int, dict[tuple[str, int], float]] = {}
    for record in records:
        group = (record["question_id"], record["gold_position"])
        lengths_by_group.setdefault(group, set()).add(record["prompt_token_count"])
        if record["score"] is None:
            continue
        scores = scores_by_permutation.setdefault(record["permutation_id"], {})
        if group in scores:
            raise ValueError(f"duplicate result: {group} permutation {record['permutation_id']}")
        scores[group] = record["score"]
    if not scores_by_permutation:
        raise ValueError("no valid records found")
    if len(scores_by_permutation) < 2:
        raise ValueError("no permutations found to compare")

    for group, lengths in lengths_by_group.items():
        if len(lengths) > 1:
            raise ValueError(f"distractor order changed prompt length for {group}: {lengths}")

    scored_groups = {group for scores in scores_by_permutation.values() for group in scores}
    unscored = set(lengths_by_group) - scored_groups
    if unscored:
        raise ValueError(f"{len(unscored)} groups have no scored permutation: {sorted(unscored)}")

    groups = [set(scores) for scores in scores_by_permutation.values()]
    if any(covered != groups[0] for covered in groups):
        raise ValueError("permutations do not cover the same question and position pairs")

    # Accuracy is the group-level quantity the position claim rests on. Item-level
    # flips between two orderings are expected even when accuracy is unchanged.
    accuracies = {
        permutation: sum(scores.values()) / len(scores)
        for permutation, scores in scores_by_permutation.items()
    }
    spread = max(accuracies.values()) - min(accuracies.values())
    if spread > ORDER_TOLERANCE:
        raise ValueError(
            f"accuracy across distractor permutations spans more than {ORDER_TOLERANCE}: "
            f"{accuracies}"
        )
