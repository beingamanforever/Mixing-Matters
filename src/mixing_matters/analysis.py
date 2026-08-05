import math
import random
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


def bootstrap_paired_edges(scores_by_question: dict[str, dict[int, float]], rng: random.Random, n_resamples: int = 10000) -> tuple[tuple[float, float], tuple[float, float]]:
    primacy_dist = []
    recency_dist = []
    qids = list(scores_by_question.keys())
    n = len(qids)
    
    for _ in range(n_resamples):
        sampled = rng.choices(qids, k=n)
        def mean_acc(pos_list):
            scores = [scores_by_question[q][p] for q in sampled for p in pos_list]
            return sum(scores) / len(scores) if scores else 0.0
            
        primacy_dist.append(mean_acc([0, 1]) - mean_acc([4, 5]))
        recency_dist.append(mean_acc([8, 9]) - mean_acc([4, 5]))
        
    primacy_dist.sort()
    recency_dist.sort()
    
    ci_primacy = (primacy_dist[int(0.025 * n_resamples)], primacy_dist[int(0.975 * n_resamples)])
    ci_recency = (recency_dist[int(0.025 * n_resamples)], recency_dist[int(0.975 * n_resamples)])
    return ci_primacy, ci_recency


def validate_negative(records: list[dict], floor_records: list[dict] = None) -> None:
    # prompt length invariance
    lengths_by_q = {}
    for r in records:
        lengths_by_q.setdefault(r["question_id"], set()).add(r["prompt_token_count"])
    for qid, lengths in lengths_by_q.items():
        if len(lengths) > 1:
            raise ValueError(f"per-question length non-invariance detected: {lengths}")
            
    # structure scores
    scores_by_question = {}
    valid_count = 0
    
    for r in records:
        if r["score"] is None:
            continue
        qid = r["question_id"]
        pos = r["gold_position"]
        if qid not in scores_by_question:
            scores_by_question[qid] = {}
        scores_by_question[qid][pos] = r
        valid_count += 1
        
    if valid_count == 0:
        raise ValueError("no valid records found")
        
    # Check all 10 positions present per question, and mean diff from floor per question
    scores_for_bootstrap = {}
    for qid, pos_dict in scores_by_question.items():
        if len(pos_dict) != 10:
            raise ValueError(f"question {qid} does not have exactly 10 positions (found {len(pos_dict)})")
            
        mean_neg = sum(r["score"] for r in pos_dict.values()) / 10.0
        floor_acc = next(iter(pos_dict.values()))["floor_accuracy"]
        if abs(mean_neg - floor_acc) > 0.05:
            raise ValueError(f"negative control mean differs from floor by > 0.05 for question {qid}: {mean_neg} vs {floor_acc}")
            
        scores_for_bootstrap[qid] = {pos: r["score"] for pos, r in pos_dict.items()}
        
    # Flatness check
    rng = random.Random(42)
    ci_primacy, ci_recency = bootstrap_paired_edges(scores_for_bootstrap, rng)
    
    def contains_zero(ci):
        return ci[0] <= 0 <= ci[1]
        
    if not contains_zero(ci_primacy):
        raise ValueError(f"flatness CI for primacy does not contain 0: {ci_primacy}")
    if not contains_zero(ci_recency):
        raise ValueError(f"flatness CI for recency does not contain 0: {ci_recency}")


def validate_order(records: list[dict]) -> None:
    # Collect scores by (question, pos) -> list of scores across perms
    scores_by_group = {}
    for r in records:
        if r["score"] is None:
            continue
        key = (r["question_id"], r["gold_position"])
        scores_by_group.setdefault(key, []).append(r["score"])
        
    if not scores_by_group:
        raise ValueError("no valid records found")
        
    ranges = []
    for key, scores in scores_by_group.items():
        if len(scores) > 1:
            ranges.append(max(scores) - min(scores))
            
    if not ranges:
        raise ValueError("no permutations found to compare")
        
    mean_range = sum(ranges) / len(ranges)
    if mean_range > 0.10:
        raise ValueError(f"mean range across distractor permutations is > 0.10: {mean_range}")
