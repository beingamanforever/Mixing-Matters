import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

BOOTSTRAP_SEED = 240521
N_RESAMPLES = 10000

KV_SLOTS = tuple(range(10))
PHASE1_CONDITIONS = ("closed_book", "gold_first", "gold_middle", "oracle")
SCORE_FIELDS = (
    ("score", "primary"),
    ("score_normalized_em", "normalized_em"),
    ("score_first_line", "first_line"),
)


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[int(0.025 * n)], ordered[int(0.975 * n)]


def _kv_slot(condition: str) -> int:
    prefix = "kv_position_"
    if not condition.startswith(prefix):
        raise ValueError(f"unexpected kv-position condition: {condition}")
    return int(condition[len(prefix) :])


def kv_position_curve(records: list[dict]) -> dict:
    """Accuracy per key-value slot with a paired bootstrap 95 percent CI.

    Each resample draws whole control_id bundles (all ten slots for a unit
    move together), so every slot's estimate is backed by the same synthetic
    units in a given resample.
    """
    scores_by_control: dict[object, dict[int, float]] = {}
    for record in records:
        slot = _kv_slot(record["condition"])
        bucket = scores_by_control.setdefault(record["control_id"], {})
        if slot in bucket:
            raise ValueError(f"duplicate kv-position record: {record['control_id']} slot {slot}")
        bucket[slot] = float(record["score"])

    control_ids = sorted(scores_by_control, key=str)
    if not control_ids:
        raise ValueError("no positive-control records found")
    for control_id in control_ids:
        if set(scores_by_control[control_id]) != set(KV_SLOTS):
            raise ValueError(f"control_id {control_id} is missing kv-position slots")

    def slot_means(ids: list) -> dict[int, float]:
        return {
            slot: sum(scores_by_control[cid][slot] for cid in ids) / len(ids) for slot in KV_SLOTS
        }

    accuracy = slot_means(control_ids)

    rng = random.Random(BOOTSTRAP_SEED)
    draws: dict[int, list[float]] = {slot: [] for slot in KV_SLOTS}
    for _ in range(N_RESAMPLES):
        sampled = rng.choices(control_ids, k=len(control_ids))
        for slot, value in slot_means(sampled).items():
            draws[slot].append(value)

    slots = {
        slot: {"accuracy": accuracy[slot], "ci": list(_percentile_ci(draws[slot]))}
        for slot in KV_SLOTS
    }
    return {"n_bundles": len(control_ids), "slots": slots}


def _bundle_phase1_records(records: list[dict]) -> dict[str, dict[str, dict]]:
    bundles: dict[str, dict[str, dict]] = {}
    for record in records:
        per_question = bundles.setdefault(record["question_id"], {})
        condition = record["condition"]
        if condition in per_question:
            raise ValueError(f"duplicate Phase 1 record: {record['question_id']} / {condition}")
        per_question[condition] = record
    return bundles


def _condition_mean(bundles: dict, ids: list, condition: str, field: str) -> float | None:
    values = [
        bundles[qid][condition][field]
        for qid in ids
        if condition in bundles[qid] and bundles[qid][condition][field] is not None
    ]
    return sum(values) / len(values) if values else None


def _anchor_mean(bundles: dict, ids: list, field: str) -> float | None:
    values = []
    for qid in ids:
        for record in bundles[qid].values():
            value = record.get(field)
            if value is not None:
                values.append(value)
                break
    return sum(values) / len(values) if values else None


def _paired_diff_mean(
    bundles: dict, ids: list, field: str, condition_a: str, condition_b: str
) -> float | None:
    diffs = []
    for qid in ids:
        per_condition = bundles[qid]
        if condition_a not in per_condition or condition_b not in per_condition:
            continue
        value_a = per_condition[condition_a][field]
        value_b = per_condition[condition_b][field]
        if value_a is not None and value_b is not None:
            diffs.append(value_a - value_b)
    return sum(diffs) / len(diffs) if diffs else None


def phase1_condition_accuracy(records: list[dict]) -> dict:
    """Primary and scoring-variant accuracy per condition, plus floor and ceiling.

    Every quantity is bootstrapped by resampling complete question bundles (a
    question's records across every condition move together in one resample),
    so the reported numbers are drawn from one shared set of synthetic
    questions per resample. The gold_first vs gold_middle contrast is
    accumulated as a paired per-question difference in the same loop, since
    the two conditions share questions and are strongly correlated, so
    comparing their overlapping marginal intervals understates the evidence.
    """
    bundles = _bundle_phase1_records(records)
    question_ids = sorted(bundles)
    if not question_ids:
        raise ValueError("no Phase 1 records found")

    observed_conditions = {record["condition"] for record in records}
    missing_conditions = [
        condition for condition in PHASE1_CONDITIONS if condition not in observed_conditions
    ]
    if missing_conditions:
        raise ValueError(f"Phase 1 records are missing conditions: {', '.join(missing_conditions)}")

    def point_estimate(ids: list) -> dict:
        result = {
            metric_name: {
                condition: _condition_mean(bundles, ids, condition, field)
                for condition in PHASE1_CONDITIONS
            }
            for field, metric_name in SCORE_FIELDS
        }
        result["floor_accuracy"] = _anchor_mean(bundles, ids, "floor_accuracy")
        result["ceiling_accuracy"] = _anchor_mean(bundles, ids, "ceiling_accuracy")
        result["gold_first_minus_gold_middle"] = _paired_diff_mean(
            bundles, ids, "score", "gold_first", "gold_middle"
        )
        return result

    point = point_estimate(question_ids)

    rng = random.Random(BOOTSTRAP_SEED)
    draws = {
        metric_name: {condition: [] for condition in PHASE1_CONDITIONS}
        for _, metric_name in SCORE_FIELDS
    }
    draws["floor_accuracy"] = []
    draws["ceiling_accuracy"] = []
    draws["gold_first_minus_gold_middle"] = []
    for _ in range(N_RESAMPLES):
        sampled = rng.choices(question_ids, k=len(question_ids))
        resample = point_estimate(sampled)
        for _, metric_name in SCORE_FIELDS:
            for condition in PHASE1_CONDITIONS:
                value = resample[metric_name][condition]
                if value is not None:
                    draws[metric_name][condition].append(value)
        for field in ("floor_accuracy", "ceiling_accuracy", "gold_first_minus_gold_middle"):
            if resample[field] is not None:
                draws[field].append(resample[field])

    conditions_out = {}
    for condition in PHASE1_CONDITIONS:
        entry = {}
        for _, metric_name in SCORE_FIELDS:
            values = draws[metric_name][condition]
            entry[metric_name] = {
                "accuracy": point[metric_name][condition],
                "ci": list(_percentile_ci(values)) if values else None,
            }
        conditions_out[condition] = entry

    anchors_out = {}
    for field in ("floor_accuracy", "ceiling_accuracy"):
        values = draws[field]
        anchors_out[field] = {
            "mean": point[field],
            "ci": list(_percentile_ci(values)) if values else None,
        }

    diff_values = draws["gold_first_minus_gold_middle"]
    diff_out = {
        "estimate": point["gold_first_minus_gold_middle"],
        "ci": list(_percentile_ci(diff_values)) if diff_values else None,
    }

    return {
        "n_questions": len(question_ids),
        "conditions": conditions_out,
        "gold_first_minus_gold_middle": diff_out,
        **anchors_out,
    }


def _provenance_caption(records: list[dict]) -> str:
    """Name the model and the sample size so a figure stands alone in a slide."""
    first = records[0]
    model = first.get("model_name") or first.get("model", "unknown model")
    units = len({record.get("question_id", record.get("control_id")) for record in records})
    return f"{model}, {units} instances"


def write_figures(
    kv_records: list[dict], phase1_records: list[dict], directory: Path
) -> list[Path]:
    kv = kv_position_curve(kv_records)
    phase1 = phase1_condition_accuracy(phase1_records)

    kv_path = directory / "kv-position-curve.png"
    phase1_path = directory / "phase1-condition-accuracy.png"
    summary_path = directory / "figures-summary.json"
    for path in (kv_path, phase1_path, summary_path):
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    slots = sorted(kv["slots"])
    accuracies = [kv["slots"][slot]["accuracy"] for slot in slots]
    low = [kv["slots"][slot]["ci"][0] for slot in slots]
    high = [kv["slots"][slot]["ci"][1] for slot in slots]
    lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
    upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]

    fig, ax = plt.subplots()
    ax.errorbar(slots, accuracies, yerr=[lower_error, upper_error], fmt="o-", capsize=4)
    ax.set_title(f"Key-value position accuracy\n{_provenance_caption(kv_records)}")
    ax.set_xlabel("Key-value slot")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    ax.set_xticks(slots)
    fig.savefig(kv_path)
    plt.close(fig)

    accuracies = [
        phase1["conditions"][condition]["primary"]["accuracy"] for condition in PHASE1_CONDITIONS
    ]
    ci = [phase1["conditions"][condition]["primary"]["ci"] for condition in PHASE1_CONDITIONS]
    lower_error = [max(0.0, value - bound[0]) for value, bound in zip(accuracies, ci)]
    upper_error = [max(0.0, bound[1] - value) for value, bound in zip(accuracies, ci)]

    fig, ax = plt.subplots()
    ax.bar(PHASE1_CONDITIONS, accuracies, yerr=[lower_error, upper_error], capsize=4)
    ax.set_title(f"Phase 1 condition accuracy\n{_provenance_caption(phase1_records)}")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    fig.savefig(phase1_path)
    plt.close(fig)

    summary = {"kv_position_curve": kv, "phase1_condition_accuracy": phase1}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return [kv_path, phase1_path, summary_path]
