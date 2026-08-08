import itertools
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .models import DATA_PAIR, MODELS, PHASE8_SYSTEMS
from .phase2 import DEFAULT_RESAMPLES, GOLD_POSITIONS, edges, interaction, position_curve
from .phase4 import scale_trend, trend_summary
from .phase5 import compare_to_architecture, data_control
from .phase6 import phase6_summary
from .phase8 import phase8_summary

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


def _floor_ceiling_means(records: list[dict]) -> dict[str, dict[str, float]]:
    """Mean floor and ceiling accuracy per model, one value per question.

    floor_accuracy and ceiling_accuracy are the same for every gold position
    of a question, so only the first gold record seen per question is kept.
    """
    per_model_question: dict[str, dict[str, tuple[float, float]]] = {}
    for record in records:
        if record.get("condition") != "gold":
            continue
        bucket = per_model_question.setdefault(record["model_key"], {})
        bucket.setdefault(
            record["question_id"], (record["floor_accuracy"], record["ceiling_accuracy"])
        )

    means = {}
    for model, questions in per_model_question.items():
        floors = [floor for floor, _ in questions.values()]
        ceilings = [ceiling for _, ceiling in questions.values()]
        means[model] = {
            "floor_accuracy": sum(floors) / len(floors),
            "ceiling_accuracy": sum(ceilings) / len(ceilings),
        }
    return means


def _model_caption(question_counts: dict[str, int]) -> str:
    """Name every model and its question count so a figure stands alone in a slide.

    Wrapped to a fixed character width, independent of the number of models,
    so the caption stays inside the figure instead of running off the edge.
    """
    caption = ", ".join(
        f"{model} ({count} questions)" for model, count in sorted(question_counts.items())
    )
    return "\n".join(textwrap.wrap(caption, width=60))


def phase2_summary(records: list[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """The position curve, edges, pairwise interactions, and floor/ceiling per model.

    This is the machine-readable form of both Phase 2 figures: the position
    curve and edges come straight from ``phase2.position_curve`` and
    ``phase2.edges``, interactions are computed for every pair of models
    found in ``edges``, and floor/ceiling means are the reference lines drawn
    on the position curve figure.
    """
    curve = position_curve(records, n_resamples=n_resamples)
    edge = edges(records, n_resamples=n_resamples)
    models = sorted(edge)

    interactions = [
        {
            "first_model": first_model,
            "second_model": second_model,
            **interaction(records, first_model, second_model, n_resamples=n_resamples),
        }
        for first_model, second_model in itertools.combinations(models, 2)
    ]

    return {
        "models": models,
        "n_resamples": n_resamples,
        "position_curve": curve,
        "edges": edge,
        "interactions": interactions,
        "floor_ceiling": _floor_ceiling_means(records),
    }


def write_phase2_figures(
    records: list[dict], directory: Path, n_resamples: int = DEFAULT_RESAMPLES
) -> list[Path]:
    """Write the Phase 2 position-curve and edge figures plus their summary.

    Refuses to overwrite any of ``position-curves.png``, ``position-edges.png``,
    or ``phase2-summary.json`` if they already exist in ``directory``.
    """
    summary = phase2_summary(records, n_resamples=n_resamples)

    curve_path = directory / "position-curves.png"
    edges_path = directory / "position-edges.png"
    summary_path = directory / "phase2-summary.json"
    for path in (curve_path, edges_path, summary_path):
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    models = summary["models"]
    curve = summary["position_curve"]
    edge = summary["edges"]
    floor_ceiling = summary["floor_ceiling"]
    question_counts = {model: curve[model]["positions"][0]["question_count"] for model in models}

    fig, ax = plt.subplots()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model, color in zip(models, itertools.cycle(color_cycle)):
        positions = curve[model]["positions"]
        accuracies = [positions[position]["accuracy"] for position in GOLD_POSITIONS]
        low = [positions[position]["ci_low"] for position in GOLD_POSITIONS]
        high = [positions[position]["ci_high"] for position in GOLD_POSITIONS]
        lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]
        ax.errorbar(
            GOLD_POSITIONS,
            accuracies,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=4,
            color=color,
            label=model,
        )
        # Floor/ceiling reference lines are subordinate to the curves: thin,
        # dotted, low alpha, and left out of the legend.
        ax.axhline(floor_ceiling[model]["floor_accuracy"], color=color, linestyle=":", alpha=0.35)
        ax.axhline(floor_ceiling[model]["ceiling_accuracy"], color=color, linestyle=":", alpha=0.35)
    ax.set_title(f"Position curve: accuracy by gold position\n{_model_caption(question_counts)}")
    ax.set_xlabel("Gold position (0 first, 9 last)")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(GOLD_POSITIONS))
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)

    edge_question_counts = {model: edge[model]["question_count"] for model in models}
    x = range(len(models))
    width = 0.35

    def edge_errors(edge_name: str) -> tuple[list[float], list[float], list[float]]:
        estimates = [edge[model][edge_name]["estimate"] for model in models]
        low = [edge[model][edge_name]["ci_low"] for model in models]
        high = [edge[model][edge_name]["ci_high"] for model in models]
        lower_error = [max(0.0, value - bound) for value, bound in zip(estimates, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(estimates, high)]
        return estimates, lower_error, upper_error

    primacy_estimates, primacy_lower, primacy_upper = edge_errors("primacy")
    recency_estimates, recency_lower, recency_upper = edge_errors("recency")

    fig, ax = plt.subplots()
    ax.bar(
        [position - width / 2 for position in x],
        primacy_estimates,
        width,
        yerr=[primacy_lower, primacy_upper],
        capsize=4,
        label="Primacy edge",
    )
    ax.bar(
        [position + width / 2 for position in x],
        recency_estimates,
        width,
        yerr=[recency_lower, recency_upper],
        capsize=4,
        label="Recency edge",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Position edges: primacy and recency\n{_model_caption(edge_question_counts)}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Edge minus center accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend()
    fig.tight_layout()
    fig.savefig(edges_path)
    plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return [curve_path, edges_path, summary_path]


def write_phase4_figures(
    records: list[dict], directory: Path, n_resamples: int = DEFAULT_RESAMPLES
) -> list[Path]:
    """Write the Phase 4 scale-trend figures plus their summary.

    Refuses to overwrite any of ``scale-primacy-gap.png``, ``scale-curves.png``,
    or ``phase4-summary.json`` if they already exist in ``directory``.
    """
    trend = scale_trend(records, n_resamples=n_resamples)
    trend_desc = trend_summary(trend)
    pairs = trend["pairs"]

    gap_path = directory / "scale-primacy-gap.png"
    curves_path = directory / "scale-curves.png"
    summary_path = directory / "phase4-summary.json"
    pair_paths = [directory / f"scale-curve-{pair['pair']}.png" for pair in pairs]
    for path in (gap_path, curves_path, summary_path, *pair_paths):
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    def diff_errors(field: str) -> tuple[list[float], list[float], list[float]]:
        estimates = [pair[field]["estimate"] for pair in pairs]
        low = [pair[field]["ci_low"] for pair in pairs]
        high = [pair[field]["ci_high"] for pair in pairs]
        lower_error = [max(0.0, value - bound) for value, bound in zip(estimates, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(estimates, high)]
        return estimates, lower_error, upper_error

    primacy_estimates, primacy_lower, primacy_upper = diff_errors("primacy_diff")
    recency_estimates, recency_lower, recency_upper = diff_errors("recency_diff")

    x = range(len(pairs))
    width = 0.15
    labels = [f"{pair['pair']}\n(n={pair['question_count']})" for pair in pairs]

    fig, ax = plt.subplots()
    ax.errorbar(
        [position - width / 2 for position in x],
        primacy_estimates,
        yerr=[primacy_lower, primacy_upper],
        fmt="o",
        capsize=4,
        label="Primacy edge difference",
    )
    ax.errorbar(
        [position + width / 2 for position in x],
        recency_estimates,
        yerr=[recency_lower, recency_upper],
        fmt="s",
        capsize=4,
        label="Recency edge difference",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Pythia minus Mamba edge difference by scale pair")
    ax.set_xlabel("Scale pair, matched by parameter count")
    ax.set_ylabel("Edge difference, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gap_path)
    plt.close(fig)

    curve = position_curve(records, n_resamples=n_resamples)

    def draw_pair(ax, pair) -> None:
        for model_key, family_label, color in (
            (pair["pythia_model"], "Pythia", "tab:blue"),
            (pair["mamba_model"], "Mamba", "tab:orange"),
        ):
            positions = curve[model_key]["positions"]
            accuracies = [positions[position]["accuracy"] for position in GOLD_POSITIONS]
            low = [positions[position]["ci_low"] for position in GOLD_POSITIONS]
            high = [positions[position]["ci_high"] for position in GOLD_POSITIONS]
            lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
            upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]
            question_count = positions[0]["question_count"]
            ax.errorbar(
                GOLD_POSITIONS,
                accuracies,
                yerr=[lower_error, upper_error],
                fmt="o-",
                capsize=3,
                color=color,
                label=f"{family_label} ({model_key}, {question_count} questions)",
            )
        ax.set_xlabel("Gold position (0 first, 9 last)")
        ax.set_xticks(list(GOLD_POSITIONS))
        ax.legend(fontsize="small")

    # One standalone figure per size pair, so each comparison reads on its own.
    for pair, pair_path in zip(pairs, pair_paths):
        fig, ax = plt.subplots()
        draw_pair(ax, pair)
        ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
        ax.set_title(f"Position curve at scale pair {pair['pair']}: Pythia vs Mamba")
        fig.tight_layout()
        fig.savefig(pair_path)
        plt.close(fig)

    # A shared-axis grid for a single side-by-side view of the whole trend.
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4), sharey=True, squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        draw_pair(ax, pair)
        ax.set_title(pair["pair"])
    axes[0][0].set_ylabel("Accuracy, 95 percent bootstrap interval")
    fig.suptitle("Position curves by scale pair: Pythia vs Mamba")
    fig.tight_layout()
    fig.savefig(curves_path)
    plt.close(fig)

    summary_path.write_text(
        json.dumps({"scale_trend": trend, "trend_summary": trend_desc}, indent=2, sort_keys=True)
    )

    return [gap_path, curves_path, summary_path, *pair_paths]


def _corpus_label(model_key: str) -> str:
    """A figure-friendly name pairing the corpus with the model key."""
    corpus = MODELS[model_key].training_corpus or "unknown"
    return f"{corpus} ({model_key})"


def phase5_summary(
    records: list[dict],
    architecture_records: list[dict] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """The corpus contrast, per-model curves and edges, and floor/ceiling.

    When ``architecture_records`` is given (the Phase 2 Pythia and Pile-Mamba
    sweeps), the Pythia-minus-Mamba architecture interaction is computed and the
    Phase 5 corpus effect is placed beside it under ``cross_phase``, so the
    data-control result can be read against the architecture result on one
    scale. Without it, those two keys are omitted.
    """
    control = data_control(records, n_resamples=n_resamples)
    curve = position_curve(records, n_resamples=n_resamples)
    floor_ceiling = _floor_ceiling_means(records)

    summary = {
        "models": list(DATA_PAIR),
        "n_resamples": n_resamples,
        "data_control": control,
        "position_curve": curve,
        "floor_ceiling": {key: floor_ceiling[key] for key in DATA_PAIR if key in floor_ceiling},
    }

    if architecture_records is not None:
        architecture = interaction(
            architecture_records, "pythia-2.8b", "mamba-2.8b", n_resamples=n_resamples
        )
        summary["architecture_interaction"] = {
            "first_model": "pythia-2.8b",
            "second_model": "mamba-2.8b",
            **architecture,
        }
        summary["cross_phase"] = compare_to_architecture(control, architecture)

    return summary


def write_phase5_figures(
    records: list[dict],
    directory: Path,
    architecture_records: list[dict] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> list[Path]:
    """Write the Phase 5 data-control figures plus their summary.

    Draws the Pile and SlimPajama position curves on one axis, the primacy and
    recency edges per model, and the corpus edge effect (beside the Phase 2
    architecture effect when ``architecture_records`` is given). Refuses to
    overwrite any existing output.
    """
    summary = phase5_summary(records, architecture_records, n_resamples=n_resamples)
    control = summary["data_control"]
    curve = summary["position_curve"]
    floor_ceiling = summary["floor_ceiling"]

    curve_path = directory / "position-curves.png"
    edges_path = directory / "position-edges.png"
    effect_path = directory / "corpus-effect.png"
    summary_path = directory / "phase5-summary.json"
    for path in (curve_path, edges_path, effect_path, summary_path):
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    models = list(DATA_PAIR)
    question_counts = {model: curve[model]["positions"][0]["question_count"] for model in models}

    fig, ax = plt.subplots()
    for model, color in (
        (DATA_PAIR[0], "tab:blue"),
        (DATA_PAIR[1], "tab:orange"),
    ):
        positions = curve[model]["positions"]
        accuracies = [positions[position]["accuracy"] for position in GOLD_POSITIONS]
        low = [positions[position]["ci_low"] for position in GOLD_POSITIONS]
        high = [positions[position]["ci_high"] for position in GOLD_POSITIONS]
        lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]
        ax.errorbar(
            GOLD_POSITIONS,
            accuracies,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=4,
            color=color,
            label=_corpus_label(model),
        )
        ax.axhline(floor_ceiling[model]["floor_accuracy"], color=color, linestyle=":", alpha=0.35)
        ax.axhline(floor_ceiling[model]["ceiling_accuracy"], color=color, linestyle=":", alpha=0.35)
    ax.set_title(
        "Position curve by training corpus (2.8B Mamba, architecture fixed)\n"
        f"{_model_caption(question_counts)}"
    )
    ax.set_xlabel("Gold position (0 first, 9 last)")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(GOLD_POSITIONS))
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)

    edge_map = {
        DATA_PAIR[0]: control["pile_edges"],
        DATA_PAIR[1]: control["slimpajama_edges"],
    }
    x = range(len(models))
    width = 0.35

    def edge_errors(edge_name: str) -> tuple[list[float], list[float], list[float]]:
        estimates = [edge_map[model][edge_name]["estimate"] for model in models]
        low = [edge_map[model][edge_name]["ci_low"] for model in models]
        high = [edge_map[model][edge_name]["ci_high"] for model in models]
        lower_error = [max(0.0, value - bound) for value, bound in zip(estimates, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(estimates, high)]
        return estimates, lower_error, upper_error

    primacy_estimates, primacy_lower, primacy_upper = edge_errors("primacy")
    recency_estimates, recency_lower, recency_upper = edge_errors("recency")

    fig, ax = plt.subplots()
    ax.bar(
        [position - width / 2 for position in x],
        primacy_estimates,
        width,
        yerr=[primacy_lower, primacy_upper],
        capsize=4,
        label="Primacy edge",
    )
    ax.bar(
        [position + width / 2 for position in x],
        recency_estimates,
        width,
        yerr=[recency_lower, recency_upper],
        capsize=4,
        label="Recency edge",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Position edges by training corpus")
    ax.set_xlabel("Training corpus")
    ax.set_ylabel("Edge minus center accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_corpus_label(model) for model in models])
    ax.legend()
    fig.tight_layout()
    fig.savefig(edges_path)
    plt.close(fig)

    # The corpus edge effect (Pile minus SlimPajama), beside the Phase 2
    # architecture effect (Pythia minus Mamba) when it is available, so the two
    # sources of curve difference read on one scale.
    effect_groups = [("Primacy", "primacy_diff", "primacy"), ("Recency", "recency_diff", "recency")]
    corpus_values = [control[field]["estimate"] for _, field, _ in effect_groups]
    corpus_low = [control[field]["ci_low"] for _, field, _ in effect_groups]
    corpus_high = [control[field]["ci_high"] for _, field, _ in effect_groups]
    corpus_lower = [max(0.0, value - bound) for value, bound in zip(corpus_values, corpus_low)]
    corpus_upper = [max(0.0, bound - value) for value, bound in zip(corpus_values, corpus_high)]

    ex = range(len(effect_groups))
    fig, ax = plt.subplots()
    has_architecture = "architecture_interaction" in summary
    bar_width = 0.35 if has_architecture else 0.5
    offset = bar_width / 2 if has_architecture else 0.0
    ax.bar(
        [position - offset for position in ex],
        corpus_values,
        bar_width,
        yerr=[corpus_lower, corpus_upper],
        capsize=4,
        label="Corpus effect (Pile - SlimPajama)",
    )
    if has_architecture:
        architecture = summary["architecture_interaction"]
        arch_values = [architecture[key]["estimate"] for _, _, key in effect_groups]
        arch_low = [architecture[key]["ci_low"] for _, _, key in effect_groups]
        arch_high = [architecture[key]["ci_high"] for _, _, key in effect_groups]
        arch_lower = [max(0.0, value - bound) for value, bound in zip(arch_values, arch_low)]
        arch_upper = [max(0.0, bound - value) for value, bound in zip(arch_values, arch_high)]
        ax.bar(
            [position + offset for position in ex],
            arch_values,
            bar_width,
            yerr=[arch_lower, arch_upper],
            capsize=4,
            label="Architecture effect (Pythia - Mamba, Pile)",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Edge difference: training corpus vs architecture")
    ax.set_ylabel("Edge difference, 95 percent bootstrap interval")
    ax.set_xticks(list(ex))
    ax.set_xticklabels([name for name, _, _ in effect_groups])
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(effect_path)
    plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return [curve_path, edges_path, effect_path, summary_path]


def _draw_position_curve(ax, models, curve, floor_ceiling) -> None:
    """Draw one accuracy-by-position curve per model, with anchor reference lines."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model, color in zip(models, itertools.cycle(color_cycle)):
        positions = curve[model]["positions"]
        accuracies = [positions[position]["accuracy"] for position in GOLD_POSITIONS]
        low = [positions[position]["ci_low"] for position in GOLD_POSITIONS]
        high = [positions[position]["ci_high"] for position in GOLD_POSITIONS]
        lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]
        ax.errorbar(
            GOLD_POSITIONS,
            accuracies,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=4,
            color=color,
            label=model,
        )
        if model in floor_ceiling:
            ax.axhline(floor_ceiling[model]["floor_accuracy"], color=color, linestyle=":", alpha=0.35)
            ax.axhline(
                floor_ceiling[model]["ceiling_accuracy"], color=color, linestyle=":", alpha=0.35
            )
    ax.set_xlabel("Needle depth (0 first, 9 last)")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(GOLD_POSITIONS))
    ax.legend()


def _draw_edges(ax, models, edge) -> None:
    """Draw grouped primacy and recency edge bars per model."""
    x = range(len(models))
    width = 0.35

    def edge_errors(edge_name: str) -> tuple[list[float], list[float], list[float]]:
        estimates = [edge[model][edge_name]["estimate"] for model in models]
        low = [edge[model][edge_name]["ci_low"] for model in models]
        high = [edge[model][edge_name]["ci_high"] for model in models]
        lower_error = [max(0.0, value - bound) for value, bound in zip(estimates, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(estimates, high)]
        return estimates, lower_error, upper_error

    primacy_estimates, primacy_lower, primacy_upper = edge_errors("primacy")
    recency_estimates, recency_lower, recency_upper = edge_errors("recency")
    ax.bar(
        [position - width / 2 for position in x],
        primacy_estimates,
        width,
        yerr=[primacy_lower, primacy_upper],
        capsize=4,
        label="Primacy edge",
    )
    ax.bar(
        [position + width / 2 for position in x],
        recency_estimates,
        width,
        yerr=[recency_lower, recency_upper],
        capsize=4,
        label="Recency edge",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Edge minus center accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend()


def write_phase6_figures(
    records: list[dict],
    directory: Path,
    qa_records: list[dict] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> list[Path]:
    """Write per-length niah position curves and edges plus a summary.

    One position-curve and one edge figure per context length, so each length
    reads on its own, and a niah-versus-QA edge comparison per length when the
    Phase 2 sweep is supplied. Refuses to overwrite any existing output.
    """
    summary = phase6_summary(records, qa_records, n_resamples=n_resamples)
    lengths = summary["lengths"]

    curve_paths = {length: directory / f"position-curve-{length}.png" for length in lengths}
    edge_paths = {length: directory / f"position-edges-{length}.png" for length in lengths}
    summary_path = directory / "phase6-summary.json"
    comparison_paths = {}
    if "task_comparison" in summary:
        comparison_paths = {
            length: directory / f"task-comparison-{length}.png" for length in lengths
        }
    planned = [*curve_paths.values(), *edge_paths.values(), *comparison_paths.values(), summary_path]
    for path in planned:
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    for length in lengths:
        entry = lengths[length]
        models = entry["models"]
        curve = entry["position_curve"]
        edge = entry["edges"]
        floor_ceiling = entry["floor_ceiling"]
        question_count = curve[models[0]]["positions"][0]["question_count"]

        fig, ax = plt.subplots()
        _draw_position_curve(ax, models, curve, floor_ceiling)
        ax.set_title(
            f"niah_single_1 position curve at {length} tokens\n"
            f"{len(models)} models, {question_count} needle instances each"
        )
        fig.tight_layout()
        fig.savefig(curve_paths[length])
        plt.close(fig)

        fig, ax = plt.subplots()
        _draw_edges(ax, models, edge)
        ax.set_title(f"niah_single_1 position edges at {length} tokens")
        ax.set_xlabel("Model")
        fig.tight_layout()
        fig.savefig(edge_paths[length])
        plt.close(fig)

    if "task_comparison" in summary:
        qa_edges = summary["qa_edges"]
        for length in lengths:
            niah_edge = lengths[length]["edges"]
            models = [model for model in lengths[length]["models"] if model in qa_edges]
            x = range(len(models))
            width = 0.2
            fig, ax = plt.subplots()
            series = [
                ("niah primacy", niah_edge, "primacy", -1.5),
                ("QA primacy", qa_edges, "primacy", -0.5),
                ("niah recency", niah_edge, "recency", 0.5),
                ("QA recency", qa_edges, "recency", 1.5),
            ]
            for label, edge_map, edge_name, offset in series:
                estimates = [edge_map[model][edge_name]["estimate"] for model in models]
                ax.bar([position + offset * width for position in x], estimates, width, label=label)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"niah vs QA edges at {length} tokens (QA is Phase 2)")
            ax.set_ylabel("Edge minus center accuracy")
            ax.set_xticks(list(x))
            ax.set_xticklabels(models)
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(comparison_paths[length])
            plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return [*curve_paths.values(), *edge_paths.values(), *comparison_paths.values(), summary_path]


def write_phase8_figures(
    records: list[dict], directory: Path, n_resamples: int = DEFAULT_RESAMPLES
) -> list[Path]:
    """Write the Phase 8 descriptive system comparison figures plus summary.

    Draws position curves for every Phase 8 system on one axis with per-model
    floor and ceiling reference lines, a grouped primacy/recency edge chart,
    and writes ``phase8-summary.json`` carrying per-model curves, edges,
    every pairwise interaction, and the small system descriptor block.
    Refuses to overwrite any existing output.
    """
    summary = phase8_summary(records, n_resamples=n_resamples)
    models = summary["models"]
    curve = summary["position_curve"]
    edge = summary["edges"]
    floor_ceiling = summary["floor_ceiling"]

    curve_path = directory / "position-curves.png"
    edges_path = directory / "position-edges.png"
    summary_path = directory / "phase8-summary.json"
    for path in (curve_path, edges_path, summary_path):
        if path.exists():
            raise FileExistsError(path)

    directory.mkdir(parents=True, exist_ok=True)

    question_counts = {model: curve[model]["positions"][0]["question_count"] for model in models}

    fig, ax = plt.subplots()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model, color in zip(models, itertools.cycle(color_cycle)):
        positions = curve[model]["positions"]
        accuracies = [positions[position]["accuracy"] for position in GOLD_POSITIONS]
        low = [positions[position]["ci_low"] for position in GOLD_POSITIONS]
        high = [positions[position]["ci_high"] for position in GOLD_POSITIONS]
        lower_error = [max(0.0, value - bound) for value, bound in zip(accuracies, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(accuracies, high)]
        ax.errorbar(
            GOLD_POSITIONS,
            accuracies,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=4,
            color=color,
            label=model,
        )
        if model in floor_ceiling:
            ax.axhline(
                floor_ceiling[model]["floor_accuracy"], color=color, linestyle=":", alpha=0.35
            )
            ax.axhline(
                floor_ceiling[model]["ceiling_accuracy"], color=color, linestyle=":", alpha=0.35
            )
    ax.set_title(
        "Phase 8 position curves (descriptive comparison across full systems)\n"
        f"{_model_caption(question_counts)}"
    )
    ax.set_xlabel("Gold position (0 first, 9 last)")
    ax.set_ylabel("Accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(GOLD_POSITIONS))
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)

    edge_question_counts = {model: edge[model]["question_count"] for model in models}
    x = range(len(models))
    width = 0.35

    def edge_errors(edge_name: str) -> tuple[list[float], list[float], list[float]]:
        estimates = [edge[model][edge_name]["estimate"] for model in models]
        low = [edge[model][edge_name]["ci_low"] for model in models]
        high = [edge[model][edge_name]["ci_high"] for model in models]
        lower_error = [max(0.0, value - bound) for value, bound in zip(estimates, low)]
        upper_error = [max(0.0, bound - value) for value, bound in zip(estimates, high)]
        return estimates, lower_error, upper_error

    primacy_estimates, primacy_lower, primacy_upper = edge_errors("primacy")
    recency_estimates, recency_lower, recency_upper = edge_errors("recency")

    fig, ax = plt.subplots()
    ax.bar(
        [position - width / 2 for position in x],
        primacy_estimates,
        width,
        yerr=[primacy_lower, primacy_upper],
        capsize=4,
        label="Primacy edge",
    )
    ax.bar(
        [position + width / 2 for position in x],
        recency_estimates,
        width,
        yerr=[recency_lower, recency_upper],
        capsize=4,
        label="Recency edge",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Phase 8 position edges by system\n{_model_caption(edge_question_counts)}")
    ax.set_xlabel("System")
    ax.set_ylabel("Edge minus center accuracy, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend()
    fig.tight_layout()
    fig.savefig(edges_path)
    plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return [curve_path, edges_path, summary_path]
