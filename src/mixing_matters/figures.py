import itertools
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .phase2 import DEFAULT_RESAMPLES, GOLD_POSITIONS, edges, interaction, position_curve
from .phase4 import scale_trend, trend_summary

# Publication-grade defaults applied to every figure this module writes, so the
# committed PNGs are consistent and re-running any report reproduces the same
# look. Styling only: no estimate, interval, or label is computed here.
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.figsize": (7.2, 4.6),
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.45,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.8,
    }
)

# Fixed family colors so Pythia and Mamba read the same across every panel.
PYTHIA_COLOR = "#1f77b4"
MAMBA_COLOR = "#ff7f0e"


def _family_colors(models: list[str]) -> dict[str, str]:
    """Map each model to a stable color: Pythia blue, Mamba orange.

    A study can contain more than one Mamba variant (Phase 2 has two), so the
    first Mamba takes the family color and any further model falls through to a
    distinct palette color, never colliding with the two family colors.
    """
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    spare = (color for color in palette if color not in (PYTHIA_COLOR, MAMBA_COLOR))
    colors: dict[str, str] = {}
    mamba_taken = False
    for model in models:
        lowered = model.lower()
        if "pythia" in lowered:
            colors[model] = PYTHIA_COLOR
        elif "mamba" in lowered and not mamba_taken:
            colors[model] = MAMBA_COLOR
            mamba_taken = True
        else:
            colors[model] = next(spare)
    return colors

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
    family_colors = _family_colors(models)
    for model in models:
        color = family_colors[model]
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
    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=1)
    primacy_x = [position - width / 2 for position in x]
    recency_x = [position + width / 2 for position in x]
    # Faint connecting lines carry the eye along the trend; the markers and
    # intervals carry the numbers.
    ax.plot(primacy_x, primacy_estimates, "-", color=PYTHIA_COLOR, alpha=0.35, zorder=2)
    ax.plot(recency_x, recency_estimates, "-", color="#ff7f0e", alpha=0.35, zorder=2)
    ax.errorbar(
        primacy_x,
        primacy_estimates,
        yerr=[primacy_lower, primacy_upper],
        fmt="o",
        capsize=4,
        color=PYTHIA_COLOR,
        markeredgecolor="white",
        zorder=3,
        label="Primacy edge difference",
    )
    ax.errorbar(
        recency_x,
        recency_estimates,
        yerr=[recency_lower, recency_upper],
        fmt="s",
        capsize=4,
        color="#ff7f0e",
        markeredgecolor="white",
        zorder=3,
        label="Recency edge difference",
    )
    ax.set_title("Pythia minus Mamba edge difference by scale pair")
    ax.set_xlabel("Scale pair, matched by parameter count")
    ax.set_ylabel("Edge difference, 95 percent bootstrap interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.margins(x=0.08)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(gap_path)
    plt.close(fig)

    curve = position_curve(records, n_resamples=n_resamples)

    def draw_pair(ax, pair) -> None:
        for model_key, family_label, color in (
            (pair["pythia_model"], "Pythia", PYTHIA_COLOR),
            (pair["mamba_model"], "Mamba", MAMBA_COLOR),
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
