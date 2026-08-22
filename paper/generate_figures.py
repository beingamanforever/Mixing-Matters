#!/usr/bin/env python3
"""Generate the paper's canonical SVG figures and matching PDF companions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

PHASE1 = "artifacts/phase1/figures/figures-summary.json"
PHASE2 = "artifacts/phase2/report/phase2-summary.json"
PHASE3 = "artifacts/phase3/report/phase3-summary.json"
PHASE4 = "artifacts/phase4/report/phase4-summary.json"
PHASE5 = "artifacts/phase5/report/phase5-summary.json"
PHASE6 = "artifacts/phase6/report/phase6-summary.json"
PHASE7 = "artifacts/phase7-mechanisms/report/phase7-summary.json"
SINK = "artifacts/phase7-mechanisms/4c-sink-scan/report/sink-mass-summary.json"
QUERY = "artifacts/phase7-mechanisms/4a-query-position/report/phase7-variants-summary.json"
TEMPLATE = "artifacts/phase7-mechanisms/4e-template/report/phase7-variants-summary.json"
MAMBA_PROBE = "artifacts/phase7-mechanisms/4d-probe/mamba-2.8b-layer32-probe.json"
PYTHIA_PROBE = "artifacts/phase7-mechanisms/4d-probe/pythia-2.8b-layer16-probe.json"
PHASE8 = "artifacts/phase8/report/phase8-summary.json"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#5F6368"
BLACK = "#222222"
ENCODING = "attention=blue; state-space=orange; Pythia=star; Mamba=circle; Mamba-2=diamond"

STYLES = {
    "pythia-160m": (BLUE, "*"),
    "pythia-410m": (BLUE, "*"),
    "pythia-1b": (BLUE, "*"),
    "pythia-1.4b": (BLUE, "*"),
    "pythia-2.8b": (BLUE, "*"),
    "mamba-130m": (ORANGE, "o"),
    "mamba-370m": (ORANGE, "o"),
    "mamba-790m": (ORANGE, "o"),
    "mamba-1.4b": (ORANGE, "o"),
    "mamba-2.8b": (ORANGE, "o"),
    "mamba-2.8b-slimpj": (ORANGE, "^"),
    "mamba2-2.7b": (ORANGE, "D"),
    "mamba2-8b": (ORANGE, "D"),
    "mamba2-hybrid-8b": (BLUE, "P"),
    "nemotron-h-8b": (ORANGE, "D"),
    "llama-3.1-8b": (BLUE, "*"),
    "qwen2.5-7b": (BLUE, "s"),
}

LABELS = {
    "pythia-160m": "Pythia 160M",
    "pythia-410m": "Pythia 410M",
    "pythia-1b": "Pythia 1B",
    "pythia-1.4b": "Pythia 1.4B",
    "pythia-2.8b": "Pythia 2.8B",
    "mamba-130m": "Mamba 130M",
    "mamba-370m": "Mamba 370M",
    "mamba-790m": "Mamba 790M",
    "mamba-1.4b": "Mamba 1.4B",
    "mamba-2.8b": "Mamba 2.8B",
    "mamba-2.8b-slimpj": "Mamba 2.8B, SlimPajama",
    "mamba2-2.7b": "Mamba-2 2.7B",
    "mamba2-8b": "Mamba-2 8B",
    "mamba2-hybrid-8b": "Hybrid Mamba-2 8B",
    "nemotron-h-8b": "Nemotron-H 8B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "qwen2.5-7b": "Qwen2.5 7B",
}

FIGURES = (
    "paper-phase2-position",
    "paper-phase3-position",
    "paper-phase4-scale",
    "paper-phase5-corpus",
    "paper-phase1-calibration",
    "paper-phase6-task",
    "paper-phase7-mechanisms",
    "paper-phase8-production",
)

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#D7DADD",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.7,
    "grid.linestyle": "-",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "lines.linewidth": 1.45,
    "lines.markersize": 4.8,
    "lines.markeredgewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "svg.fonttype": "none",
    "svg.hashsalt": "mixing-matters-paper",
    "pdf.fonttype": 42,
}
plt.rcParams.update(PAPER_STYLE)


def _read(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def _style(model: str) -> tuple[str, str]:
    return STYLES[model]


def _label(model: str) -> str:
    return LABELS[model]


def _ci_error(point: Mapping[str, float], value_key: str) -> list[list[float]]:
    value = point[value_key]
    return [[value - point["ci_low"]], [point["ci_high"] - value]]


def _panel_label(ax: Axes, label: str) -> None:
    ax.text(-0.16, 1.03, label, transform=ax.transAxes, fontsize=8, fontweight="bold")


def _position_curve(
    ax: Axes,
    curves: Mapping[str, Any],
    models: Sequence[str],
    *,
    line_styles: Mapping[str, str] | None = None,
) -> None:
    for model in models:
        positions = curves[model]["positions"]
        x = sorted(int(position) for position in positions)
        points = [positions[str(position)] for position in x]
        y = [point["accuracy"] for point in points]
        low = [point["ci_low"] for point in points]
        high = [point["ci_high"] for point in points]
        color, marker = _style(model)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=(line_styles or {}).get(model, "-"),
            label=_label(model),
        )
    ax.set_xlabel("Gold passage position")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(10), [str(position + 1) for position in range(10)])
    ax.legend(ncol=1, handlelength=2.2)


def _save(fig: Figure, output: Path, stem: str, sources: Iterable[str]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    source_text = "; ".join(sources)
    description = f"Sources: {source_text}. Encoding: {ENCODING}."
    paths = []
    for suffix in ("svg", "pdf"):
        target = output / f"{stem}.{suffix}"
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise FileExistsError(f"refusing to replace non-regular output: {target}")
        metadata = (
            {
                "Title": stem,
                "Description": description,
                "Creator": "paper/generate_figures.py",
                "Date": None,
            }
            if suffix == "svg"
            else {
                "Title": stem,
                "Subject": description,
                "Creator": "paper/generate_figures.py",
                "CreationDate": None,
                "ModDate": None,
            }
        )
        with tempfile.NamedTemporaryFile(dir=output, suffix=f".{suffix}", delete=False) as file:
            temporary = Path(file.name)
        try:
            fig.savefig(temporary, format=suffix, metadata=metadata)
            if suffix == "svg":
                lines = temporary.read_text().splitlines()
                temporary.write_text("\n".join(line.rstrip() for line in lines) + "\n")
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(target)
    plt.close(fig)
    return paths


def _phase2(output: Path) -> list[Path]:
    summary = _read(PHASE2)
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    _position_curve(ax, summary["position_curve"], summary["models"])
    return _save(fig, output, FIGURES[0], [PHASE2])


def _phase3(output: Path) -> list[Path]:
    summary = _read(PHASE3)
    models = ["mamba2-8b", "mamba2-hybrid-8b"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    _position_curve(ax, summary["position_curve"], models)
    return _save(fig, output, FIGURES[1], [PHASE3])


def _phase4(output: Path) -> list[Path]:
    pairs = _read(PHASE4)["scale_trend"]["pairs"]
    x = [(pair["pythia_params_millions"] + pair["mamba_params_millions"]) / 2 for pair in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), layout="constrained")
    for key, model_key in (("pythia_edges", "pythia_model"), ("mamba_edges", "mamba_model")):
        models = [pair[model_key] for pair in pairs]
        points = [pair[key]["primacy"] for pair in pairs]
        color, marker = _style(models[-1])
        axes[0].errorbar(
            x,
            [point["estimate"] for point in points],
            yerr=[
                [point["estimate"] - point["ci_low"] for point in points],
                [point["ci_high"] - point["estimate"] for point in points],
            ],
            color=color,
            marker=marker,
            capsize=2,
            label="Pythia" if key == "pythia_edges" else "Mamba",
        )
    interactions = [pair["primacy_diff"] for pair in pairs]
    axes[1].errorbar(
        x,
        [point["estimate"] for point in interactions],
        yerr=[
            [point["estimate"] - point["ci_low"] for point in interactions],
            [point["ci_high"] - point["estimate"] for point in interactions],
        ],
        color=BLUE,
        marker="s",
        markerfacecolor="white",
        capsize=2,
        label="Pythia minus Mamba",
    )
    for label, ax in zip(("a", "b"), axes, strict=True):
        _panel_label(ax, label)
        ax.axhline(0, color="#777777", linewidth=0.7, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(x, ["145", "390", "895", "1.4k", "2.8k"])
        ax.minorticks_off()
        ax.set_xlabel("Mean parameters (millions)")
        ax.legend()
    axes[0].set_ylabel("Primacy effect")
    axes[1].set_ylabel("Paired primacy difference")
    return _save(fig, output, FIGURES[2], [PHASE4])


def _phase5(output: Path) -> list[Path]:
    summary = _read(PHASE5)
    models = ["mamba-2.8b", "mamba-2.8b-slimpj"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    _position_curve(
        ax,
        summary["position_curve"],
        models,
        line_styles={"mamba-2.8b-slimpj": "--"},
    )
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "Pile"
    labels[1] = "SlimPajama"
    ax.legend(handles, labels)
    return _save(fig, output, FIGURES[3], [PHASE5])


def _phase1(output: Path) -> list[Path]:
    summary = _read(PHASE1)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), layout="constrained")
    slots = summary["kv_position_curve"]["slots"]
    x = sorted(int(slot) for slot in slots)
    points = [slots[str(slot)] for slot in x]
    axes[0].errorbar(
        x,
        [point["accuracy"] for point in points],
        yerr=[
            [point["accuracy"] - point["ci"][0] for point in points],
            [point["ci"][1] - point["accuracy"] for point in points],
        ],
        color=BLUE,
        marker="s",
        markerfacecolor="white",
        capsize=2,
    )
    axes[0].set_xlabel("Key-value slot")
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_xticks(x, [str(slot + 1) for slot in x])

    calibration = summary["phase1_condition_accuracy"]
    condition_keys = ["closed_book", "gold_middle", "gold_first", "oracle"]
    condition_labels = ["Closed\nbook", "Gold\nmiddle", "Gold\nfirst", "Oracle"]
    condition_points = [calibration["conditions"][key]["primary"] for key in condition_keys]
    axes[1].errorbar(
        range(len(condition_keys)),
        [point["accuracy"] for point in condition_points],
        yerr=[
            [point["accuracy"] - point["ci"][0] for point in condition_points],
            [point["ci"][1] - point["accuracy"] for point in condition_points],
        ],
        color=BLUE,
        marker="*",
        capsize=2,
        label="QA condition",
    )
    for key, color, label in (
        ("floor_accuracy", GRAY, "Floor"),
        ("ceiling_accuracy", BLACK, "Ceiling"),
    ):
        anchor = calibration[key]
        axes[1].axhspan(anchor["ci"][0], anchor["ci"][1], color=color, alpha=0.07, linewidth=0)
        axes[1].axhline(anchor["mean"], color=color, linestyle="--", linewidth=1, label=label)
    axes[1].set_xlabel("Calibration condition")
    axes[1].set_ylabel("Answer accuracy")
    axes[1].set_xticks(range(len(condition_keys)), condition_labels)
    axes[1].legend(ncol=3, columnspacing=0.9, handlelength=1.5)
    for label, ax in zip(("a", "b"), axes, strict=True):
        _panel_label(ax, label)
    return _save(fig, output, FIGURES[4], [PHASE1])


def _phase6(output: Path) -> list[Path]:
    summary = _read(PHASE6)
    models = ["mamba-2.8b", "mamba2-2.7b", "pythia-2.8b"]
    qa = summary["qa_edges"]
    niah = summary["lengths"]["2048"]["edges"]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 2.15), sharex=True, sharey=True, layout="constrained"
    )
    rows = list(reversed(range(len(models))))
    for ax, edge, panel in zip(axes, ("primacy", "recency"), ("a", "b"), strict=True):
        for row, model in zip(rows, models, strict=True):
            color, marker = _style(model)
            for offset, source, task, filled in (
                (0.11, qa, "QA", True),
                (-0.11, niah, "Needle, 2048 tokens", False),
            ):
                point = source[model][edge]
                ax.errorbar(
                    point["estimate"],
                    row + offset,
                    xerr=_ci_error(point, "estimate"),
                    color=color,
                    marker=marker,
                    markerfacecolor=color if filled else "white",
                    capsize=2,
                    linestyle="none",
                    label=task if row == rows[0] else None,
                )
        ax.text(
            0 if panel == "a" else 1,
            1.03,
            f"{panel}  {edge.capitalize()}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left" if panel == "a" else "right",
        )
        ax.axvline(0, color="#777777", linewidth=0.7, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.05, 0.21)
    axes[0].set_yticks(rows, ["Mamba", "Mamba-2", "Pythia"])
    fig.supxlabel("Edge effect", fontsize=8)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.16),
        columnspacing=1.5,
    )
    return _save(fig, output, FIGURES[5], [PHASE6])


def _final_sink_mass(model: Mapping[str, Any]) -> float:
    per_position = [
        position["mean_sink_mass_per_layer"][-1] for position in model["positions"].values()
    ]
    return sum(per_position) / len(per_position)


def _variant_points(
    ax: Axes,
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    models: Sequence[str],
) -> None:
    for model in models:
        color, marker = _style(model)
        points = [source["edges"][model][variant]["primacy"] for variant, source in sources]
        ax.errorbar(
            range(len(points)),
            [point["estimate"] for point in points],
            yerr=[
                [point["estimate"] - point["ci_low"] for point in points],
                [point["ci_high"] - point["estimate"] for point in points],
            ],
            color=color,
            marker=marker,
            capsize=2,
            label=_label(model),
        )
    ax.axhline(0, color="#777777", linewidth=0.7, zorder=0)


def _phase7(output: Path) -> list[Path]:
    depth = _read(PHASE7)["depth_trend"]["models"]
    sink = _read(SINK)["by_model"]
    query = _read(QUERY)
    template = _read(TEMPLATE)
    probes = {"mamba-2.8b": _read(MAMBA_PROBE), "pythia-2.8b": _read(PYTHIA_PROBE)}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), layout="constrained")

    by_model = {entry["model_key"]: entry for entry in depth}
    annotation_offsets = {
        "pythia-160m": (3, -9),
        "pythia-410m": (3, 2),
        "pythia-1b": (3, -9),
        "pythia-1.4b": (3, 4),
        "pythia-2.8b": (3, 2),
    }
    for model in sink:
        point = by_model[model]["primacy"]
        color, marker = _style(model)
        axes[0].errorbar(
            _final_sink_mass(sink[model]),
            point["estimate"],
            yerr=_ci_error(point, "estimate"),
            color=color,
            marker=marker,
            capsize=2,
            linestyle="none",
        )
        axes[0].annotate(
            _label(model).replace("Pythia ", ""),
            (_final_sink_mass(sink[model]), point["estimate"]),
            xytext=annotation_offsets[model],
            textcoords="offset points",
            fontsize=6,
        )
    axes[0].axhline(0, color="#777777", linewidth=0.7, zorder=0)
    axes[0].set_xlabel("Final-layer sink mass")
    axes[0].set_ylabel("Primacy effect")

    for index, (model, probe) in enumerate(probes.items()):
        color, marker = _style(model)
        axes[1].plot(
            index,
            probe["accuracy"],
            color=color,
            marker=marker,
            label="Observed" if index == 0 else None,
        )
        axes[1].plot(
            index,
            probe["shuffled_accuracy"],
            color=color,
            marker=marker,
            markerfacecolor="white",
            label="Shuffled" if index == 0 else None,
        )
    axes[1].set_xticks(range(2), ["Mamba", "Pythia"])
    axes[1].set_xlabel("Linear probe model")
    axes[1].set_ylabel("5-fold probe accuracy")
    axes[1].legend()

    variants = [
        ("baseline", query),
        ("bookend", query),
        ("gold_padded", query),
        ("question_first", query),
        ("baseline+tmpl:concise", template),
        ("baseline+tmpl:instructional", template),
    ]
    _variant_points(axes[2], variants, ["mamba-2.8b", "pythia-2.8b"])
    axes[2].set_xticks(
        range(len(variants)),
        ["Base", "Bookend", "Gold\npadded", "Question\nfirst", "Concise", "Instructional"],
        rotation=40,
        ha="right",
    )
    axes[2].tick_params(axis="x", labelsize=6.4)
    axes[2].set_xlabel("Prompt variant")
    axes[2].set_ylabel("Primacy effect")
    axes[2].legend()
    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        _panel_label(ax, label)
    sources = [PHASE7, SINK, QUERY, TEMPLATE, MAMBA_PROBE, PYTHIA_PROBE]
    return _save(fig, output, FIGURES[6], sources)


def _phase8(output: Path) -> list[Path]:
    summary = _read(PHASE8)
    models = ["nemotron-h-8b", "llama-3.1-8b", "qwen2.5-7b"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    _position_curve(
        ax,
        summary["position_curve"],
        models,
        line_styles={"qwen2.5-7b": "--"},
    )
    return _save(fig, output, FIGURES[7], [PHASE8])


GENERATORS: tuple[Callable[[Path], list[Path]], ...] = (
    _phase2,
    _phase3,
    _phase4,
    _phase5,
    _phase1,
    _phase6,
    _phase7,
    _phase8,
)


def generate_all(output: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Generate every paper figure into an exact, caller-selected directory."""
    with plt.rc_context(PAPER_STYLE):
        return [path for generate in GENERATORS for path in generate(output)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in generate_all(args.output_dir.resolve()):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
