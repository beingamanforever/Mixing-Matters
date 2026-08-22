"""Shape the committed phase summaries into the single JSON the project page reads.

The per-phase summary files grew independently and nest their numbers
differently. The page should not carry that history, so this module flattens
them once, at build time, into flat records the browser can render directly.
Every number here is copied from a committed summary; nothing is recomputed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import MODELS
from .release import MIXERS

SUMMARIES: dict[str, str] = {
    "phase1": "artifacts/phase1/figures/figures-summary.json",
    "phase2": "artifacts/phase2/report/phase2-summary.json",
    "phase3": "artifacts/phase3/report/phase3-summary.json",
    "phase4": "artifacts/phase4/report/phase4-summary.json",
    "phase5": "artifacts/phase5/report/phase5-summary.json",
    "phase8": "artifacts/phase8/report/phase8-summary.json",
}

LABELS: dict[str, str] = {
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
    "mamba-2.8b-slimpj": "Mamba 2.8B (SlimPajama)",
    "mamba2-2.7b": "Mamba-2 2.7B",
    "mamba2-8b": "Mamba-2 8B (pure)",
    "mamba2-hybrid-8b": "Mamba-2 8B (hybrid)",
    "nemotron-h-8b": "Nemotron-H 8B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "qwen2.5-7b": "Qwen2.5 7B",
}

# Panels the page renders as position curves, in reading order.
CURVE_PANELS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "phase2",
        "Exploratory 2.8B family comparison",
        (
            "Pythia, Mamba, and Mamba-2 at similar scale on the same 800 questions; "
            "depth, positional encoding, state size, checkpoint family, and capability differ."
        ),
        ("pythia-2.8b", "mamba-2.8b", "mamba2-2.7b"),
    ),
    (
        "phase3",
        "Tightly matched 8B checkpoint contrast",
        (
            "Two NVIDIA checkpoints that share training data, scale, depth, and tokenizer; "
            "the hybrid changes both attention and MLP composition."
        ),
        ("mamba2-8b", "mamba2-hybrid-8b"),
    ),
    (
        "phase5",
        "Pretraining corpus control",
        "The same Mamba 2.8B architecture trained on the Pile and on SlimPajama.",
        ("mamba-2.8b", "mamba-2.8b-slimpj"),
    ),
    (
        "phase8",
        "Production systems",
        "Three complete 7-8B systems that differ on many axes at once.",
        ("nemotron-h-8b", "llama-3.1-8b", "qwen2.5-7b"),
    ),
)


def _read(root: Path, key: str) -> dict[str, Any]:
    return json.loads((root / SUMMARIES[key]).read_text())


def _effect(point: Mapping[str, Any]) -> dict[str, float]:
    return {
        "estimate": point["estimate"],
        "ci": [point["ci_low"], point["ci_high"]],
        "p": point["p_value_holm"],
    }


def _edges(point: Mapping[str, Any]) -> dict[str, Any]:
    return {"primacy": _effect(point["primacy"]), "recency": _effect(point["recency"])}


def _curve(entry: Mapping[str, Any]) -> list[dict[str, float]]:
    positions = entry["positions"]
    return [
        {
            "position": index + 1,
            "accuracy": positions[str(index)]["accuracy"],
            "ci": [positions[str(index)]["ci_low"], positions[str(index)]["ci_high"]],
        }
        for index in sorted(int(key) for key in positions)
    ]


def _model_entry(key: str) -> dict[str, Any]:
    spec = MODELS[key]
    return {
        "label": LABELS[key],
        "mixer": MIXERS[spec.family],
        "family": spec.family,
        "repo": spec.repo,
        "revision": spec.revision,
        "params_millions": spec.params_millions,
        "corpus": spec.training_corpus,
    }


def _phase3_edges(root: Path) -> dict[str, Any]:
    control = _read(root, "phase3")["attention_control"]
    return {
        control["pure_model"]: _edges(control["pure_edges"]),
        control["hybrid_model"]: _edges(control["hybrid_edges"]),
    }


def _phase5_edges(root: Path) -> dict[str, Any]:
    control = _read(root, "phase5")["data_control"]
    return {
        control["pile_model"]: _edges(control["pile_edges"]),
        control["slimpajama_model"]: _edges(control["slimpajama_edges"]),
    }


def _panel_edges(root: Path, key: str, models: tuple[str, ...]) -> dict[str, Any]:
    """Per-model edges, which Phase 3 and Phase 5 nest under their control block."""
    if key == "phase3":
        return _phase3_edges(root)
    if key == "phase5":
        return _phase5_edges(root)
    edges = _read(root, key)["edges"]
    return {model: _edges(edges[model]) for model in models}


def _panels(root: Path) -> list[dict[str, Any]]:
    panels = []
    for key, title, caption, models in CURVE_PANELS:
        summary = _read(root, key)
        panels.append(
            {
                "id": key,
                "title": title,
                "caption": caption,
                "models": list(models),
                "curves": {model: _curve(summary["position_curve"][model]) for model in models},
                "floor_ceiling": {model: summary["floor_ceiling"][model] for model in models},
                "edges": _panel_edges(root, key, models),
            }
        )
    return panels


def _contrasts(root: Path) -> list[dict[str, Any]]:
    """Every paired between-model contrast the paper reports, in one flat table."""
    rows = []
    for interaction in _read(root, "phase2")["interactions"]:
        rows.append(
            {
                "phase": "phase2",
                "label": f"{LABELS[interaction['first_model']]} minus "
                f"{LABELS[interaction['second_model']]}",
                "kind": "family",
                **_edges(interaction),
            }
        )
    phase3 = _read(root, "phase3")["attention_control"]
    rows.append(
        {
            "phase": "phase3",
            "label": "Hybrid minus pure Mamba-2 8B",
            "kind": "checkpoint",
            "primacy": _effect(phase3["primacy_diff"]),
            "recency": _effect(phase3["recency_diff"]),
        }
    )
    phase5 = _read(root, "phase5")["data_control"]
    rows.append(
        {
            "phase": "phase5",
            "label": "Pile minus SlimPajama (Mamba 2.8B)",
            "kind": "corpus",
            "primacy": _effect(phase5["primacy_diff"]),
            "recency": _effect(phase5["recency_diff"]),
        }
    )
    for pair in _read(root, "phase4")["scale_trend"]["pairs"]:
        rows.append(
            {
                "phase": "phase4",
                "label": f"{LABELS[pair['pythia_model']]} minus {LABELS[pair['mamba_model']]}",
                "kind": "scale",
                "primacy": _effect(pair["primacy_diff"]),
                "recency": _effect(pair["recency_diff"]),
            }
        )
    for interaction in _read(root, "phase8")["interactions"]:
        rows.append(
            {
                "phase": "phase8",
                "label": f"{LABELS[interaction['first_model']]} minus "
                f"{LABELS[interaction['second_model']]}",
                "kind": "production",
                **_edges(interaction),
            }
        )
    return rows


def _scale(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "pair": pair["pair"],
            "params_millions": (pair["pythia_params_millions"] + pair["mamba_params_millions"]) / 2,
            "attention": {
                "model": pair["pythia_model"],
                "primacy": _effect(pair["pythia_edges"]["primacy"]),
            },
            "state_space": {
                "model": pair["mamba_model"],
                "primacy": _effect(pair["mamba_edges"]["primacy"]),
            },
            "difference": _effect(pair["primacy_diff"]),
        }
        for pair in _read(root, "phase4")["scale_trend"]["pairs"]
    ]


def _calibration(root: Path) -> dict[str, Any]:
    summary = _read(root, "phase1")
    calibration = summary["phase1_condition_accuracy"]
    slots = summary["kv_position_curve"]["slots"]
    return {
        "conditions": [
            {
                "label": label,
                "accuracy": calibration["conditions"][key]["primary"]["accuracy"],
                "ci": calibration["conditions"][key]["primary"]["ci"],
            }
            for key, label in (
                ("closed_book", "Closed book"),
                ("gold_middle", "Gold middle"),
                ("gold_first", "Gold first"),
                ("oracle", "Oracle"),
            )
        ],
        "kv_curve": [
            {"slot": index + 1, "accuracy": slots[str(index)]["accuracy"]}
            for index in sorted(int(key) for key in slots)
        ],
    }


def build_site_data(root: Path) -> dict[str, Any]:
    """Assemble every number the project page renders."""
    panels = _panels(root)
    models = sorted(LABELS)
    return {
        "models": {model: _model_entry(model) for model in models},
        "panels": panels,
        "contrasts": _contrasts(root),
        "scale": _scale(root),
        "calibration": _calibration(root),
    }


def write_site_data(root: Path, output: Path) -> Path:
    """Write the project-page data file and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_site_data(root), indent=1, sort_keys=True)
    output.write_text(payload + "\n")
    return output


def main() -> None:
    """Entry point for ``python -m mixing_matters.site``.

    This module reads only committed summaries and the standard library, so the
    Pages workflow can build the page without installing the project. Going
    through ``cli`` instead would import ``figures`` and require matplotlib.
    """
    parser = argparse.ArgumentParser(description="Build the project-page data file.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("web") / "data" / "results.json")
    args = parser.parse_args()
    print(write_site_data(args.root, args.output))


if __name__ == "__main__":
    main()
