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
    "phase6": "artifacts/phase6/report/phase6-summary.json",
    "phase7": "artifacts/phase7-mechanisms/report/phase7-summary.json",
    "phase8": "artifacts/phase8/report/phase8-summary.json",
    "sink": "artifacts/phase7-mechanisms/4c-sink-scan/report/sink-mass-summary.json",
    "query": "artifacts/phase7-mechanisms/4a-query-position/report/phase7-variants-summary.json",
    "template": "artifacts/phase7-mechanisms/4e-template/report/phase7-variants-summary.json",
    "probe_mamba": "artifacts/phase7-mechanisms/4d-probe/mamba-2.8b-layer32-probe.json",
    "probe_pythia": "artifacts/phase7-mechanisms/4d-probe/pythia-2.8b-layer16-probe.json",
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
        "Matched 2.8B comparison",
        "Pythia, Mamba, and Mamba-2 at the same scale on the same 800 questions.",
        ("pythia-2.8b", "mamba-2.8b", "mamba2-2.7b"),
    ),
    (
        "phase3",
        "Matched 8B pure vs hybrid",
        (
            "Two NVIDIA checkpoints that share training data and scale; the hybrid "
            "replaces about 7% of Mamba-2 blocks with attention."
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

PHASES: tuple[dict[str, str], ...] = (
    {
        "id": "phase1",
        "name": "Calibration",
        "question": "Can the harness detect a position effect that is known to exist?",
        "finding": "Yes. The key-value positive control moves 38 points between edge and "
        "middle slots, and the four QA calibration conditions order as designed.",
        "status": "instrument check",
    },
    {
        "id": "phase2",
        "name": "Matched 2.8B architectures",
        "question": "Do attention and state-space mixers use the same evidence differently?",
        "finding": "Pythia carries a +5.19 point primacy edge; both Mamba variants sit at or "
        "below zero. Both paired differences clear Holm correction at p < 0.0001.",
        "status": "primary result",
    },
    {
        "id": "phase3",
        "name": "Pure vs hybrid at 8B",
        "question": "Does adding attention blocks to a state-space model create primacy?",
        "finding": "The hybrid has a significant within-model primacy edge and the pure model "
        "does not, but the paired difference is +1.88 points with Holm p = 0.1442.",
        "status": "suggestive, not significant",
    },
    {
        "id": "phase4",
        "name": "Scale sweep",
        "question": "Is the family gap present at every model size?",
        "finding": "No. It is near zero at the two smallest pairs and appears from 790M vs 1B "
        "onward. Capability and architecture remain confounded.",
        "status": "scale-emergent",
    },
    {
        "id": "phase5",
        "name": "Pretraining corpus",
        "question": "Does swapping the pretraining corpus reproduce the effect?",
        "finding": "No. Pile vs SlimPajama moves the level of the curve by 16 points but the "
        "primacy interaction is -0.69 points with Holm p = 0.568.",
        "status": "null shape change",
    },
    {
        "id": "phase6",
        "name": "Synthetic retrieval",
        "question": "Does the effect transfer off multi-document QA?",
        "finding": "Partly. Pythia reproduces a +12 point primacy edge on RULER at 2K tokens; "
        "both Mamba models saturate at 1.0 and cannot be compared there.",
        "status": "one arm only",
    },
    {
        "id": "phase7",
        "name": "Mechanisms",
        "question": "What distinguishes the two families internally?",
        "finding": "Late-layer attention-sink mass tracks primacy across Pythia scale, and "
        "position stays linearly decodable in both families, favouring a "
        "utilisation gap over a storage gap.",
        "status": "correlational",
    },
    {
        "id": "phase8",
        "name": "Production systems",
        "question": "Do deployed 7-8B systems show the same shape?",
        "finding": "All three show positive primacy edges, but their many differences make "
        "this a prevalence description rather than an architecture test.",
        "status": "descriptive",
    },
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
                "kind": "architecture",
                **_edges(interaction),
            }
        )
    phase3 = _read(root, "phase3")["attention_control"]
    rows.append(
        {
            "phase": "phase3",
            "label": "Hybrid minus pure Mamba-2 8B",
            "kind": "attention",
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


def _final_sink_mass(entry: Mapping[str, Any]) -> float:
    layers = [position["mean_sink_mass_per_layer"][-1] for position in entry["positions"].values()]
    return sum(layers) / len(layers)


def _mechanisms(root: Path) -> dict[str, Any]:
    sink = _read(root, "sink")["by_model"]
    depth = {entry["model_key"]: entry for entry in _read(root, "phase7")["depth_trend"]["models"]}
    query = _read(root, "query")
    template = _read(root, "template")
    variants = (
        ("Liu baseline", "baseline", query),
        ("Bookend", "bookend", query),
        ("Gold padded", "gold_padded", query),
        ("Question first", "question_first", query),
        ("Concise template", "baseline+tmpl:concise", template),
        ("Instructional template", "baseline+tmpl:instructional", template),
    )
    return {
        "sink": [
            {
                "model": model,
                "label": LABELS[model],
                "final_layer_sink_mass": _final_sink_mass(sink[model]),
                "primacy": _effect(depth[model]["primacy"]),
            }
            for model in sorted(sink, key=lambda key: MODELS[key].params_millions)
        ],
        "probe": [
            {
                "model": model,
                "label": LABELS[model],
                "layer": probe["layer"],
                "accuracy": probe["accuracy"],
                "shuffled_accuracy": probe["shuffled_accuracy"],
            }
            for model, probe in (
                ("mamba-2.8b", _read(root, "probe_mamba")),
                ("pythia-2.8b", _read(root, "probe_pythia")),
            )
        ],
        "variants": [
            {
                "label": label,
                "primacy": {
                    model: _effect(source["edges"][model][key]["primacy"])
                    for model in ("pythia-2.8b", "mamba-2.8b")
                },
            }
            for label, key, source in variants
        ],
    }


def _mean_accuracy(entry: Mapping[str, Any]) -> float:
    positions = entry["positions"].values()
    return sum(point["accuracy"] for point in positions) / len(positions)


def _task_transfer(root: Path) -> dict[str, Any]:
    summary = _read(root, "phase6")
    needle = summary["lengths"]["2048"]
    models = ("pythia-2.8b", "mamba-2.8b", "mamba2-2.7b")
    return {
        "task": summary["task"],
        "rows": [
            {
                "model": model,
                "label": LABELS[model],
                "qa": _edges(summary["qa_edges"][model]),
                "needle": _edges(needle["edges"][model]),
                # Mean over the ten needle depths, which is what shows the
                # state-space models saturating rather than being flat.
                "needle_accuracy": _mean_accuracy(needle["position_curve"][model]),
            }
            for model in models
        ],
    }


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
        "design": {
            "questions_total": 2655,
            "exploratory": 800,
            "confirmatory_held_out": 1855,
            "positions": 10,
            "resamples": 10000,
            "split_seed": 240521,
            "dataset_sha256": ("192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9"),
        },
        "models": {model: _model_entry(model) for model in models},
        "phases": list(PHASES),
        "panels": panels,
        "contrasts": _contrasts(root),
        "scale": _scale(root),
        "mechanisms": _mechanisms(root),
        "task_transfer": _task_transfer(root),
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
