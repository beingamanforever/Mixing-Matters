"""Phase 4 statistics: does the Mamba/Pythia curve-shape gap move with scale.

Phase 2 answers, for one matched pair, whether architecture interacts with
evidence position. Phase 4 reuses that per-pair contrast (``phase2.edges``
and ``phase2.interaction``) across the five matched size points named in
``models.SCALE_PAIRS`` and asks whether the Pythia-minus-Mamba edge
difference grows, shrinks, or stays stable as parameter count increases.

Only the two size endpoints among the pairs present in the data are
compared. This is a descriptive read of a five-point trend, not a fitted
slope with its own inference: see ``trend_summary``.
"""

from collections.abc import Iterable

from .models import MODELS, SCALE_PAIRS
from .phase2 import DEFAULT_RESAMPLES, edges, interaction


def _pair_members() -> dict[str, dict[str, str]]:
    """Map each scale pair label to its pythia and mamba model keys."""
    members: dict[str, dict[str, str]] = {}
    for model in MODELS.values():
        if model.scale_pair is None:
            continue
        members.setdefault(model.scale_pair, {})[model.family] = model.key
    return members


def scale_trend(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Per matched size pair, the Pythia-minus-Mamba primacy and recency gap.

    Groups ``records`` by ``model_key``, finds each ``SCALE_PAIRS`` pair's
    pythia and mamba model keys from the registry, and for every pair present
    in the data computes ``phase2.interaction`` with pythia as the first
    model and mamba as the second, so ``estimate`` is pythia's edge minus
    mamba's edge. Each model's own edges (``phase2.edges``, computed on the
    pair's two models only, so the question universe matches the
    interaction) are included alongside the difference.

    Returns::

        {
            "pairs": [
                {
                    "pair": str,
                    "pythia_model": str,
                    "mamba_model": str,
                    "pythia_params_millions": int,
                    "mamba_params_millions": int,
                    "question_count": int,
                    "primacy_diff": {"estimate": float, "ci_low": float,
                                      "ci_high": float, "p_value": float,
                                      "p_value_holm": float},
                    "recency_diff": {...},
                    "pythia_edges": {"primacy": {...}, "recency": {...}},
                    "mamba_edges": {"primacy": {...}, "recency": {...}},
                }
                for pair in SCALE_PAIRS if both its models are in the data
            ],
            "missing_pairs": [pair for pair in SCALE_PAIRS not fully present],
        }

    ``pairs`` is ordered by increasing size, following ``SCALE_PAIRS``.
    """
    records = list(records)
    model_keys = {record["model_key"] for record in records}
    members = _pair_members()

    present_pairs: list[tuple[str, str, str]] = []
    missing_pairs: list[str] = []
    for pair in SCALE_PAIRS:
        pythia_key = members.get(pair, {}).get("pythia")
        mamba_key = members.get(pair, {}).get("mamba")
        if pythia_key in model_keys and mamba_key in model_keys:
            present_pairs.append((pair, pythia_key, mamba_key))
        else:
            missing_pairs.append(pair)

    if not present_pairs:
        raise ValueError("no scale pairs found in records")

    pairs_out = []
    for pair, pythia_key, mamba_key in present_pairs:
        pair_records = [
            record for record in records if record["model_key"] in (pythia_key, mamba_key)
        ]
        contrast = interaction(pair_records, pythia_key, mamba_key, n_resamples=n_resamples)
        model_edges = edges(pair_records, n_resamples=n_resamples)
        pairs_out.append(
            {
                "pair": pair,
                "pythia_model": pythia_key,
                "mamba_model": mamba_key,
                "pythia_params_millions": MODELS[pythia_key].params_millions,
                "mamba_params_millions": MODELS[mamba_key].params_millions,
                "question_count": contrast["question_count"],
                "primacy_diff": contrast["primacy"],
                "recency_diff": contrast["recency"],
                "pythia_edges": {
                    "primacy": model_edges[pythia_key]["primacy"],
                    "recency": model_edges[pythia_key]["recency"],
                },
                "mamba_edges": {
                    "primacy": model_edges[mamba_key]["primacy"],
                    "recency": model_edges[mamba_key]["recency"],
                },
            }
        )

    return {"pairs": pairs_out, "missing_pairs": missing_pairs}


def _intervals_overlap(first: dict, second: dict) -> bool:
    return first["ci_low"] <= second["ci_high"] and second["ci_low"] <= first["ci_high"]


def _endpoint_direction(smallest: dict, largest: dict) -> str:
    if _intervals_overlap(smallest, largest):
        return "stable"
    return "grows" if largest["estimate"] > smallest["estimate"] else "shrinks"


def _endpoint_summary(pairs: list[dict], field: str) -> dict:
    smallest, largest = pairs[0][field], pairs[-1][field]
    return {
        "smallest_pair": pairs[0]["pair"],
        "largest_pair": pairs[-1]["pair"],
        "smallest_estimate": smallest["estimate"],
        "largest_estimate": largest["estimate"],
        "change": largest["estimate"] - smallest["estimate"],
        "direction": _endpoint_direction(smallest, largest),
    }


def trend_summary(scale_trend_result: dict) -> dict:
    """Describe how the Pythia-minus-Mamba edge gap moves across size.

    This compares only the smallest and largest size points among the pairs
    present in ``scale_trend_result``; it is a descriptive contrast of two
    endpoints across five size points, not a fitted slope with its own
    inference. The direction is "grows" or "shrinks" only when the two
    endpoints' 95 percent bootstrap intervals do not overlap; whenever they
    overlap, the conservative label "stable" is used instead.

    Returns::

        {
            "primacy": {"smallest_pair": str, "largest_pair": str,
                        "smallest_estimate": float, "largest_estimate": float,
                        "change": float, "direction": str},
            "recency": {...},
        }
    """
    pairs = scale_trend_result["pairs"]
    if len(pairs) < 2:
        raise ValueError("need at least two scale pairs present to describe a trend")

    return {
        "primacy": _endpoint_summary(pairs, "primacy_diff"),
        "recency": _endpoint_summary(pairs, "recency_diff"),
    }
