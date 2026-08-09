"""Phase 7 mechanism analyses over already-collected sweep data.

Phase 7 tests why the primacy and recency arms appear where they do.
The sub-experiments in this module are the ones that need no additional
GPU generations: they aggregate existing Phase 2 and Phase 4 sweep
records under different lenses. The GPU-bound sub-experiments live in
their own modules (query-position ablation, sink-mass measurement,
sink-blocking, linear probes) once their sweeps have been produced.

Three lenses live here:

- ``depth_trend`` reads per-model primacy and recency edges from a set
  of registered models and joins them to their layer counts. It answers
  the ``P410m vs P1b`` depth-width contrast in the Phase 7 spec.
- ``scoring_sensitivity`` recomputes edges under the two secondary
  scores (``score_normalized_em`` and ``score_first_line``) already
  recorded on every sweep record, so the primary claim can be checked
  against alternative answer extraction and normalization.
- ``length_sensitivity`` bins questions by their prompt token count and
  recomputes edges within each bin, so a possible confound between
  prompt length and edge magnitude is bounded.

Every analysis uses the same paired bootstrap over questions defined by
``phase2.edges`` / ``phase2.position_curve``, so its confidence
intervals are directly comparable to the earlier phases.
"""

from collections.abc import Iterable
from statistics import median

from .phase2 import DEFAULT_RESAMPLES, edges, position_curve

# Layer counts per model_key for the depth analysis. Numbers come from
# the model configurations of the pinned revisions in ``models.MODELS``
# (Pythia: GPT-NeoX-20B family; Mamba: state-spaces canonical layers).
# Ordered by ``params_millions`` to match the Phase 4 scale axis.
PYTHIA_LAYERS: dict[str, int] = {
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
    "pythia-2.8b": 32,
}

MAMBA_LAYERS: dict[str, int] = {
    "mamba-130m": 24,
    "mamba-370m": 48,
    "mamba-790m": 48,
    "mamba-1.4b": 48,
    "mamba-2.8b": 64,
    "mamba2-2.7b": 64,
}

MODEL_LAYERS: dict[str, int] = {**PYTHIA_LAYERS, **MAMBA_LAYERS}


def _score_field(record: dict, field: str) -> float | None:
    value = record.get(field)
    return None if value is None else float(value)


def _rescore_records(records: Iterable[dict], field: str) -> list[dict]:
    """Return copies of ``records`` with ``score`` replaced by ``field``.

    ``phase2.edges`` and ``phase2.position_curve`` read the ``score``
    field. Rewriting the field in place would corrupt the raw record, so
    each record is copied first.
    """
    rescored = []
    for record in records:
        copy = dict(record)
        copy["score"] = _score_field(record, field)
        rescored.append(copy)
    return rescored


def depth_trend(
    records: Iterable[dict],
    n_resamples: int = DEFAULT_RESAMPLES,
    layer_counts: dict[str, int] = MODEL_LAYERS,
) -> dict:
    """Per-model primacy and recency edges joined to depth.

    Reads a stream of gold-condition records from one or more model
    sweeps, computes per-model edges through the shared Phase 2
    bootstrap, and joins the layer count for every model key that has
    an entry in ``layer_counts``. Returns::

        {
            "n_resamples": int,
            "models": [
                {
                    "model_key": str,
                    "family": str,          # from the model key prefix
                    "layers": int,
                    "primacy": {...},
                    "recency": {...},
                    "question_count": int,
                }
                ...
            ],
        }

    Models present in the records but missing from ``layer_counts``
    are skipped rather than silently defaulted, so an unregistered
    model does not corrupt the trend.
    """
    edge = edges(records, n_resamples=n_resamples)
    trend = []
    for model_key, entry in edge.items():
        if model_key not in layer_counts:
            continue
        family = model_key.split("-", 1)[0]
        trend.append(
            {
                "model_key": model_key,
                "family": family,
                "layers": layer_counts[model_key],
                "primacy": entry["primacy"],
                "recency": entry["recency"],
                "question_count": entry["question_count"],
            }
        )
    # Order by layer count so figures and tables read left to right by depth.
    trend.sort(key=lambda entry: (entry["family"], entry["layers"], entry["model_key"]))
    return {"n_resamples": n_resamples, "models": trend}


def depth_width_contrast(trend: dict, first_key: str, second_key: str) -> dict | None:
    """Return the (primacy, recency) edges of the two models named.

    Convenience wrapper for the ``pythia-410m`` vs ``pythia-1b`` contrast
    the Phase 7 spec calls out: the 410m has 24 layers on 1024 wide, the
    1b has 16 layers on 2048 wide, so this is the one within-family pair
    where depth and width move in opposite directions.
    """
    lookup = {entry["model_key"]: entry for entry in trend["models"]}
    if first_key not in lookup or second_key not in lookup:
        return None
    return {"first": lookup[first_key], "second": lookup[second_key]}


def scoring_sensitivity(
    records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES
) -> dict:
    """Recompute per-model edges under three scoring variants.

    Uses the already-computed alternative scores every sweep writes:
    ``score`` (primary, best_subspan_em), ``score_normalized_em`` (Liu
    et al. normalized exact match), and ``score_first_line`` (first-line
    extraction). Returns per-model edges under each score, so a reader
    can check whether the sign or magnitude of the primacy and recency
    arms depends on the scorer.
    """
    variants = ("score", "score_normalized_em", "score_first_line")
    records = list(records)
    out: dict[str, dict] = {}
    for variant in variants:
        rescored = _rescore_records(records, variant)
        try:
            variant_edges = edges(rescored, n_resamples=n_resamples)
        except ValueError as error:
            out[variant] = {"error": str(error)}
            continue
        out[variant] = variant_edges
    return {"n_resamples": n_resamples, "variants": out}


def _prompt_length_bins(records: list[dict], n_bins: int) -> list[tuple[float, float]]:
    """Return closed bin edges that split questions into ``n_bins`` groups.

    A question's prompt-token count is taken from the median of its gold
    records, so the ten positions of a question live in one bin.
    """
    per_question: dict[str, list[int]] = {}
    for record in records:
        if record.get("condition") != "gold":
            continue
        per_question.setdefault(record["question_id"], []).append(record["prompt_token_count"])
    per_question_median = sorted(median(counts) for counts in per_question.values())
    if not per_question_median:
        return []
    boundaries = []
    for index in range(1, n_bins):
        cut = per_question_median[int(len(per_question_median) * index / n_bins)]
        boundaries.append(cut)
    lo = per_question_median[0]
    hi = per_question_median[-1] + 1
    bins = []
    previous = lo
    for boundary in boundaries:
        bins.append((previous, boundary))
        previous = boundary
    bins.append((previous, hi))
    return bins


def length_sensitivity(
    records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES, n_bins: int = 3
) -> dict:
    """Per-model primacy and recency edges within prompt-length bins.

    Splits questions into ``n_bins`` groups by their gold-record
    prompt-token count median, so every group holds the same fraction of
    questions and each group's ten positions travel together. Runs the
    Phase 2 bootstrap independently inside each group. Returns::

        {
            "n_resamples": int,
            "n_bins": int,
            "bins": [
                {
                    "lower": int, "upper": int,
                    "question_count": int,
                    "edges": {model_key: {"primacy": {...}, "recency": {...}}, ...},
                }
                ...
            ],
        }

    ``lower`` is the closed lower bound of the bin (inclusive) and
    ``upper`` is exclusive. The bins are sorted left to right.
    """
    records = list(records)
    bins = _prompt_length_bins(records, n_bins)
    if not bins:
        return {"n_resamples": n_resamples, "n_bins": n_bins, "bins": []}

    per_question: dict[str, float] = {}
    for record in records:
        if record.get("condition") != "gold":
            continue
        per_question.setdefault(record["question_id"], []).append(record["prompt_token_count"])
    question_median = {qid: median(counts) for qid, counts in per_question.items()}

    def assign(qid: str) -> int:
        length = question_median[qid]
        for index, (lower, upper) in enumerate(bins):
            if lower <= length < upper:
                return index
        return len(bins) - 1

    grouped: list[list[dict]] = [[] for _ in bins]
    for record in records:
        if record.get("condition") != "gold":
            continue
        grouped[assign(record["question_id"])].append(record)

    out_bins = []
    for (lower, upper), group in zip(bins, grouped):
        question_ids = {record["question_id"] for record in group}
        entry = {
            "lower": int(lower),
            "upper": int(upper),
            "question_count": len(question_ids),
            "edges": {},
        }
        if group:
            try:
                entry["edges"] = edges(group, n_resamples=n_resamples)
            except ValueError as error:
                entry["error"] = str(error)
        out_bins.append(entry)
    return {"n_resamples": n_resamples, "n_bins": n_bins, "bins": out_bins}


def phase7_summary(
    records: Iterable[dict],
    n_resamples: int = DEFAULT_RESAMPLES,
    layer_counts: dict[str, int] = MODEL_LAYERS,
    length_bins: int = 3,
) -> dict:
    """The three compute-free Phase 7 lenses joined into one summary.

    ``records`` is the union of every sweep the caller wants to include,
    tagged by model_key. Returns the position curve, the depth trend,
    the scoring-variant edges, and the length-bin edges. Downstream
    figures read this shape.
    """
    records = list(records)
    return {
        "n_resamples": n_resamples,
        "position_curve": position_curve(records, n_resamples=n_resamples),
        "depth_trend": depth_trend(records, n_resamples=n_resamples, layer_counts=layer_counts),
        "scoring_sensitivity": scoring_sensitivity(records, n_resamples=n_resamples),
        "length_sensitivity": length_sensitivity(records, n_resamples=n_resamples, n_bins=length_bins),
    }
