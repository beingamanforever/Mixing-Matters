"""Phase 7 4a query-position report over multi-variant sweep runs.

Each sweep run in Phase 7 sub-experiment 4a produces the same
records shape as Phase 2/5/8, except every gold record now carries a
``prompt_variant`` field indicating one of ``baseline``,
``question_first``, ``bookend``, or ``gold_padded``. This module
groups the sweep records by (model_key, prompt_variant), reruns the
per-model bootstrap edges independently in each cell, and reports the
per-variant per-model position curve and edge estimates.

The prediction from the fixed-state-compression hypothesis is that
Mamba and hybrid Mamba+attention models gain more from ``question_first``
and ``bookend`` (which surface the question tokens before the recurrent
state has compressed the documents) than the dense-attention Pythia
family. Recency should shrink under ``gold_padded`` where extra tokens
sit between the gold document and the end of the prompt.
"""

from collections.abc import Iterable

from .phase2 import DEFAULT_RESAMPLES, edges, position_curve

VARIANTS = ("baseline", "question_first", "bookend", "gold_padded")


def _variant_of(record: dict) -> str:
    """Return the composite variant label recorded on ``record``.

    Older Phase 2/5 sweeps do not have a ``prompt_variant`` field; treat
    those as ``baseline`` so a caller can combine old Phase 2 sweeps with
    new Phase 7-4a variant sweeps in a single call. When a record carries
    ``sink_block=True`` the label is suffixed with ``+sink_block`` so a
    sink-blocked sweep and its baseline run under the same prompt variant
    can share a call without colliding on (model_key, prompt_variant).
    """
    label = record.get("prompt_variant") or "baseline"
    if record.get("sink_block"):
        label = f"{label}+sink_block"
    template = record.get("prompt_template")
    # A non-default instruction template is its own cell: a 4e template run
    # keeps prompt_variant=baseline, so without folding the template in it
    # would collide with the Phase 2 baseline and with the other templates.
    if template and template != "liu":
        label = f"{label}+tmpl:{template}"
    return label


def _tag_records(records: Iterable[dict]) -> list[dict]:
    """Copy records with model_key rewritten to ``"model_key::variant"``.

    ``phase2.edges`` groups by ``model_key`` so the simplest way to reuse
    it per variant is to tag the model with the variant name. The record
    remains eligible for the shared paired bootstrap; the variant lives
    in the composite key.
    """
    tagged = []
    for record in records:
        copy = dict(record)
        model_key = record.get("model_key")
        variant = _variant_of(record)
        if model_key is None:
            raise ValueError("record missing model_key")
        copy["model_key"] = f"{model_key}::{variant}"
        tagged.append(copy)
    return tagged


def variant_edges(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Return the per-model per-variant primacy and recency edges.

    Output shape::

        {
            model_key: {
                variant: {
                    "primacy": {...},
                    "recency": {...},
                    "question_count": int,
                    "excluded_record_count": int,
                    "excluded_question_count": int,
                },
                ...
            },
            ...
        }
    """
    tagged = _tag_records(records)
    all_edges = edges(tagged, n_resamples=n_resamples)
    out: dict[str, dict[str, dict]] = {}
    for tagged_key, entry in all_edges.items():
        model_key, variant = tagged_key.split("::", 1)
        out.setdefault(model_key, {})[variant] = entry
    return out


def variant_curve(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Per-model per-variant position curve."""
    tagged = _tag_records(records)
    all_curve = position_curve(tagged, n_resamples=n_resamples)
    out: dict[str, dict[str, dict]] = {}
    for tagged_key, entry in all_curve.items():
        model_key, variant = tagged_key.split("::", 1)
        out.setdefault(model_key, {})[variant] = entry
    return out


def phase7_variant_summary(
    records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES
) -> dict:
    """Convenience wrapper joining curve and edges for a report."""
    records = list(records)
    return {
        "n_resamples": n_resamples,
        "edges": variant_edges(records, n_resamples=n_resamples),
        "position_curve": variant_curve(records, n_resamples=n_resamples),
    }
