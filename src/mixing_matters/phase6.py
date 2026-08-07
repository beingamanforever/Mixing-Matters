"""Phase 6 statistics: the RULER niah_single_1 generality check.

Phase 6 reuses the Phase 2 machinery one length at a time. A niah sweep record
looks exactly like a QA sweep record - condition ``gold`` with gold_position 0
through 9, plus ``closed_book`` and ``oracle`` anchors - so the position curve,
edge contrasts, and paired-bootstrap interactions are the Phase 2 functions
applied to the records of a single context length. Grouping by
``context_length`` first is required: merging lengths would put two ``gold``
records at the same position for one instance and be rejected as a duplicate.

The comparison of interest is between tasks. Phase 2 measures the position
curve on multi-document QA; Phase 6 measures it on synthetic needle retrieval.
``task_comparison`` places each model's niah edges beside its QA edges so a
disagreement between the two tasks is reported as such rather than being read
as one task failing.
"""

from collections.abc import Iterable

from .phase2 import DEFAULT_RESAMPLES, edges, interaction, position_curve


def _by_length(records: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        length = record.get("context_length")
        if length is None:
            raise ValueError("niah record is missing context_length")
        grouped.setdefault(int(length), []).append(record)
    if not grouped:
        raise ValueError("no niah records found")
    return grouped


def _floor_ceiling_means(records: list[dict]) -> dict[str, dict[str, float]]:
    """Mean floor and ceiling accuracy per model, one value per question."""
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


def length_summary(records: list[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Curve, edges, pairwise interactions, and anchors for one length's records."""
    import itertools

    curve = position_curve(records, n_resamples=n_resamples)
    edge = edges(records, n_resamples=n_resamples)
    models = sorted(edge)
    interactions = [
        {
            "first_model": first,
            "second_model": second,
            **interaction(records, first, second, n_resamples=n_resamples),
        }
        for first, second in itertools.combinations(models, 2)
    ]
    return {
        "models": models,
        "position_curve": curve,
        "edges": edge,
        "interactions": interactions,
        "floor_ceiling": _floor_ceiling_means(records),
    }


def task_comparison(
    niah_edges: dict, qa_edges: dict, edge_name: str
) -> list[dict]:
    """Per model shared by both tasks, its niah edge beside its QA edge.

    ``agree`` is true when the two edges have the same sign (both point the same
    way relative to the center positions) or either is effectively zero. A
    disagreement is a genuine cross-task finding, not an error.
    """
    shared = sorted(set(niah_edges) & set(qa_edges))
    rows = []
    for model in shared:
        niah = niah_edges[model][edge_name]["estimate"]
        qa = qa_edges[model][edge_name]["estimate"]
        rows.append(
            {
                "model": model,
                "edge": edge_name,
                "niah_estimate": niah,
                "qa_estimate": qa,
                "difference": niah - qa,
                "agree": (niah >= 0) == (qa >= 0),
            }
        )
    return rows


def phase6_summary(
    records: list[dict],
    qa_records: list[dict] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """Per-length niah summaries, and a cross-task comparison against QA edges.

    When ``qa_records`` (the Phase 2 sweep) is given, the QA edges are computed
    once and each niah length's edges are compared against them under
    ``task_comparison`` for both the primacy and recency edges.
    """
    by_length = _by_length(records)
    lengths = {
        length: length_summary(group, n_resamples=n_resamples)
        for length, group in sorted(by_length.items())
    }

    summary = {
        "task": "niah_single_1",
        "n_resamples": n_resamples,
        "lengths": lengths,
    }

    if qa_records is not None:
        qa_edges = edges(qa_records, n_resamples=n_resamples)
        summary["qa_edges"] = qa_edges
        summary["task_comparison"] = {
            str(length): {
                "primacy": task_comparison(lengths[length]["edges"], qa_edges, "primacy"),
                "recency": task_comparison(lengths[length]["edges"], qa_edges, "recency"),
            }
            for length in lengths
        }

    return summary
