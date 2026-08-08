"""Phase 8 descriptive system comparison.

Phase 8 places three complete production 7-8B systems next to each other
on the Lost-in-the-Middle 10-document task. The three systems differ from
each other on many axes at once (architecture, pretraining corpus, token
count, tokenizer, alignment status, depth, and positional encoding), so
this phase is descriptive rather than a matched control: it reports how
full-system position curves look side by side, not which single variable
is responsible for any curve difference.

The statistics reuse the Phase 2 primitives unchanged. Each model is
treated as its own unit, per-position accuracy and primacy/recency edges
are estimated with the same paired bootstrap over question bundles, and
pairwise interactions are reported for every pair of the three systems
so a reader can locate curve differences without a single confounded
"attention effect" interpretation.
"""

import itertools
from collections.abc import Iterable

from .models import MODELS, PHASE8_SYSTEMS
from .phase2 import DEFAULT_RESAMPLES, edges, interaction, position_curve


def _floor_ceiling_means(records: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Mean floor and ceiling per model, one value per question."""
    per_model_question: dict[str, dict[str, tuple[float, float]]] = {}
    for record in records:
        if record.get("condition") != "gold":
            continue
        bucket = per_model_question.setdefault(record["model_key"], {})
        bucket.setdefault(
            record["question_id"], (record["floor_accuracy"], record["ceiling_accuracy"])
        )

    means: dict[str, dict[str, float]] = {}
    for model, questions in per_model_question.items():
        floors = [floor for floor, _ in questions.values()]
        ceilings = [ceiling for _, ceiling in questions.values()]
        means[model] = {
            "floor_accuracy": sum(floors) / len(floors),
            "ceiling_accuracy": sum(ceilings) / len(ceilings),
        }
    return means


def _model_descriptor(model_key: str) -> dict:
    """A stable, human-readable descriptor of the axes this model differs on.

    Phase 8 is a descriptive comparison, so figures and summaries carry the
    axes that are known to move across the group. The values come straight
    from the registry and the model's own tokenizer.
    """
    spec = MODELS[model_key]
    return {
        "model_key": spec.key,
        "repo": spec.repo,
        "family": spec.family,
        "training_corpus": spec.training_corpus,
    }


def phase8_summary(
    records: list[dict],
    n_resamples: int = DEFAULT_RESAMPLES,
    systems: tuple[str, ...] = PHASE8_SYSTEMS,
) -> dict:
    """Descriptive summary of the three Phase 8 systems.

    Returns the per-model position curve, per-model edges, every pairwise
    interaction, mean floor/ceiling per model, and a small descriptor per
    model that records which registry axes are known to move across the
    group.
    """
    curve = position_curve(records, n_resamples=n_resamples)
    edge = edges(records, n_resamples=n_resamples)
    observed = sorted(edge)
    # Preserve the registry order for the Phase 8 systems that were run;
    # append any other model keys present so unexpected inclusions surface.
    ordered = [key for key in systems if key in observed]
    extra = [key for key in observed if key not in systems]
    models = ordered + extra

    interactions = [
        {
            "first_model": first_model,
            "second_model": second_model,
            **interaction(records, first_model, second_model, n_resamples=n_resamples),
        }
        for first_model, second_model in itertools.combinations(models, 2)
    ]

    floor_ceiling = _floor_ceiling_means(records)
    descriptors = {model: _model_descriptor(model) for model in models}

    return {
        "models": models,
        "n_resamples": n_resamples,
        "position_curve": curve,
        "edges": edge,
        "interactions": interactions,
        "floor_ceiling": floor_ceiling,
        "system_descriptors": descriptors,
    }
