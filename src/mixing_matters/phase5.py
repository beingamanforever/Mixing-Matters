"""Phase 5 statistics: does the training corpus move the position curve.

Phase 2 asks whether the sequence-mixing architecture interacts with evidence
position. Phase 5 holds the architecture fixed -- the same 2.8B Mamba -- and
changes only the pretraining corpus, from the Pile (``mamba-2.8b``) to
SlimPajama (``mamba-2.8b-slimpj``). The contrast is therefore the same paired
edge-difference bootstrap Phase 2 uses (``phase2.interaction``), applied to the
two models named in ``models.DATA_PAIR`` with the Pile model first, so the
estimate reads as the Pile-minus-SlimPajama corpus effect on each edge.

``compare_to_architecture`` then places that corpus effect beside the Phase 2
architecture effect (Pythia minus Mamba on the Pile) so the two can be read on
the same scale. Like Phase 4's trend labels, the comparison is deliberately
descriptive: it calls one effect larger than the other only when their 95
percent bootstrap intervals do not overlap, and "comparable" whenever they do.
"""

from collections.abc import Iterable

from .models import DATA_PAIR, MODELS
from .phase2 import DEFAULT_RESAMPLES, edges, interaction


def corpus_interaction(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Pile-minus-SlimPajama difference in primacy and recency edges.

    A thin wrapper over ``phase2.interaction`` pinned to ``models.DATA_PAIR``
    with the Pile model as the first argument, so a positive ``estimate`` means
    the Pile model has the larger edge. Both models are recomputed from the
    same bootstrap draw of question ids in every resample, so the contrast is
    paired over the questions both models answered completely.
    """
    pile_key, slimpj_key = DATA_PAIR
    return interaction(records, pile_key, slimpj_key, n_resamples=n_resamples)


def data_control(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """The corpus contrast plus each model's own edges over the shared questions.

    Returns::

        {
            "pair": str,
            "pile_model": str,
            "slimpajama_model": str,
            "question_count": int,
            "primacy_diff": {"estimate": float, "ci_low": float, "ci_high": float,
                              "p_value": float, "p_value_holm": float},
            "recency_diff": {...},
            "pile_edges": {"primacy": {...}, "recency": {...}},
            "slimpajama_edges": {"primacy": {...}, "recency": {...}},
        }

    ``pile_edges`` and ``slimpajama_edges`` are computed on the two Phase 5
    models only, so their question universe matches the paired contrast.
    """
    records = list(records)
    pile_key, slimpj_key = DATA_PAIR

    contrast = corpus_interaction(records, n_resamples=n_resamples)
    model_edges = edges(
        [record for record in records if record.get("model_key") in (pile_key, slimpj_key)],
        n_resamples=n_resamples,
    )

    return {
        "pair": MODELS[pile_key].data_pair,
        "pile_model": pile_key,
        "slimpajama_model": slimpj_key,
        "question_count": contrast["question_count"],
        "primacy_diff": contrast["primacy"],
        "recency_diff": contrast["recency"],
        "pile_edges": {
            "primacy": model_edges[pile_key]["primacy"],
            "recency": model_edges[pile_key]["recency"],
        },
        "slimpajama_edges": {
            "primacy": model_edges[slimpj_key]["primacy"],
            "recency": model_edges[slimpj_key]["recency"],
        },
    }


def _magnitude_interval(contrast: dict) -> tuple[float, float]:
    """The range of |effect| implied by a signed bootstrap interval.

    An interval that straddles zero admits a magnitude as small as zero; one
    that stays on one side of zero has its smaller endpoint's absolute value as
    the floor. This lets magnitudes be compared without assuming both effects
    share a sign, so an equal-magnitude pair of opposite sign reads as
    comparable rather than distinguishable.
    """
    low, high = contrast["ci_low"], contrast["ci_high"]
    magnitude_high = max(abs(low), abs(high))
    magnitude_low = 0.0 if low <= 0 <= high else min(abs(low), abs(high))
    return magnitude_low, magnitude_high


def _relative_size(corpus: dict, architecture: dict) -> str:
    """Whether the corpus effect is larger, smaller, or comparable in size.

    Compares the two effects on the magnitude of the edge difference, using
    their bootstrap intervals mapped onto |effect|. It calls one larger than
    the other only when those magnitude intervals do not overlap, and uses the
    conservative label "comparable" whenever they do, matching the way Phase 4
    refuses to name a direction across overlapping intervals. Because the two
    effects are estimated on different question universes with independent
    resamples, this is a descriptive size comparison, not a paired test of a
    difference.
    """
    corpus_low, corpus_high = _magnitude_interval(corpus)
    architecture_low, architecture_high = _magnitude_interval(architecture)
    if corpus_low > architecture_high:
        return "larger"
    if corpus_high < architecture_low:
        return "smaller"
    return "comparable"


def _edge_comparison(corpus: dict, architecture: dict) -> dict:
    return {
        "corpus_estimate": corpus["estimate"],
        "corpus_ci": [corpus["ci_low"], corpus["ci_high"]],
        "architecture_estimate": architecture["estimate"],
        "architecture_ci": [architecture["ci_low"], architecture["ci_high"]],
        "corpus_effect_vs_architecture": _relative_size(corpus, architecture),
    }


def compare_to_architecture(data_control_result: dict, architecture_interaction: dict) -> dict:
    """Place the Phase 5 corpus effect beside the Phase 2 architecture effect.

    ``architecture_interaction`` is a ``phase2.interaction`` result, expected to
    be the Pythia-minus-Mamba contrast on the Pile from Phase 2. The comparison
    is descriptive: per edge it reports both estimates and intervals and labels
    the corpus effect "larger", "smaller", or "comparable" relative to the
    architecture effect, using non-overlap of the two bootstrap intervals as
    the bar for calling them distinguishable.

    Returns ``{"primacy": {...}, "recency": {...}}``.
    """
    return {
        "primacy": _edge_comparison(
            data_control_result["primacy_diff"], architecture_interaction["primacy"]
        ),
        "recency": _edge_comparison(
            data_control_result["recency_diff"], architecture_interaction["recency"]
        ),
    }
