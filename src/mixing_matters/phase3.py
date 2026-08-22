"""Phase 3 statistics for the released pure-versus-hybrid checkpoint contrast.

Phase 2 compares model families on the same question bundles. Phase 3 holds
training data, tokenizer, scale, depth, and positional encoding fixed across
released NVIDIA 8B checkpoints. The hybrid changes both attention and MLP
composition, so the paired edge difference is a composite checkpoint contrast,
not an attention-only effect. ``models.ARCH_PAIR`` orders the hybrid first, so
the estimate reads as hybrid minus pure on each edge.

Unlike Phase 5, Phase 3 is deliberately self-contained: it reports the
checkpoint contrast and each model's own edges over the shared questions. The
legacy function and output-key names are retained for artifact compatibility.
"""

from collections.abc import Iterable

from .models import ARCH_PAIR, MODELS
from .phase2 import DEFAULT_RESAMPLES, edges, interaction


def attention_interaction(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Hybrid-minus-pure difference in primacy and recency edges.

    A thin wrapper over ``phase2.interaction`` pinned to ``models.ARCH_PAIR``
    with the hybrid model as the first argument, so a positive ``estimate``
    means the released hybrid checkpoint has the larger edge.
    Both models are recomputed from the same bootstrap draw of question ids in
    every resample, so the contrast is paired over the questions both models
    answered completely.
    """
    hybrid_key, pure_key = ARCH_PAIR
    return interaction(records, hybrid_key, pure_key, n_resamples=n_resamples)


def attention_control(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """The checkpoint contrast plus each model's own edges over the shared questions.

    Returns::

        {
            "pair": str,
            "hybrid_model": str,
            "pure_model": str,
            "question_count": int,
            "primacy_diff": {"estimate": float, "ci_low": float, "ci_high": float,
                              "p_value": float, "p_value_holm": float},
            "recency_diff": {...},
            "hybrid_edges": {"primacy": {...}, "recency": {...}},
            "pure_edges": {"primacy": {...}, "recency": {...}},
        }

    ``hybrid_edges`` and ``pure_edges`` are computed on the two Phase 3 models
    only, so their question universe matches the paired contrast.
    """
    records = list(records)
    hybrid_key, pure_key = ARCH_PAIR

    contrast = attention_interaction(records, n_resamples=n_resamples)
    model_edges = edges(
        [record for record in records if record.get("model_key") in (hybrid_key, pure_key)],
        n_resamples=n_resamples,
    )

    return {
        "pair": MODELS[hybrid_key].arch_pair,
        "hybrid_model": hybrid_key,
        "pure_model": pure_key,
        "question_count": contrast["question_count"],
        "primacy_diff": contrast["primacy"],
        "recency_diff": contrast["recency"],
        "hybrid_edges": {
            "primacy": model_edges[hybrid_key]["primacy"],
            "recency": model_edges[hybrid_key]["recency"],
        },
        "pure_edges": {
            "primacy": model_edges[pure_key]["primacy"],
            "recency": model_edges[pure_key]["recency"],
        },
    }
