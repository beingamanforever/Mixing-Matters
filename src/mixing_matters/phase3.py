"""Phase 3 statistics: does adding attention layers move the position curve.

Phase 2 asks whether the sequence-mixing architecture interacts with evidence
position. Phase 3 holds almost everything fixed -- the same NVIDIA 8B Mamba-2
training run, the same tokenizer, the same scale and depth, the same
positional-encoding setup -- and changes only whether attention layers are
present, comparing the pure Mamba-2 (``mamba2-8b``) against the hybrid that
mixes in roughly seven percent attention layers (``mamba2-hybrid-8b``). The
contrast is therefore the same paired edge-difference bootstrap Phase 2 uses
(``phase2.interaction``), applied to the two models named in
``models.ARCH_PAIR`` with the hybrid model first, so the estimate reads as the
hybrid-minus-pure effect of adding attention on each edge.

Unlike Phase 5, Phase 3 is deliberately self-contained: it reports the
attention effect and each model's own edges over the shared questions, and does
not place the effect beside another phase's contrast.
"""

from collections.abc import Iterable

from .models import ARCH_PAIR, MODELS
from .phase2 import DEFAULT_RESAMPLES, edges, interaction


def attention_interaction(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Hybrid-minus-pure difference in primacy and recency edges.

    A thin wrapper over ``phase2.interaction`` pinned to ``models.ARCH_PAIR``
    with the hybrid model as the first argument, so a positive ``estimate``
    means the hybrid has the larger edge, i.e. adding attention widened it.
    Both models are recomputed from the same bootstrap draw of question ids in
    every resample, so the contrast is paired over the questions both models
    answered completely.
    """
    hybrid_key, pure_key = ARCH_PAIR
    return interaction(records, hybrid_key, pure_key, n_resamples=n_resamples)


def attention_control(records: Iterable[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    """The attention contrast plus each model's own edges over the shared questions.

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
