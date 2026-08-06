from lost_in_the_middle.metrics import best_subspan_em, normalize_answer


def _first_line(generation: str) -> str:
    for line in generation.splitlines():
        if line.strip():
            return line
    return ""


def score_variants(generation: str, answers: list[str]) -> dict[str, float]:
    """Score a generation against the gold answers under three related metrics.

    - score: the primary metric, best-subspan exact match over the whole generation.
    - score_normalized_em: exact match of the extracted answer after normalization.
    - score_first_line: best-subspan exact match restricted to the first non-empty line.

    Normalized exact match is computed on the first non-empty line rather than the
    whole generation. A base model keeps generating past its answer, so whole
    generation exact match was 0.000 in all 800 Phase 1 generations, including the
    oracle condition that scored 0.650 under the primary metric, which makes it
    useless as a scoring sensitivity check.
    """
    normalized_generation = normalize_answer(_first_line(generation))
    normalized_em = any(normalized_generation == normalize_answer(answer) for answer in answers)
    return {
        "score": float(best_subspan_em(generation, answers)),
        "score_normalized_em": float(normalized_em),
        "score_first_line": float(best_subspan_em(_first_line(generation), answers)),
    }
