"""Phase 6 generality check: RULER ``niah_single_1`` needle retrieval.

Phase 2 measures the accuracy-versus-evidence-position curve on the Lost in
the Middle multi-document QA task. Phase 6 asks whether the same position
effect shows up on a synthetic needle-in-a-haystack task, where the "document"
is a single fact sentence hidden in repeated noise and there are no distractor
documents at all. If the two tasks disagree, that disagreement is the finding,
not a failure of either pipeline.

The task strings are vendored verbatim from NVIDIA RULER at the pinned commit
below (``niah_single_1``: noise haystack, one numeric needle, one word key):

  https://github.com/NVIDIA/RULER  commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
  scripts/data/synthetic/niah.py, scripts/data/synthetic/constants.py,
  scripts/synthetic.yaml, scripts/eval/synthetic/constants.py

Two deliberate, documented departures from running RULER's script directly, so
this fits the study's one-variable-at-a-time and reproducibility rules:

- The needle is placed at ten deterministic depths (0 through 9, mirroring the
  ten gold positions of the QA sweep) instead of RULER's random per-sample
  depth, so a position curve can be read off the same way as Phase 2. The noise
  sentence count is held fixed within a length, so total prompt length is
  invariant across the ten depths exactly as gold position is length-invariant
  in Phase 2.
- The key word is drawn from a small vendored word list rather than from the
  ``wonderwords`` package. The key is only a label the question echoes; it does
  not affect retrieval difficulty, and vendoring the list keeps the instances
  reproducible without depending on an external word corpus. The numeric needle
  and the noise haystack are exact.
"""

import random

from lost_in_the_middle.metrics import normalize_answer

RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"

# The niah_single_1 noise sentence, repeated to build the haystack.
NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

# The needle fact sentence. type_needle_v stays plural ("numbers") here, exactly
# as RULER builds the needle before it singularises the surrounding template.
NEEDLE = "One of the special magic numbers for {key} is: {value}."

# RULER's niah task template with its answer prefix appended, which is the full
# string fed to a base model. The single-needle rewrites below turn the plural
# wording singular the same way RULER does when num_needle_q * num_needle_v == 1.
_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the "
    "provided text? The special magic {type_needle_v} for {query} mentioned in the "
    "provided text are"
)

# The ten needle depths, mirroring the ten gold positions of the QA sweep.
DEPTHS = tuple(range(10))
NUMBER_DIGITS = 7
INSTANCE_SEED = 42
MIN_HAYSTACK = len(DEPTHS)

# A small fixed word list for the key label. The key does not affect retrieval;
# vendoring the list keeps instances reproducible without an external corpus.
_ADJECTIVES = (
    "brave", "calm", "eager", "fancy", "gentle", "happy", "jolly", "kind",
    "lively", "mighty", "noble", "proud", "quiet", "rapid", "shiny", "tidy",
    "vivid", "witty", "zesty", "ample",
)
_NOUNS = (
    "anchor", "beacon", "cabin", "dolphin", "ember", "falcon", "garden",
    "harbor", "island", "jacket", "kettle", "lantern", "meadow", "nectar",
    "orchard", "pebble", "quartz", "ribbon", "saddle", "temple",
)


def _single_needle_template() -> str:
    """RULER's rewrite of the template for a single needle and value."""
    template = _TEMPLATE
    template = template.replace("Some", "A")
    template = template.replace("are all", "is")
    template = template.replace("are", "is")
    template = template.replace("answers", "answer")
    return template


def make_instance(index: int, seed: int = INSTANCE_SEED) -> dict:
    """A deterministic needle: a word key and a fixed-width numeric value."""
    rng = random.Random(f"niah_single_1:{seed}:{index}")
    key = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"
    lower = 10 ** (NUMBER_DIGITS - 1)
    upper = 10**NUMBER_DIGITS - 1
    value = str(rng.randint(lower, upper))
    return {"index": index, "key": key, "value": value, "seed": seed}


def _needle_sentence(instance: dict) -> str:
    return NEEDLE.format(key=instance["key"], value=instance["value"])


def _insertion_index(depth: int, num_haystack: int) -> int:
    """Map depth 0-9 to an insertion slot among ``num_haystack`` noise sentences.

    Depth 0 puts the needle before all noise, depth 9 after all of it, so the
    needle sweeps front to back while the noise count, hence prompt length,
    stays fixed.
    """
    if depth not in DEPTHS:
        raise ValueError(f"depth must be between 0 and 9: {depth!r}")
    return round(depth / (len(DEPTHS) - 1) * num_haystack)


def build_gold_prompt(instance: dict, depth: int, num_haystack: int) -> str:
    """A haystack of ``num_haystack`` noise sentences with the needle at ``depth``."""
    sentences = [NOISE_SENTENCE] * num_haystack
    sentences.insert(_insertion_index(depth, num_haystack), _needle_sentence(instance))
    context = "\n".join(sentences)
    return _single_needle_template().format(
        type_needle_v="number", context=context, query=instance["key"]
    )


def build_floor_prompt(instance: dict, num_haystack: int) -> str:
    """The haystack with no needle: the closed-book floor for guessing."""
    context = "\n".join([NOISE_SENTENCE] * num_haystack)
    return _single_needle_template().format(
        type_needle_v="number", context=context, query=instance["key"]
    )


def build_ceiling_prompt(instance: dict) -> str:
    """The needle alone with no noise: the oracle ceiling for perfect retrieval."""
    return _single_needle_template().format(
        type_needle_v="number", context=_needle_sentence(instance), query=instance["key"]
    )


def solve_haystack_size(
    count_tokens, target_tokens: int, gen_tokens: int, reference: dict
) -> int:
    """Largest noise-sentence count whose gold prompt fits ``target_tokens``.

    ``count_tokens`` maps a string to its token count under the model's
    tokenizer. The reference needle is placed at the middle depth for
    measurement; the noise sentence is constant length so every depth of every
    instance lands within a couple of tokens of this budget, checked per
    instance by the runner.
    """
    if target_tokens - gen_tokens <= 0:
        raise ValueError(f"target_tokens {target_tokens} too small for gen_tokens {gen_tokens}")

    def fits(num_haystack: int) -> bool:
        prompt = build_gold_prompt(reference, len(DEPTHS) // 2, num_haystack)
        return count_tokens(prompt) + gen_tokens <= target_tokens

    if not fits(MIN_HAYSTACK):
        raise ValueError(
            f"target_tokens {target_tokens} cannot hold {MIN_HAYSTACK} noise sentences "
            "plus the needle and template"
        )

    lower = MIN_HAYSTACK
    upper = MIN_HAYSTACK
    while fits(upper * 2):
        upper *= 2
    upper *= 2
    best = lower
    while lower <= upper:
        mid = (lower + upper) // 2
        if fits(mid):
            best = mid
            lower = mid + 1
        else:
            upper = mid - 1
    return best


def _first_line(generation: str) -> str:
    for line in generation.splitlines():
        if line.strip():
            return line
    return ""


def score_variants(generation: str, answers: list[str]) -> dict[str, float]:
    """Score a needle generation, primary metric first.

    - score: RULER's official ``string_match_all`` for a single needle, a
      case-insensitive substring test of the magic number over the whole
      generation.
    - score_first_line: the same substring test restricted to the first
      non-empty line, a stricter extraction check.
    - score_normalized_em: normalized exact match of the first line against the
      value, the tightest variant.
    """
    lowered = generation.lower()
    primary = float(any(answer.lower() in lowered for answer in answers))
    first = _first_line(generation)
    first_lowered = first.lower()
    first_line = float(any(answer.lower() in first_lowered for answer in answers))
    normalized_first = normalize_answer(first)
    normalized_em = float(any(normalized_first == normalize_answer(answer) for answer in answers))
    return {
        "score": primary,
        "score_normalized_em": normalized_em,
        "score_first_line": first_line,
    }
