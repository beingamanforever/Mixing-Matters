"""Prompt-order variants for the Phase 7 query-position ablation.

The Phase 2 through Phase 8 sweeps use the vendored Liu et al. QA prompt
in its default order: instructions, documents, then question. Phase 7's
fixed-state-compression sub-experiment asks whether the position at
which the question appears in the prompt changes the accuracy-versus-
evidence-position curve, especially for sequence-mixing architectures
whose recurrent state has already compressed the documents by the time
the question tokens are processed.

This module packages three additional prompt orders that reuse the same
instructions and the same formatted document block as the Liu et al.
template:

- ``baseline`` (documents then question) is the Phase 2 baseline and is
  emitted by ``lost_in_the_middle.prompting.get_qa_prompt`` with default
  flags.
- ``question_first`` places the question immediately after the
  instructions and before the documents. The Mamba state sees the query
  tokens before compressing the documents.
- ``bookend`` places the question both before and after the documents,
  matching Liu et al.'s ``query_aware_contextualization`` template.
- ``gold_padded`` reserves a fixed count of filler tokens after the
  gold document by appending a run of a repeated noise token. Total
  prompt length is intentionally allowed to grow with the pad so the
  distance between gold and the end can be varied independently of the
  ten distractors. Callers gate for their own prompt-length budget.

The four variants share the Liu template's document rendering, but they
jointly change question placement, repetition, and token layout. They
therefore support descriptive per-condition comparisons rather than a
single-variable prompt-order claim.
"""

from collections.abc import Iterable

from lost_in_the_middle.prompting import Document, get_qa_prompt

VARIANTS = ("baseline", "question_first", "bookend", "gold_padded")

_QUESTION_FIRST_TEMPLATE = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant).\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "{search_results}\n"
    "\n"
    "Answer:"
)


def _format_documents(documents: Iterable[Document]) -> str:
    """Format a document list the same way Liu et al.'s template does."""
    return "\n".join(
        f"Document [{index + 1}](Title: {document.title}) {document.text}"
        for index, document in enumerate(documents)
    )


# Phase 7 sub-experiment 4e template variation: alternative wordings of the
# instruction and answer cue, documents-then-question order kept fixed. The
# measurement-artifact check asks whether the edge magnitudes survive a change
# of template. ``liu`` is the vendored default; the other two paraphrase the
# instruction and rename the answer cue without changing the task.
TEMPLATES: dict[str, str] = {
    "liu": (
        "Write a high-quality answer for the given question using only the provided "
        "search results (some of which might be irrelevant).\n"
        "\n"
        "{search_results}\n"
        "\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "concise": (
        "Answer the question below from the documents provided. Some documents may "
        "not be relevant.\n"
        "\n"
        "{search_results}\n"
        "\n"
        "Question: {question}\n"
        "Response:"
    ),
    "instructional": (
        "You are given a set of documents and a question. Use only the documents to "
        "answer. If several documents are irrelevant, ignore them.\n"
        "\n"
        "Documents:\n"
        "{search_results}\n"
        "\n"
        "Question: {question}\n"
        "Answer:"
    ),
}


def build_template_prompt(question: str, documents: Iterable[Document], template: str) -> str:
    """Assemble a QA prompt under one of the 4e instruction templates.

    All templates keep the documents-then-question order; only the
    instruction wording and the answer cue change. ``template`` must be a
    key of ``TEMPLATES``.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template: {template!r}; valid: {sorted(TEMPLATES)}")
    document_list = list(documents)
    if not document_list:
        raise ValueError("documents must be a non-empty iterable")
    return TEMPLATES[template].format(
        question=question, search_results=_format_documents(document_list)
    )


def build_variant_prompt(
    question: str,
    documents: Iterable[Document],
    variant: str,
    gold_padded_tokens: int = 0,
    pad_token: str = " padding",
) -> str:
    """Assemble a QA prompt for the requested Phase 7 variant.

    ``gold_padded_tokens`` is only consulted for ``variant='gold_padded'``.
    Each unit of ``gold_padded_tokens`` inserts one copy of ``pad_token``
    directly after the closing brace of the document block, before the
    trailing "Question:" line, so the number of tokens between the last
    document and the question can be raised without touching the ten
    distractors.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant!r}")
    document_list = list(documents)
    if not document_list:
        raise ValueError("documents must be a non-empty iterable")
    if variant == "baseline":
        return get_qa_prompt(
            question,
            document_list,
            mention_random_ordering=False,
            query_aware_contextualization=False,
        )
    if variant == "bookend":
        return get_qa_prompt(
            question,
            document_list,
            mention_random_ordering=False,
            query_aware_contextualization=True,
        )
    if variant == "question_first":
        return _QUESTION_FIRST_TEMPLATE.format(
            question=question, search_results=_format_documents(document_list)
        )
    if variant == "gold_padded":
        if gold_padded_tokens < 0:
            raise ValueError("gold_padded_tokens must be non-negative")
        baseline = get_qa_prompt(
            question,
            document_list,
            mention_random_ordering=False,
            query_aware_contextualization=False,
        )
        # Splice pad tokens directly before the closing "Question:" line so
        # every distractor and the gold document remain untouched and the
        # padding sits between the documents and the question.
        marker = "\n\nQuestion:"
        if baseline.count(marker) != 1:
            raise ValueError("baseline prompt did not contain a unique Question marker")
        pad = pad_token * gold_padded_tokens
        return baseline.replace(marker, pad + marker, 1)
    raise AssertionError(f"unreachable variant branch: {variant!r}")
