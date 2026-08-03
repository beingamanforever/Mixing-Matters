from lost_in_the_middle.prompting import Document, get_closedbook_qa_prompt, get_qa_prompt

from .build_positions import place_gold


def build_prompt(row: dict, condition: str) -> tuple[str, int | None]:
    if condition == "closed_book":
        return get_closedbook_qa_prompt(row["question"]), None
    if condition == "oracle":
        documents = [doc for doc in row["ctxs"] if doc["isgold"] is True]
        position = None
    elif condition == "gold_first":
        documents = place_gold(row, 0)["ctxs"]
        position = 0
    elif condition == "gold_middle":
        documents = place_gold(row, 4)["ctxs"]
        position = 4
    else:
        raise ValueError(f"unknown condition: {condition}")
    prompt = get_qa_prompt(
        row["question"],
        [Document.from_dict(document) for document in documents],
        mention_random_ordering=False,
        query_aware_contextualization=False,
    )
    return prompt, position
