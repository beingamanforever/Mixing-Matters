import pytest


@pytest.fixture
def row():
    documents = [
        {"title": f"d{i}", "text": f"text {i}", "isgold": i == 0, "hasanswer": i == 0}
        for i in range(10)
    ]
    return {"question": "Who?", "answers": ["Gold"], "ctxs": documents}
