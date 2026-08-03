import pytest

from mixing_matters.io import write_jsonl


def test_raw_output_is_never_overwritten(tmp_path):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, [{"a": 1}])
    with pytest.raises(FileExistsError):
        write_jsonl(path, [{"a": 2}])
