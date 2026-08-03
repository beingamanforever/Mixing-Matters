import hashlib
import urllib.request
from pathlib import Path

from . import UPSTREAM_COMMIT

SHA256 = "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9"
NAME = "nq-open-10_total_documents_gold_at_0.jsonl.gz"
URL = (
    "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/"
    f"{UPSTREAM_COMMIT}/qa_data/10_total_documents/{NAME}"
)


def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with urllib.request.urlopen(URL) as response, path.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        path.unlink()
        raise ValueError(f"dataset checksum mismatch: {digest}")
