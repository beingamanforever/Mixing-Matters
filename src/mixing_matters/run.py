import hashlib
import platform
from pathlib import Path

from lost_in_the_middle.metrics import best_subspan_em

from . import UPSTREAM_COMMIT
from .anchors import build_prompt
from .data import question_id, read_rows, split_indices
from .download import SHA256
from .io import write_jsonl

MODEL = "EleutherAI/pythia-2.8b"
SEED = 240521
MAX_NEW_TOKENS = 32


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plan(rows: list[dict]) -> list[tuple[int, dict, str]]:
    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:200]]
    work = [
        (index, row, condition)
        for index, row in selected
        for condition in ("gold_first", "gold_middle")
    ]
    work += [
        (index, row, condition)
        for index, row in selected[:50]
        for condition in ("closed_book", "oracle")
    ]
    if len(work) != 500:
        raise AssertionError("tracer plan must contain exactly 500 unique generations")
    return work


class Generator:
    def __init__(self, revision: str):
        import torch
        import transformers
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required")
        exact_revision = model_info(MODEL, revision=revision).sha
        if not exact_revision:
            raise ValueError("Hugging Face did not resolve an exact model revision")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=exact_revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=exact_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        self.model.eval()
        set_seed(SEED)
        self.max_context = int(self.model.config.max_position_embeddings)
        self.metadata = {
            "seed": SEED,
            "model": MODEL,
            "model_revision": exact_revision,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": "eager",
        }

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        if prompt_tokens + MAX_NEW_TOKENS > self.max_context:
            raise ValueError(f"context overflow: {prompt_tokens} + {MAX_NEW_TOKENS}")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS
            )
        new_ids = output_ids[0, prompt_tokens:]
        return (
            self.tokenizer.decode(new_ids, skip_special_tokens=True),
            prompt_tokens,
            int(new_ids.shape[0]),
        )


def run_tracer(data_path: Path, output: Path, revision: str, control_path: Path) -> None:
    from .positive_control import validate_control

    if output.exists():
        raise FileExistsError(output)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    rows = read_rows(data_path)
    generator = Generator(revision)
    control_digest = validate_control(control_path, generator.metadata)
    position_lengths: dict[str, int] = {}

    def generate(index: int, row: dict, condition: str) -> dict:
        prompt, gold_position = build_prompt(row, condition)
        generation, prompt_tokens, generated_tokens = generator(prompt)
        qid = question_id(row, index)
        if gold_position is not None:
            previous = position_lengths.setdefault(qid, prompt_tokens)
            if prompt_tokens != previous:
                raise ValueError(f"position changed prompt length at source index {index}")
        return {
            "question_id": qid,
            "source_index": index,
            "condition": condition,
            "gold_position": gold_position,
            "prompt": prompt,
            "generation": generation,
            "gold": row["answers"],
            "score": best_subspan_em(generation, row["answers"]),
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            "data_revision": UPSTREAM_COMMIT,
            "data_sha256": digest,
            "positive_control_sha256": control_digest,
            **generator.metadata,
        }

    write_jsonl(output, (generate(*item) for item in plan(rows)))
