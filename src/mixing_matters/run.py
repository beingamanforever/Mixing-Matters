import hashlib
import json
import platform
import uuid
from pathlib import Path

from lost_in_the_middle.metrics import best_subspan_em

from . import UPSTREAM_COMMIT
from .anchors import build_prompt, build_control_prompt
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


def plan_negative(rows: list[dict], n: int = 200) -> list[tuple[int, dict, str, int]]:
    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:n]]
    return [(index, row, "negative_control", pos) for index, row in selected for pos in range(10)]


def plan_order(
    rows: list[dict], n: int = 200, positions: tuple[int, ...] = (0, 4, 9), perms: int = 3
) -> list[tuple[int, dict, str, int, int]]:
    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:n]]
    return [
        (index, row, "distractor_order", pos, perm)
        for index, row in selected
        for pos in positions
        for perm in range(perms)
    ]


def generation_count(work: list[tuple]) -> int:
    """Count control generations plus the floor and ceiling anchors run once per question."""
    return len(work) + 2 * len({item[0] for item in work})


def _run_certify(
    data_path: Path, output: Path, revision: str, control_path: Path | None, work: list
) -> None:
    from .positive_control import validate_control

    if output.exists():
        raise FileExistsError(output)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    generator = Generator(revision)
    control_digest = (
        validate_control(control_path, generator.metadata) if control_path is not None else None
    )

    run_id = str(uuid.uuid4())
    anchor_cache: dict[int, tuple[float, float]] = {}

    # Track prompt lengths per question to ensure length invariance
    position_lengths: dict[str, int] = {}

    def anchor_scores(index: int, row: dict) -> tuple[float, float]:
        if index in anchor_cache:
            return anchor_cache[index]

        floor_prompt, _, _ = build_control_prompt(row, "closed_book")
        floor_generation, _, _ = generator(floor_prompt)
        floor_score = float(best_subspan_em(floor_generation, row["answers"]))

        ceiling_prompt, _, _ = build_control_prompt(row, "oracle")
        ceiling_generation, _, _ = generator(ceiling_prompt)
        ceiling_score = float(best_subspan_em(ceiling_generation, row["answers"]))

        anchor_cache[index] = (floor_score, ceiling_score)
        return anchor_cache[index]

    def generate(item) -> dict:
        index = item[0]
        row = item[1]
        condition = item[2]
        pos = item[3]

        qid = question_id(row, index)
        floor, ceiling = anchor_scores(index, row)

        perm_id = None
        perm_seed = None

        if condition == "negative_control":
            prompt, gold_position, gold_present = build_control_prompt(row, condition, pos=pos)
            distractors = [doc for doc in row["ctxs"] if doc["isgold"] is not True]
            fake_source_index = row["ctxs"].index(distractors[0])
        elif condition == "distractor_order":
            perm_id = item[4]
            # Permutation zero keeps the dataset order, so the shuffles are compared
            # against the ordering the main position sweep actually uses.
            if perm_id:
                perm_seed = hashlib.sha256(f"{qid}:{pos}:{perm_id:02d}".encode()).hexdigest()
            prompt, gold_position, gold_present = build_control_prompt(
                row, condition, pos=pos, perm_seed=perm_seed
            )
            fake_source_index = None
        else:
            raise ValueError(f"Unknown condition: {condition}")

        generation, prompt_tokens, generated_tokens = generator(prompt)

        # Enforce length invariance per question (for negative) or (question, position) (for order)
        invariance_key = qid if condition == "negative_control" else f"{qid}:{pos}"
        previous = position_lengths.setdefault(invariance_key, prompt_tokens)
        if prompt_tokens != previous:
            raise ValueError(
                f"control prompt length variance detected for key {invariance_key}: {prompt_tokens} != {previous}"
            )

        try:
            score = float(best_subspan_em(generation, row["answers"]))
            exclusion = None
        except Exception as error:
            score = None
            exclusion = str(error)

        record = {
            "run_id": run_id,
            "condition": condition,
            "gold_present": gold_present,
            "fake_source_index": fake_source_index,
            "question_id": qid,
            "gold_position": gold_position,
            "permutation_id": perm_id,
            "permutation_seed": perm_seed,
            "ceiling_accuracy": ceiling,
            "floor_accuracy": floor,
            "score": score,
            "prompt": prompt,
            "model_response": generation,
            "correct_answer": row["answers"][0] if row["answers"] else "",
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_seed": SEED,
            "manual_seed": SEED,
            "model_name": MODEL,
            "model_revision": generator.metadata["model_revision"],
            "software_versions": {
                "python": generator.metadata["python"],
                "torch": generator.metadata["torch"],
                "transformers": generator.metadata["transformers"],
                "cuda": generator.metadata["cuda"],
                "attention_implementation": generator.metadata["attention_implementation"],
            },
            "data_revision": UPSTREAM_COMMIT,
            "data_sha256": digest,
            "positive_control_sha256": control_digest,
        }

        if exclusion:
            sidecar = output.with_suffix(".failures.jsonl")
            with sidecar.open("a") as stream:
                stream.write(
                    json.dumps({"run_id": run_id, "question_id": qid, "error": exclusion}) + "\n"
                )

        return record

    write_jsonl(output, (generate(item) for item in work))


def run_certify_negative(
    data_path: Path, output: Path, revision: str, control_path: Path | None, n: int = 200
) -> None:
    rows = read_rows(data_path)
    work = plan_negative(rows, n)
    _run_certify(data_path, output, revision, control_path, work)


def run_certify_order(
    data_path: Path,
    output: Path,
    revision: str,
    control_path: Path | None,
    n: int = 200,
    positions: tuple[int, ...] = (0, 4, 9),
    perms: int = 3,
) -> None:
    rows = read_rows(data_path)
    work = plan_order(rows, n, positions, perms)
    _run_certify(data_path, output, revision, control_path, work)
