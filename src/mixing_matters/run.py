import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path

from lost_in_the_middle.metrics import best_subspan_em

from . import UPSTREAM_COMMIT, models
from .anchors import build_control_prompt, build_prompt
from .build_positions import place_gold
from .data import question_id, read_rows, split_indices
from .download import SHA256
from .io import write_jsonl
from .scoring import score_variants

MODEL = models.MODELS["pythia-2.8b"].repo
SEED = 240521
MAX_NEW_TOKENS = 32
# 25 of the 800 exploratory questions shift by exactly one token as the gold
# document moves position, caused by byte-pair merges at document boundaries.
# A span above this is a real invariance violation, not BPE noise.
MAX_PROMPT_TOKEN_SPAN = 2
# Phase 6 niah_single_1 length targets, in tokens including the generation
# budget, run for the 2,048-token Phase 2 models. 2K sits just under Pythia's
# hard 2,048 limit once MAX_NEW_TOKENS is reserved.
NIAH_LENGTHS = (1024, 2048)
# The needle sentence is constant length and the noise count is fixed within a
# length, so the ten depths of one instance differ only by byte-pair merges at
# the needle boundary. A span above this is a real invariance violation.
MAX_NIAH_PROMPT_TOKEN_SPAN = 4


def _resolve_driver_version(torch_module) -> str:
    """Resolve the NVIDIA driver version, failing loudly if it cannot be found."""
    getter = getattr(torch_module.cuda, "driver_version", None)
    if callable(getter):
        try:
            resolved = getter()
        except Exception:  # noqa: BLE001 - any failure here just falls through to nvidia-smi
            resolved = None
        if resolved:
            return str(resolved)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not determine the NVIDIA driver version") from error
    driver = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not driver:
        raise RuntimeError("could not determine the NVIDIA driver version")
    return driver


def _compute_capability(torch_module) -> str:
    major, minor = torch_module.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _installed_version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


TRUNCATION_PROBE_TOKENS = 3000


def _assert_no_active_truncation(tokenizer) -> None:
    """Fail loudly rather than silently corrupt long prompts.

    The mamba2 tokenizer.json ships a truncation setting with max_length 1024,
    which the default call path does not apply, while every prompt in this
    study is longer than that. Measure the behaviour instead of trusting the
    configuration: tokenize a probe far longer than any prompt and require the
    full length back.
    """
    probe = "token " * TRUNCATION_PROBE_TOKENS
    count = len(tokenizer(probe).input_ids)
    if count < TRUNCATION_PROBE_TOKENS:
        raise ValueError(
            f"tokenizer truncates: {TRUNCATION_PROBE_TOKENS} probe tokens came back as {count}"
        )


def _require_mamba_kernels(mamba_module) -> None:
    """Require the CUDA kernel dispatch path used by MambaMixer.forward.

    transformers.models.mamba.modeling_mamba.MambaMixer.forward dispatches to
    the CUDA kernel path only when selective_state_update, selective_scan_fn,
    mamba_inner_fn, and the causal-conv1d functions all resolve; otherwise it
    silently falls back to the numerically different pytorch reference path.

    The causal-conv1d functions are loaded lazily inside forward, so reading
    the module attributes before that call reports them as missing. Resolve
    them the same way the dispatch does.
    """
    resolved = {
        name: getattr(mamba_module, name, None)
        for name in ("selective_state_update", "selective_scan_fn", "mamba_inner_fn")
    }
    loader = getattr(mamba_module, "_lazy_load_causal_conv1d", None)
    if loader is None:
        resolved["causal_conv1d_update"] = getattr(mamba_module, "causal_conv1d_update", None)
        resolved["causal_conv1d_fn"] = getattr(mamba_module, "causal_conv1d_fn", None)
    else:
        resolved["causal_conv1d_update"], resolved["causal_conv1d_fn"] = loader()
    missing = sorted(name for name, value in resolved.items() if value is None)
    if missing:
        raise RuntimeError(
            "mamba CUDA kernel path unavailable, missing: "
            f"{', '.join(missing)}; refusing to fall back to the pytorch reference path"
        )


def _require_mamba2_kernels(mamba2_module) -> None:
    """Require transformers' module-level mamba2 fast-path flag to be set."""
    if not getattr(mamba2_module, "is_fast_path_available", False):
        raise RuntimeError(
            "mamba2 CUDA kernel path unavailable: is_fast_path_available is False; "
            "refusing to fall back to the pytorch reference path"
        )


def _require_nemotron_h_kernels() -> None:
    """Require the mamba-ssm and causal-conv1d packages Nemotron-H dispatches to.

    Nemotron-H is a hybrid model whose SSM blocks call into mamba-ssm and
    causal-conv1d when they are importable and fall back to a pure-pytorch
    reference implementation otherwise. Failing loudly on missing kernels
    keeps every Phase 8 nemotron-h run on the same execution path.
    """
    missing = []
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        missing.append("causal_conv1d")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        missing.append("mamba_ssm")
    if missing:
        raise RuntimeError(
            "nemotron-h CUDA kernel path unavailable, missing: "
            f"{', '.join(missing)}; refusing to fall back to the pytorch reference path"
        )


def _resolve_execution_path(family: str) -> str:
    if family == "pythia":
        return "pytorch_reference"
    if family == "mamba":
        from transformers.models.mamba import modeling_mamba

        _require_mamba_kernels(modeling_mamba)
        return "cuda_kernels"
    if family == "mamba2":
        from transformers.models.mamba2 import modeling_mamba2

        _require_mamba2_kernels(modeling_mamba2)
        return "cuda_kernels"
    if family in ("llama", "qwen2"):
        # Dense attention transformers. The eager attention implementation
        # keeps them on the same deterministic pytorch reference path Pythia
        # runs on.
        return "pytorch_reference"
    if family == "nemotron-h":
        _require_nemotron_h_kernels()
        return "cuda_kernels"
    raise ValueError(f"unknown model family: {family!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plan(rows: list[dict]) -> list[tuple[int, dict, str]]:
    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:200]]
    # Anchors are ordered before the gold conditions of the same question so
    # their scores are known in time to be attached as floor/ceiling.
    work = [
        (index, row, condition)
        for index, row in selected
        for condition in ("closed_book", "oracle", "gold_first", "gold_middle")
    ]
    if len(work) != 800:
        raise AssertionError("tracer plan must contain exactly 800 unique generations")
    return work


def plan_sweep(rows: list[dict], questions: int = 800) -> list[tuple[int, dict, str, int | None]]:
    """Plan the ten-position sweep plus floor and ceiling anchors.

    For each of the first ``questions`` exploratory questions: closed_book,
    oracle, then gold at positions 0 through 9, in that order so the anchor
    scores are known in time to be attached to every gold record.
    """
    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:questions]]
    conditions: list[tuple[str, int | None]] = [("closed_book", None), ("oracle", None)] + [
        ("gold", position) for position in range(10)
    ]
    work = [
        (index, row, condition, position)
        for index, row in selected
        for condition, position in conditions
    ]
    if len(work) != 12 * questions:
        raise AssertionError("sweep plan must contain exactly 12 items per question")
    return work


class Generator:
    def __init__(self, model_spec: models.ModelSpec, revision: str):
        import torch
        import transformers
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required")

        # An original state-spaces checkpoint cannot be loaded by transformers,
        # so it is converted to an HF directory ahead of time (see convert.py)
        # and loaded from there. Recording the source revision from the
        # conversion manifest keeps provenance pointing at the pinned upstream
        # checkpoint, and loading the converted directory keeps this run on the
        # same transformers CUDA-kernel path as every other Mamba run.
        checkpoint_dir = None
        if model_spec.format == "mamba_ssm":
            from .convert import converted_dir

            checkpoint_dir = converted_dir(model_spec)
            manifest_path = checkpoint_dir / "conversion-manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{model_spec.key} has no converted checkpoint at {checkpoint_dir}; "
                    "run scripts/convert_mamba_slimpj.py before the sweep"
                )
            exact_revision = json.loads(manifest_path.read_text())["source_revision"]
            load_source = str(checkpoint_dir)
            revision_kwargs: dict = {}
        else:
            exact_revision = model_info(model_spec.repo, revision=revision).sha
            if not exact_revision:
                raise ValueError("Hugging Face did not resolve an exact model revision")
            load_source = model_spec.repo
            revision_kwargs = {"revision": exact_revision}

        # Resolve the execution path before loading weights so an
        # unavailable CUDA kernel path fails fast, not after a multi-GB load.
        execution_path = _resolve_execution_path(model_spec.family)

        self.torch = torch
        self.spec = model_spec
        tokenizer_kwargs = dict(revision_kwargs)
        # Nemotron-H ships an auto_map for tokenizer and model that requires
        # transformers to execute repository code. Trust is scoped by
        # repository namespace: only the NVIDIA Nemotron-H family gets it, so
        # other models continue to load through the standard classes.
        trust_remote_code = model_spec.family == "nemotron-h" and model_spec.repo.startswith(
            "nvidia/"
        )
        if trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(load_source, **tokenizer_kwargs)
        _assert_no_active_truncation(self.tokenizer)

        model_kwargs = {"dtype": torch.bfloat16, **revision_kwargs}
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        attention_implementation = None
        # Attention implementation, resolved per family. Pythia and Nemotron-H
        # pin ``eager`` to match the earlier phases' deterministic pytorch
        # reference path (Nemotron-H's SSM blocks separately dispatch through
        # mamba-ssm and causal-conv1d). The Phase 8 dense transformers,
        # Llama-3.1 and Qwen2.5, use ``sdpa`` for throughput: descriptive
        # comparison across full systems is not a matched control and their
        # 8B-scale eager attention runs prohibitively slow on a single GPU.
        # SDPA still runs entirely under torch; only the attention kernel
        # differs from the earlier eager runs.
        if model_spec.family in ("pythia", "nemotron-h"):
            attention_implementation = "eager"
        elif model_spec.family in ("llama", "qwen2"):
            attention_implementation = "sdpa"
        if attention_implementation is not None:
            model_kwargs["attn_implementation"] = attention_implementation
        self.model = AutoModelForCausalLM.from_pretrained(load_source, **model_kwargs).to("cuda")
        self.model.eval()
        set_seed(SEED)

        max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None)
        self.max_context = (
            int(max_position_embeddings) if max_position_embeddings is not None else None
        )

        self.metadata = {
            "seed": SEED,
            "model": model_spec.repo,
            "model_key": model_spec.key,
            "family": model_spec.family,
            "model_revision": exact_revision,
            "checkpoint_format": model_spec.format,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "training_corpus": model_spec.training_corpus,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "driver": _resolve_driver_version(torch),
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": attention_implementation,
            # Read back from the loaded model rather than the requested dtype so a
            # silent fallback (e.g. no bf16 support) is caught, not masked.
            "dtype": str(self.model.dtype),
            "execution_path": execution_path,
            "compute_capability": _compute_capability(torch),
            "mamba_ssm": _installed_version("mamba_ssm"),
            "causal_conv1d": _installed_version("causal_conv1d"),
        }

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        if self.max_context is not None and prompt_tokens + MAX_NEW_TOKENS > self.max_context:
            raise ValueError(f"context overflow: {prompt_tokens} + {MAX_NEW_TOKENS}")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS
            )
        new_ids = output_ids[0, prompt_tokens:]
        return (
            self.tokenizer.decode(new_ids, skip_special_tokens=True),
            prompt_tokens,
            int(new_ids.shape[0]),
        )


def _provenance(
    run_id: str,
    generator: "Generator",
    digest: str,
    control_digest: str | None,
    software_versions: dict,
) -> dict:
    """Fields common to every generation record, tracer and sweep alike."""
    return {
        "run_id": run_id,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "random_seed": SEED,
        "manual_seed": SEED,
        "model_name": generator.metadata["model"],
        "model_revision": generator.metadata["model_revision"],
        "software_versions": software_versions,
        "data_revision": UPSTREAM_COMMIT,
        "data_sha256": digest,
        "positive_control_sha256": control_digest,
    }


def run_tracer(data_path: Path, output: Path, revision: str, control_path: Path) -> None:
    from .positive_control import validate_control

    if output.exists():
        raise FileExistsError(output)
    failures = output.with_suffix(".failures.jsonl")
    if failures.exists():
        raise FileExistsError(failures)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    rows = read_rows(data_path)
    generator = Generator(models.MODELS["pythia-2.8b"], revision)
    control_digest = validate_control(control_path, generator.metadata)

    run_id = str(uuid.uuid4())
    position_lengths: dict[str, int] = {}
    software_versions = {
        "python": generator.metadata["python"],
        "torch": generator.metadata["torch"],
        "transformers": generator.metadata["transformers"],
        "cuda": generator.metadata["cuda"],
        "driver": generator.metadata["driver"],
        "gpu": generator.metadata["gpu"],
        "attention_implementation": generator.metadata["attention_implementation"],
        "dtype": generator.metadata["dtype"],
    }

    def generate(index: int, row: dict, condition: str) -> tuple[dict, float | None]:
        prompt, gold_position = build_prompt(row, condition)
        generation, prompt_tokens, generated_tokens = generator(prompt)
        qid = question_id(row, index)
        if gold_position is not None:
            previous = position_lengths.setdefault(qid, prompt_tokens)
            if prompt_tokens != previous:
                raise ValueError(f"position changed prompt length at source index {index}")

        try:
            scores = score_variants(generation, row["answers"])
        except Exception as error:  # noqa: BLE001 - scoring failures are logged, not fatal
            scores = {"score": None, "score_normalized_em": None, "score_first_line": None}
            with failures.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "question_id": qid,
                            "condition": condition,
                            "source_index": index,
                            "error": str(error),
                        }
                    )
                    + "\n"
                )

        record = {
            "question_id": qid,
            "source_index": index,
            "condition": condition,
            "gold_position": gold_position,
            "prompt": prompt,
            "model_response": generation,
            "correct_answer": row["answers"][0] if row["answers"] else "",
            "answers": row["answers"],
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            **scores,
            **_provenance(run_id, generator, digest, control_digest, software_versions),
        }
        return record, scores["score"]

    def records():
        work = plan(rows)
        groups = [work[start : start + 4] for start in range(0, len(work), 4)]
        for group in groups:
            conditions = [item[2] for item in group]
            if conditions != ["closed_book", "oracle", "gold_first", "gold_middle"]:
                raise ValueError(f"unexpected condition order in tracer plan group: {conditions}")
        for group in groups:
            index, row, closed_condition = group[0]
            closed_record, floor = generate(index, row, closed_condition)
            oracle_index, oracle_row, oracle_condition = group[1]
            oracle_record, ceiling = generate(oracle_index, oracle_row, oracle_condition)
            closed_record["floor_accuracy"] = floor
            closed_record["ceiling_accuracy"] = ceiling
            oracle_record["floor_accuracy"] = floor
            oracle_record["ceiling_accuracy"] = ceiling
            yield closed_record
            yield oracle_record
            for gold_index, gold_row, gold_condition in group[2:]:
                gold_record, _ = generate(gold_index, gold_row, gold_condition)
                gold_record["floor_accuracy"] = floor
                gold_record["ceiling_accuracy"] = ceiling
                yield gold_record

    write_jsonl(output, records())


def _gold_prompt(row: dict, position: int) -> str:
    from lost_in_the_middle.prompting import Document, get_qa_prompt

    documents = place_gold(row, position)["ctxs"]
    return get_qa_prompt(
        row["question"],
        [Document.from_dict(document) for document in documents],
        mention_random_ordering=False,
        query_aware_contextualization=False,
    )


def run_sweep(
    data_path: Path,
    output: Path,
    model_key: str,
    revision: str,
    questions: int = 800,
    max_prompt_token_span: int = MAX_PROMPT_TOKEN_SPAN,
) -> None:
    """Run the closed_book/oracle/gold(0-9) sweep for one model.

    Unlike run_tracer, this never validates or gates on the key-value
    positive control: a model failing that control is a finding about the
    model, not a reason to abort the position sweep.

    ``max_prompt_token_span`` gates the byte-pair-merge noise across the
    ten gold positions of a question. The default ``MAX_PROMPT_TOKEN_SPAN``
    (``2``) is tight enough for the GPTNeoX and Mamba tokenizers used in
    Phases 2 through 6; Phase 8 raises it because Llama, Qwen, and
    Nemotron-H tokenizers each have their own merge behavior at document
    boundaries.
    """
    model_spec = models.spec(model_key)

    if output.exists():
        raise FileExistsError(output)
    failures = output.with_suffix(".failures.jsonl")
    if failures.exists():
        raise FileExistsError(failures)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    rows = read_rows(data_path)
    generator = Generator(model_spec, revision)

    run_id = str(uuid.uuid4())
    software_versions = {
        "python": generator.metadata["python"],
        "torch": generator.metadata["torch"],
        "transformers": generator.metadata["transformers"],
        "cuda": generator.metadata["cuda"],
        "driver": generator.metadata["driver"],
        "gpu": generator.metadata["gpu"],
        "attention_implementation": generator.metadata["attention_implementation"],
        "dtype": generator.metadata["dtype"],
        "model_key": generator.metadata["model_key"],
        "family": generator.metadata["family"],
        "execution_path": generator.metadata["execution_path"],
        "compute_capability": generator.metadata["compute_capability"],
        "mamba_ssm": generator.metadata["mamba_ssm"],
        "causal_conv1d": generator.metadata["causal_conv1d"],
    }

    def generate(
        index: int, row: dict, condition: str, position: int | None
    ) -> tuple[dict, float | None, int]:
        if condition == "gold":
            prompt = _gold_prompt(row, position)
            gold_position = position
        else:
            prompt, gold_position = build_prompt(row, condition)
        generation, prompt_tokens, generated_tokens = generator(prompt)
        qid = question_id(row, index)

        try:
            scores = score_variants(generation, row["answers"])
        except Exception as error:  # noqa: BLE001 - scoring failures are logged, not fatal
            scores = {"score": None, "score_normalized_em": None, "score_first_line": None}
            with failures.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "question_id": qid,
                            "condition": condition,
                            "source_index": index,
                            "error": str(error),
                        }
                    )
                    + "\n"
                )

        record = {
            "question_id": qid,
            "source_index": index,
            "condition": condition,
            "gold_position": gold_position,
            "prompt": prompt,
            "model_response": generation,
            "correct_answer": row["answers"][0] if row["answers"] else "",
            "answers": row["answers"],
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            "prompt_token_span": None,
            **scores,
            **_provenance(run_id, generator, digest, None, software_versions),
        }
        return record, scores["score"], prompt_tokens

    def records():
        work = plan_sweep(rows, questions)
        groups = [work[start : start + 12] for start in range(0, len(work), 12)]
        for group in groups:
            conditions = [item[2] for item in group]
            if conditions != ["closed_book", "oracle"] + ["gold"] * 10:
                raise ValueError(f"unexpected condition order in sweep plan group: {conditions}")
        for group in groups:
            index, row, closed_condition, _ = group[0]
            closed_record, floor, _ = generate(index, row, closed_condition, None)
            oracle_index, oracle_row, oracle_condition, _ = group[1]
            oracle_record, ceiling, _ = generate(oracle_index, oracle_row, oracle_condition, None)
            closed_record["floor_accuracy"] = floor
            closed_record["ceiling_accuracy"] = ceiling
            oracle_record["floor_accuracy"] = floor
            oracle_record["ceiling_accuracy"] = ceiling
            yield closed_record
            yield oracle_record

            gold_records = []
            gold_token_counts = []
            for gold_index, gold_row, gold_condition, position in group[2:]:
                gold_record, _, prompt_tokens = generate(
                    gold_index, gold_row, gold_condition, position
                )
                gold_record["floor_accuracy"] = floor
                gold_record["ceiling_accuracy"] = ceiling
                gold_records.append(gold_record)
                gold_token_counts.append(prompt_tokens)

            span = max(gold_token_counts) - min(gold_token_counts)
            if span > max_prompt_token_span:
                raise ValueError(
                    f"gold prompt length span too large for source index {index}: "
                    f"{span} tokens across positions 0-9 (limit {max_prompt_token_span})"
                )
            for gold_record in gold_records:
                gold_record["prompt_token_span"] = span
                yield gold_record

    write_jsonl(output, records())


def _niah_question_id(length: int, instance: dict) -> str:
    value = f"niah_single_1:{length}:{instance['index']}:{instance['key']}:{instance['value']}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def run_ruler_sweep(
    output: Path,
    model_key: str,
    revision: str,
    lengths: tuple[int, ...] = NIAH_LENGTHS,
    num_samples: int = 100,
) -> None:
    """Run the RULER niah_single_1 depth sweep for one model at each length.

    For every length target and every needle instance the runner writes twelve
    records, in the same shape as ``run_sweep``: a closed-book floor (haystack
    with no needle), an oracle ceiling (the needle with no noise), and the
    needle at depths 0 through 9. gold_position carries the depth so the Phase 2
    position-curve and edge statistics apply unchanged, one length at a time.
    Records also carry ``context_length`` and ``num_haystack`` so the report can
    separate the lengths. Never validates or gates on any control.
    """
    from . import ruler

    model_spec = models.spec(model_key)
    if output.exists():
        raise FileExistsError(output)
    failures = output.with_suffix(".failures.jsonl")
    if failures.exists():
        raise FileExistsError(failures)
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive: {num_samples}")

    generator = Generator(model_spec, revision)
    run_id = str(uuid.uuid4())
    software_versions = {
        "python": generator.metadata["python"],
        "torch": generator.metadata["torch"],
        "transformers": generator.metadata["transformers"],
        "cuda": generator.metadata["cuda"],
        "driver": generator.metadata["driver"],
        "gpu": generator.metadata["gpu"],
        "attention_implementation": generator.metadata["attention_implementation"],
        "dtype": generator.metadata["dtype"],
        "model_key": generator.metadata["model_key"],
        "family": generator.metadata["family"],
        "execution_path": generator.metadata["execution_path"],
        "compute_capability": generator.metadata["compute_capability"],
        "mamba_ssm": generator.metadata["mamba_ssm"],
        "causal_conv1d": generator.metadata["causal_conv1d"],
    }

    def count_tokens(text: str) -> int:
        return len(generator.tokenizer(text).input_ids)

    def score_and_log(generation: str, answers: list[str], qid: str, condition: str) -> dict:
        try:
            return ruler.score_variants(generation, answers)
        except Exception as error:  # noqa: BLE001 - scoring failures are logged, not fatal
            with failures.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "question_id": qid,
                            "condition": condition,
                            "error": str(error),
                        }
                    )
                    + "\n"
                )
            return {"score": None, "score_normalized_em": None, "score_first_line": None}

    def base_record(
        qid: str,
        condition: str,
        gold_position: int | None,
        prompt: str,
        generation: str,
        answers: list[str],
        prompt_tokens: int,
        generated_tokens: int,
        length: int,
        num_haystack: int,
        instance: dict,
        scores: dict,
    ) -> dict:
        return {
            "task": "niah_single_1",
            "ruler_commit": ruler.RULER_COMMIT,
            "context_length": length,
            "num_haystack": num_haystack,
            "needle_key": instance["key"],
            "instance_index": instance["index"],
            "instance_seed": instance["seed"],
            "question_id": qid,
            "condition": condition,
            "gold_position": gold_position,
            "prompt": prompt,
            "model_response": generation,
            "correct_answer": answers[0],
            "answers": answers,
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            "prompt_token_span": None,
            **scores,
            **_provenance(run_id, generator, None, None, software_versions),
        }

    def records():
        for length in lengths:
            reference = ruler.make_instance(0)
            num_haystack = ruler.solve_haystack_size(
                count_tokens, length, MAX_NEW_TOKENS, reference
            )
            for index in range(num_samples):
                instance = ruler.make_instance(index)
                answers = [instance["value"]]
                qid = _niah_question_id(length, instance)

                floor_prompt = ruler.build_floor_prompt(instance, num_haystack)
                floor_gen, floor_tokens, floor_new = generator(floor_prompt)
                floor_scores = score_and_log(floor_gen, answers, qid, "closed_book")
                floor = floor_scores["score"]

                ceiling_prompt = ruler.build_ceiling_prompt(instance)
                ceiling_gen, ceiling_tokens, ceiling_new = generator(ceiling_prompt)
                ceiling_scores = score_and_log(ceiling_gen, answers, qid, "oracle")
                ceiling = ceiling_scores["score"]

                floor_record = base_record(
                    qid, "closed_book", None, floor_prompt, floor_gen, answers,
                    floor_tokens, floor_new, length, num_haystack, instance, floor_scores,
                )
                ceiling_record = base_record(
                    qid, "oracle", None, ceiling_prompt, ceiling_gen, answers,
                    ceiling_tokens, ceiling_new, length, num_haystack, instance, ceiling_scores,
                )
                for record in (floor_record, ceiling_record):
                    record["floor_accuracy"] = floor
                    record["ceiling_accuracy"] = ceiling
                yield floor_record
                yield ceiling_record

                gold_records = []
                gold_token_counts = []
                for depth in ruler.DEPTHS:
                    prompt = ruler.build_gold_prompt(instance, depth, num_haystack)
                    generation, prompt_tokens, generated_tokens = generator(prompt)
                    scores = score_and_log(generation, answers, qid, "gold")
                    record = base_record(
                        qid, "gold", depth, prompt, generation, answers,
                        prompt_tokens, generated_tokens, length, num_haystack, instance, scores,
                    )
                    record["floor_accuracy"] = floor
                    record["ceiling_accuracy"] = ceiling
                    gold_records.append(record)
                    gold_token_counts.append(prompt_tokens)

                span = max(gold_token_counts) - min(gold_token_counts)
                if span > MAX_NIAH_PROMPT_TOKEN_SPAN:
                    raise ValueError(
                        f"niah prompt length span too large at length {length} for "
                        f"instance {index}: {span} tokens across depths 0-9"
                    )
                for record in gold_records:
                    record["prompt_token_span"] = span
                    yield record

    write_jsonl(output, records())


def run_kv_control(model_spec: models.ModelSpec, output: Path, revision: str) -> None:
    """Run the key-value positive control against an arbitrary model spec.

    Generalizes positive_control.run_control, which is hardwired to the
    Pythia Generator, so every model in the registry can be probed for
    key-value retrieval ahead of its position sweep.
    """
    from lost_in_the_middle.prompting import get_kv_retrieval_prompt

    from .positive_control import POSITIONS, control_examples

    if output.exists():
        raise FileExistsError(output)
    generator = Generator(model_spec, revision)

    def records():
        for example in control_examples():
            gold = example["pairs"][0]
            distractors = example["pairs"][1:]
            for condition, position in POSITIONS.items():
                pairs = list(distractors)
                pairs.insert(position, gold)
                prompt = get_kv_retrieval_prompt(pairs, example["key"])
                generation, prompt_tokens, generated_tokens = generator(prompt)
                yield {
                    "control_id": example["control_id"],
                    "condition": condition,
                    "prompt": prompt,
                    "generation": generation,
                    "gold": example["value"],
                    "score": float(example["value"].lower() in generation.lower()),
                    "prompt_token_count": prompt_tokens,
                    "generated_token_count": generated_tokens,
                    **generator.metadata,
                }

    write_jsonl(output, records())


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
    failures = output.with_suffix(".failures.jsonl")
    if failures.exists():
        raise FileExistsError(failures)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    generator = Generator(models.MODELS["pythia-2.8b"], revision)
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
        except Exception as error:  # noqa: BLE001 - scoring failures are logged, not fatal
            score = None
            exclusion = str(error)

        record = {
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
            **_provenance(
                run_id,
                generator,
                digest,
                control_digest,
                {
                    "python": generator.metadata["python"],
                    "torch": generator.metadata["torch"],
                    "transformers": generator.metadata["transformers"],
                    "cuda": generator.metadata["cuda"],
                    "attention_implementation": generator.metadata["attention_implementation"],
                },
            ),
        }

        if exclusion:
            with failures.open("a") as stream:
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
