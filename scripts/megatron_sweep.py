#!/usr/bin/env python
"""Run a Phase 3 position sweep for an NVIDIA Megatron-LM Mamba-2 checkpoint.

The two Phase 3 checkpoints (``nvidia/mamba2-8b-3t-4k`` and its hybrid) are
published only as Megatron-LM distributed checkpoints; transformers cannot load
them. This driver runs inside the NVIDIA Megatron-LM container (see
examples/mamba/Dockerfile at commit df61e60) and loads the checkpoint through
Megatron's own MambaModel, exactly as tools/run_mamba_text_generation_server.py
does, then drives the repository's own ``run.run_sweep`` and
``run.run_kv_control`` through the injected-generator seam.

Because run_sweep is pure Python apart from the generator, this keeps the whole
Phase 3 pipeline -- prompt construction, floor/ceiling anchors, the
length-invariance check, scoring, and the JSONL record schema -- byte-for-byte
identical to every other phase. Only the backend that turns a prompt into a
greedy generation changes.

Launch (inside the container, repository mounted at /workspace/megatron-mixing):

    torchrun --nproc_per_node 1 scripts/megatron_sweep.py \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
        --untie-embeddings-and-output-weights --num-layers 56 --hidden-size 4096 \
        --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
        --hybrid-attention-ratio <0.0 pure | 0.08 hybrid> \
        --hybrid-mlp-ratio <0.0 pure | 0.5 hybrid> \
        --attention-dropout 0.0 --hidden-dropout 0.0 --disable-bias-linear \
        --normalization RMSNorm --seq-length 4096 --max-position-embeddings 4096 \
        --position-embedding-type none --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model <tokenizer.model> --distributed-backend nccl --bf16 \
        --micro-batch-size 1 --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec --seed 42 \
        --load <checkpoint-dir> \
        --mm-model-key <mamba2-8b|mamba2-hybrid-8b> --mm-revision <sha> \
        --mm-data <data.jsonl.gz> --mm-questions 800 \
        --mm-sweep-output <out.jsonl> [--mm-kv-output <kv.jsonl>] [--mm-add-bos]
"""

import os
import platform
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

import torch

# Bare-metal Python 3.12 compatibility: torch.compile (TorchDynamo) is
# unsupported on Python 3.12 in torch 2.2, but Megatron and Transformer Engine
# apply @torch.compile at import time. torch.compile is only an optimization, so
# for inference eager execution is numerically identical. This guard is a no-op
# on the Python 3.10 container path where torch.compile works.
if sys.version_info >= (3, 12):
    def _noop_torch_compile(model=None, *args, **kwargs):
        return model if callable(model) else (lambda fn: fn)
    torch.compile = _noop_torch_compile
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.spec_utils import import_module
from megatron.inference.text_generation import generate_and_post_process
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

GEN_TOKENS = 32
SEED = 240521


def model_provider(pre_process=True, post_process=True) -> MambaModel:
    """Build the MambaModel from args, mirroring the NVIDIA mamba example."""
    args = get_args()
    print_rank_0("building Mamba model ...")
    config = core_transformer_config_from_args(args)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"
    if args.spec is None:
        raise ValueError("a Mamba layer spec is required (--spec ...)")
    mamba_stack_spec = import_module(args.spec)
    return MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
    )


def add_sweep_args(parser):
    group = parser.add_argument_group(title="phase3 sweep")
    group.add_argument("--mm-model-key", required=True)
    group.add_argument("--mm-revision", required=True)
    group.add_argument("--mm-data", default=None)
    group.add_argument("--mm-questions", type=int, default=800)
    group.add_argument("--mm-sweep-output", default=None)
    group.add_argument("--mm-kv-output", default=None)
    group.add_argument("--mm-add-bos", action="store_true")
    group.add_argument("--port", type=int, default=5000)
    return parser


class MegatronGenerator:
    """A run.Generator-compatible backend backed by Megatron greedy decoding."""

    def __init__(self, model, args):
        import transformers

        from mixing_matters import models as mm_models

        self.model = model
        self.tokenizer = get_tokenizer()
        self.add_bos = args.mm_add_bos
        spec = mm_models.spec(args.mm_model_key)
        self.metadata = {
            "seed": SEED,
            "model": spec.repo,
            "model_key": args.mm_model_key,
            "family": spec.family,
            "model_revision": args.mm_revision,
            "checkpoint_format": "megatron",
            "checkpoint_dir": args.load,
            "training_corpus": None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "driver": _driver_version(),
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": None,
            "dtype": "torch.bfloat16",
            "execution_path": "megatron_cuda_kernels",
            "compute_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(0)),
            "mamba_ssm": _pkg_version("mamba_ssm"),
            "causal_conv1d": _pkg_version("causal_conv1d"),
            "hybrid_attention_ratio": args.hybrid_attention_ratio,
            "hybrid_mlp_ratio": args.hybrid_mlp_ratio,
            "hybrid_override_pattern": "".join(getattr(self.model.decoder, "layer_type_list", []))
            if hasattr(self.model, "decoder")
            else None,
            "add_bos": self.add_bos,
        }

    def _prompt_tokens(self, prompt: str) -> int:
        ids = self.tokenizer.tokenize(prompt)
        return len(ids) + (1 if self.add_bos else 0)

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        prompt_tokens = self._prompt_tokens(prompt)
        result = generate_and_post_process(
            self.model,
            prompts=[prompt],
            tokens_to_generate=GEN_TOKENS,
            return_output_log_probs=False,
            top_k_sampling=1,  # greedy argmax
            top_p_sampling=0.0,
            temperature=1.0,
            add_BOS=self.add_bos,
            use_eod_token_for_early_termination=True,
            random_seed=SEED,
        )
        # Only the first pipeline stage returns; with PP=1 that is this process.
        full_texts, _segments, _logprobs, tokens = result
        full = full_texts[0]
        total_tokens = len(tokens[0])
        generated_tokens = max(0, total_tokens - prompt_tokens)
        if full.startswith(prompt):
            generation = full[len(prompt):]
        else:
            # Detokenization did not reproduce the prompt prefix exactly; fall
            # back to detokenizing only the generated token tail.
            generation = self.tokenizer.detokenize(tokens[0][prompt_tokens:])
        return generation, prompt_tokens, generated_tokens


def _driver_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - driver string is best-effort metadata
        return "unknown"


def _pkg_version(name: str):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    initialize_megatron(
        extra_args_provider=add_sweep_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    args = get_args()
    args.exit_on_missing_checkpoint = True

    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        load_checkpoint(model, None, None)
    assert len(model) == 1
    model = model[0]
    model.eval()

    generator = MegatronGenerator(model, args)
    print_rank_0(f"loaded {args.mm_model_key}; layer pattern: "
                 f"{generator.metadata.get('hybrid_override_pattern')}")

    from pathlib import Path

    from mixing_matters import models as mm_models
    from mixing_matters import run as mm_run

    if args.mm_kv_output:
        print_rank_0(f"running KV positive control -> {args.mm_kv_output}")
        mm_run.run_kv_control(
            mm_models.spec(args.mm_model_key),
            Path(args.mm_kv_output),
            args.mm_revision,
            generator=generator,
        )

    if args.mm_sweep_output:
        print_rank_0(f"running position sweep -> {args.mm_sweep_output}")
        mm_run.run_sweep(
            Path(args.mm_data),
            Path(args.mm_sweep_output),
            args.mm_model_key,
            args.mm_revision,
            questions=args.mm_questions,
            generator=generator,
        )

    print_rank_0("megatron_sweep complete")


if __name__ == "__main__":
    main()
