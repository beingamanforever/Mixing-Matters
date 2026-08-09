#!/usr/bin/env python
"""Validate a converted/loaded NVIDIA Megatron Mamba-2 checkpoint.

Two checks, run inside the NVIDIA Megatron-LM container after the model loads:

1. Coherence: greedy completions on a handful of factual prompts, printed for a
   human to eyeball. A garbled checkpoint load (wrong layer pattern, wrong
   tokenizer) produces obvious nonsense here.
2. A published zero-shot benchmark: PIQA validation accuracy by continuation
   log-likelihood, the standard multiple-choice scoring. Waleffe et al. 2024
   (arXiv:2406.07887, Table 3) report PIQA = 79.82 for the pure 8B Mamba-2 at
   3.5T tokens. The gate is +/- 1.0 point on a large-enough sample.

Launch with the same Megatron model args as scripts/megatron_sweep.py, plus:
    --mm-model-key <key> --mm-piqa-samples <N> [--mm-piqa-file <path>]
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

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
import torch.nn.functional as F
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.spec_utils import import_module
from megatron.inference.text_generation import generate_and_post_process
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

COHERENCE_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The first president of the United States was",
    "Two plus two equals",
    "The opposite of hot is",
    "The largest planet in the solar system is",
]
# Waleffe et al. 2024 (arXiv:2406.07887), Table 3 (pure) and Table 7 (hybrid),
# both at 3.5T tokens.
PIQA_TARGET = {"mamba2-8b": 79.82, "mamba2-hybrid-8b": 79.65}


def model_provider(pre_process=True, post_process=True) -> MambaModel:
    args = get_args()
    config = core_transformer_config_from_args(args)
    assert args.use_legacy_models is False
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


def add_args(parser):
    group = parser.add_argument_group(title="validate")
    group.add_argument("--mm-model-key", default="mamba2-8b")
    group.add_argument("--mm-piqa-samples", type=int, default=1838)
    group.add_argument("--mm-piqa-file", default="/workspace/piqa_valid.jsonl")
    group.add_argument("--port", type=int, default=5000)
    return parser


def _loglikelihood(model, tokenizer, context: str, continuation: str) -> float:
    """Sum log-prob of the continuation tokens given the context, one forward."""
    ctx_ids = tokenizer.tokenize(context)
    full_ids = tokenizer.tokenize(context + continuation)
    cont_start = len(ctx_ids)
    if len(full_ids) <= cont_start:
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_ids, position_ids, attention_mask=None)
    # Megatron returns [b, s, vocab] with parallel_output on a single TP rank.
    if logits.shape[0] != 1 and logits.shape[1] == 1:
        logits = logits.transpose(0, 1)
    logits = logits[0].float()
    log_probs = F.log_softmax(logits, dim=-1)
    total = 0.0
    for position in range(cont_start, len(full_ids)):
        token = full_ids[position]
        total += log_probs[position - 1, token].item()
    return total


def run_piqa(model, tokenizer, path: str, limit: int) -> float:
    examples = []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = examples[:limit]
    correct = 0
    for i, ex in enumerate(examples):
        goal = ex["goal"].strip()
        context = goal + " "
        ll0 = _loglikelihood(model, tokenizer, context, ex["sol1"].strip())
        ll1 = _loglikelihood(model, tokenizer, context, ex["sol2"].strip())
        pred = 0 if ll0 >= ll1 else 1
        if pred == int(ex["label"]):
            correct += 1
        if (i + 1) % 100 == 0:
            print_rank_0(f"  piqa {i + 1}/{len(examples)} running acc={100 * correct / (i + 1):.2f}")
    return 100 * correct / len(examples) if examples else 0.0


def main():
    initialize_megatron(
        extra_args_provider=add_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    args = get_args()
    args.exit_on_missing_checkpoint = True
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        load_checkpoint(model, None, None)
    model = model[0]
    model.eval()
    tokenizer = get_tokenizer()

    print_rank_0("=== coherence completions (greedy, 16 tokens) ===")
    for prompt in COHERENCE_PROMPTS:
        result = generate_and_post_process(
            model, prompts=[prompt], tokens_to_generate=16, top_k_sampling=1,
            top_p_sampling=0.0, temperature=1.0, random_seed=240521,
        )
        print_rank_0(f"  {prompt!r} -> {result[0][0][len(prompt):]!r}")

    if os.path.exists(args.mm_piqa_file):
        print_rank_0(f"=== PIQA zero-shot ({args.mm_piqa_samples} samples) ===")
        acc = run_piqa(model, tokenizer, args.mm_piqa_file, args.mm_piqa_samples)
        target = PIQA_TARGET[args.mm_model_key]
        delta = acc - target
        verdict = "PASS" if abs(delta) <= 1.0 else "OUTSIDE +/-1.0"
        print_rank_0(f"PIQA acc={acc:.2f} target={target} delta={delta:+.2f} [{verdict}]")
    else:
        print_rank_0(f"PIQA file {args.mm_piqa_file} not found; skipped benchmark gate")

    print_rank_0("validate complete")


if __name__ == "__main__":
    main()
