"""Check the community Mamba-2 conversion against the published checkpoint.

Phase 2 loads `AntonV/mamba2-2.7b-hf` because `state-spaces/mamba2-2.7b` cannot
be loaded by transformers: its config carries no `model_type`. That conversion
is community work with no published numerical validation, so this script
compares it against the original weights run through the authors' own
`mamba_ssm` implementation, on both greedy generations and next-token logits.

Differences are expected. The two paths order tensor contractions differently
and bfloat16 makes that visible, which the transformers documentation states
directly. The purpose here is to measure the size of the difference rather than
to assume it is small.

Usage, from the repository root with the venv built by scripts/setup_gpu.sh:
    PYTHONPATH=src .venv/bin/python scripts/validate_mamba2_conversion.py
"""

import gc
from pathlib import Path

import torch

from mixing_matters.anchors import build_prompt
from mixing_matters.data import read_rows, split_indices
from mixing_matters.models import MODELS

ORIGINAL_REPO = "state-spaces/mamba2-2.7b"
DATA = Path("data/nq-open-10_total_documents_gold_at_0.jsonl.gz")
SAMPLES = 5
NEW_TOKENS = 32
SEED = 240521


def sweep_prompts(count: int) -> list[str]:
    rows = read_rows(DATA)
    exploratory, _ = split_indices(len(rows), SEED)
    return [build_prompt(rows[index], "gold_first")[0] for index in exploratory[:count]]


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS["mamba2-2.7b"]
    prompts = sweep_prompts(SAMPLES)
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    batches = [tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") for prompt in prompts]

    converted_model = (
        AutoModelForCausalLM.from_pretrained(
            spec.repo, revision=spec.revision, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    converted = []
    with torch.inference_mode():
        for input_ids in batches:
            output = converted_model.generate(
                input_ids, do_sample=False, num_beams=1, max_new_tokens=NEW_TOKENS
            )
            text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
            logits = converted_model(input_ids).logits[0, -1].float().cpu()
            converted.append((text, logits))
    del converted_model
    gc.collect()
    torch.cuda.empty_cache()

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    reference_model = MambaLMHeadModel.from_pretrained(
        ORIGINAL_REPO, device="cuda", dtype=torch.bfloat16
    ).eval()

    identical = 0
    top1_agree = 0
    with torch.inference_mode():
        for index, input_ids in enumerate(batches):
            output = reference_model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + NEW_TOKENS,
                cg=False,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
            )
            text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
            logits = reference_model(input_ids).logits[0, -1].float().cpu()

            converted_text, converted_logits = converted[index]
            same_text = text == converted_text
            same_top1 = int(logits.argmax()) == int(converted_logits.argmax())
            identical += same_text
            top1_agree += same_top1
            difference = (logits - converted_logits).abs()
            print(
                f"prompt {index}: tokens {input_ids.shape[1]} "
                f"| identical generation {same_text} | top1 agree {same_top1} "
                f"| max abs logit diff {difference.max().item():.4f} "
                f"| mean abs logit diff {difference.mean().item():.5f} "
                f"| logit scale {logits.abs().max().item():.2f}"
            )
            if not same_text:
                print(f"  converted: {converted_text[:90]!r}")
                print(f"  reference: {text[:90]!r}")

    print(f"\nidentical generations {identical}/{SAMPLES}, top1 agreement {top1_agree}/{SAMPLES}")


if __name__ == "__main__":
    main()
