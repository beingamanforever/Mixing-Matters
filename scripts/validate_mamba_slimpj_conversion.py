"""Check the SlimPajama Mamba conversion against the original checkpoint.

Phase 5 loads a local HF conversion of ``state-spaces/mamba-2.8b-slimpj``
because the published checkpoint is in the original state-spaces format, which
transformers cannot load. That conversion carries no published numerical
validation, so this script compares it against the original weights run
through the authors' own ``mamba_ssm`` implementation, on both greedy
generations and next-token logits, exactly as the Phase 2 Mamba-2 conversion
was checked.

Differences are expected and are measured rather than assumed small: the two
paths order tensor contractions differently and bfloat16 makes that visible.
Compare whatever this prints against the Phase 2 Mamba-2 numbers recorded in
docs/phase2-runbook.md.

Usage, from the repository root with the venv built by scripts/setup_gpu.sh,
after scripts/convert_mamba_slimpj.py has produced the conversion:
    PYTHONPATH=src .venv/bin/python scripts/validate_mamba_slimpj_conversion.py
"""

import gc
from pathlib import Path

import torch

from mixing_matters.anchors import build_prompt
from mixing_matters.convert import converted_dir
from mixing_matters.data import read_rows, split_indices
from mixing_matters.models import spec

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

    model_spec = spec("mamba-2.8b-slimpj")
    local = converted_dir(model_spec)
    if not (local / "conversion-manifest.json").exists():
        raise SystemExit(f"no conversion at {local}; run scripts/convert_mamba_slimpj.py first")

    prompts = sweep_prompts(SAMPLES)
    tokenizer = AutoTokenizer.from_pretrained(str(local))
    batches = [tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") for prompt in prompts]

    converted_model = (
        AutoModelForCausalLM.from_pretrained(str(local), dtype=torch.bfloat16).to("cuda").eval()
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
        model_spec.repo, device="cuda", dtype=torch.bfloat16
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
