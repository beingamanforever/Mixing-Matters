#!/usr/bin/env python3
"""
Sanity check script to validate converted Megatron-LM Mamba-2 checkpoints.

This runs a short published benchmark (HellaSwag) using EleutherAI's lm-evaluation-harness.
It uses a limit of 200 examples so it runs in minutes, not hours, just to verify
the conversion didn't destroy the weights (i.e., accuracy should be way above random guessing).

Requirements:
    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
"""

import argparse
import subprocess
import sys


def validate(
    model_repo: str,
    tasks: str = "hellaswag",
    limit: int = 200,
    device: str = "cuda:0",
    batch_size: str = "8",
) -> None:
    print(f"[*] Validating checkpoint: {model_repo}")
    print(f"[*] Tasks: {tasks} | Limit: {limit} instances")

    trust_remote_code = model_repo.startswith("nvidia/")
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model_args = f"pretrained={model_repo},trust_remote_code={trust_remote_code},dtype={dtype}"
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--limit",
        str(limit),
        "--device",
        device,
        "--batch_size",
        batch_size,
    ]

    print(f"[*] Running command: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\n[!] Error: 'lm_eval' command not found.")
        print("[!] Please install the evaluation harness first:")
        print("    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Evaluation failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate converted Mamba-2 checkpoints.")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/mamba2-8b-3t-4k",
        help="HuggingFace model repo to validate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag",
        help="Tasks to evaluate (e.g. hellaswag, wikitext)",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of instances to evaluate")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on (e.g. cuda:0, cpu)"
    )
    parser.add_argument("--batch_size", type=str, default="8", help="Batch size for evaluation")
    args = parser.parse_args()
    validate(
        args.model,
        tasks=args.tasks,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
    )
