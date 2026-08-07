#!/usr/bin/env python3
"""
Sanity check script to validate converted Megatron-LM Mamba-2 checkpoints.

This runs a short published benchmark (HellaSwag) using EleutherAI's lm-evaluation-harness.
It uses a limit of 200 examples so it runs in minutes, not hours, just to verify
the conversion didn't destroy the weights (i.e., accuracy should be way above random guessing).

Requirements:
    pip install lm_eval
"""

import argparse
import subprocess
import sys


def validate(model_repo: str, tasks: str = "hellaswag", limit: int = 200) -> None:
    print(f"[*] Validating checkpoint: {model_repo}")
    print(f"[*] Tasks: {tasks} | Limit: {limit} instances")

    # We must pass trust_remote_code=True for NVIDIA's Mamba architectures
    model_args = f"pretrained={model_repo},trust_remote_code=True,dtype=bfloat16"

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
        "cuda:0",
        "--batch_size",
        "8",
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
    args = parser.parse_args()
    validate(args.model)
