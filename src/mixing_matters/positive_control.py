import hashlib
import json
import random
import statistics
import uuid
from pathlib import Path

from lost_in_the_middle.prompting import get_kv_retrieval_prompt

from .io import read_jsonl, write_jsonl
from .run import SEED, Generator

EXAMPLES = 50
KEYS = 30
POSITIONS = {f"kv_position_{slot}": round(slot * (KEYS - 1) / 9) for slot in range(10)}
MIN_EDGE_ADVANTAGE = 0.05


def control_examples() -> list[dict]:
    rng = random.Random(SEED)
    examples = []
    for index in range(EXAMPLES):
        pairs = [
            (str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128))))
            for _ in range(KEYS)
        ]
        key, value = pairs[0]
        examples.append({"control_id": index, "pairs": pairs, "key": key, "value": value})
    return examples


def run_control(output: Path, revision: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    generator = Generator(revision)

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


def validate_control(path: Path, metadata: dict) -> str:
    records = read_jsonl(path)
    if len(records) != EXAMPLES * len(POSITIONS):
        raise ValueError("positive control is incomplete")
    keys = {(record.get("control_id"), record.get("condition")) for record in records}
    expected = {(index, condition) for index in range(EXAMPLES) for condition in POSITIONS}
    if keys != expected:
        raise ValueError("positive control pairs are incomplete or duplicated")
    for control_id in range(EXAMPLES):
        lengths = {
            record["prompt_token_count"] for record in records if record["control_id"] == control_id
        }
        if len(lengths) != 1:
            raise ValueError(f"positive control position changed prompt length: {control_id}")
    fields = (
        "model",
        "model_revision",
        "seed",
        "python",
        "torch",
        "transformers",
        "cuda",
        "gpu",
        "attention_implementation",
    )
    for field in fields:
        if {record.get(field) for record in records} != {metadata[field]}:
            raise ValueError(f"positive control {field} does not match tracer")
    accuracies = {
        condition: sum(record["score"] for record in records if record["condition"] == condition)
        / EXAMPLES
        for condition in POSITIONS
    }
    edge = statistics.mean(accuracies[f"kv_position_{slot}"] for slot in (0, 9))
    middle = statistics.mean(accuracies[f"kv_position_{slot}"] for slot in (4, 5))
    if edge - middle < MIN_EDGE_ADVANTAGE:
        raise ValueError(f"positive control failed: {json.dumps(accuracies, sort_keys=True)}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
