"""Attention-sink mass measurement for the Phase 7 sink hypothesis test.

The sink hypothesis says the primacy arm is downstream of a learned
attention sink at the first prompt token in dense-attention causal
transformers, and (under the hybrid variant) the sub-set of attention
layers in a Nemotron-H style hybrid model. This module measures, on
frozen forward passes over gold-condition prompts, the fraction of
attention mass that lands on token 0 at each attention layer.

It never generates any tokens: it runs a single forward pass with
``output_attentions=True``, extracts the per-layer per-head attention
weights at the last prompt position, averages across heads, and writes
one JSONL record per (question, gold_position, layer) with the
sink-mass value alongside the model_key and family. The consumer joins
these records back to the Phase 2 or Phase 4 sweeps by ``question_id``
and ``gold_position`` to correlate sink mass with primacy per question.

Only dense-attention families (``pythia``, ``llama``, ``qwen2``,
``nemotron-h``) expose attention weights the transformers API can
return. Pure-SSM families (``mamba``, ``mamba2``) raise a ``ValueError``
here rather than silently emitting zeros.
"""

import json
import uuid
from pathlib import Path

from . import UPSTREAM_COMMIT, models
from .anchors import build_prompt
from .build_positions import place_gold
from .data import question_id, read_rows
from .download import SHA256
from .io import write_jsonl
from .prompt_variants import build_variant_prompt
from .run import (
    MAX_NEW_TOKENS,
    SEED,
    Generator,
    _installed_version,
    _resolve_driver_version,
    file_sha256,
)

_ATTENTION_FAMILIES = ("pythia", "llama", "qwen2", "nemotron-h")


def _sink_mass_per_layer(model, tokenizer, prompt: str) -> list[float]:
    """Return the token-0 attention share per attention layer.

    Reads ``model.config.output_attentions``-guarded attentions for
    the encoded prompt, takes the last prompt-position row of each
    layer's attention matrix, averages across heads, and returns one
    float per layer. Guaranteed to be numerically in ``[0, 1]``.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda") if "attention_mask" in inputs else None
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError(
            "model.forward did not return attentions; the requested family "
            "cannot report sink mass through this path"
        )
    per_layer = []
    last_index = int(input_ids.shape[1]) - 1
    for layer_attn in attentions:
        # layer_attn shape: (batch, heads, query, key)
        row = layer_attn[0, :, last_index, :]  # (heads, key)
        sink_share = row[:, 0].mean().item()
        per_layer.append(float(sink_share))
    return per_layer


def run_sink_scan(
    data_path: Path,
    output: Path,
    model_key: str,
    revision: str,
    questions: int = 200,
    prompt_variant: str = "baseline",
    gold_padded_tokens: int = 0,
) -> None:
    """Write per-(question, gold_position, layer) sink-mass records.

    Each output JSONL line carries::

        {
            "run_id": str,
            "model_key": str,
            "family": str,
            "model": str,
            "model_revision": str,
            "question_id": str,
            "source_index": int,
            "condition": "gold",
            "gold_position": int,
            "layer": int,
            "sink_mass": float,      # [0, 1], token-0 attention share
            "prompt_token_count": int,
            "prompt_variant": str,
            "gold_padded_tokens": int,
            "seed": int,
            "software_versions": {...},
            "data_sha256": str,
        }

    ``run_sink_scan`` sweeps the first ``questions`` exploratory questions,
    all ten gold positions per question, with the same seed and split as
    ``run_sweep``. It never generates any tokens; the model is only used
    for its attention weights.
    """
    from lost_in_the_middle.prompting import Document

    from .data import split_indices

    model_spec = models.spec(model_key)
    if model_spec.family not in _ATTENTION_FAMILIES:
        raise ValueError(
            f"sink-mass scan requires a dense-attention family; {model_key!r} "
            f"has family {model_spec.family!r}"
        )
    if output.exists():
        raise FileExistsError(output)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    rows = read_rows(data_path)

    generator = Generator(model_spec, revision)
    # Ensure the model returns attentions on this run even if the config default is off.
    if hasattr(generator.model.config, "output_attentions"):
        generator.model.config.output_attentions = True

    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:questions]]

    run_id = str(uuid.uuid4())
    software_versions = {
        "python": generator.metadata["python"],
        "torch": generator.metadata["torch"],
        "transformers": generator.metadata["transformers"],
        "cuda": generator.metadata["cuda"],
        "driver": _resolve_driver_version(__import__("torch")),
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

    def records():
        for source_index, row in selected:
            qid = question_id(row, source_index)
            for position in range(10):
                documents = place_gold(row, position)["ctxs"]
                prompt = build_variant_prompt(
                    row["question"],
                    [Document.from_dict(document) for document in documents],
                    variant=prompt_variant,
                    gold_padded_tokens=gold_padded_tokens,
                )
                # Prompt-token count for the reader; matches the sweep field.
                prompt_tokens = int(generator.tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
                sink_mass = _sink_mass_per_layer(generator.model, generator.tokenizer, prompt)
                for layer, mass in enumerate(sink_mass):
                    yield {
                        "run_id": run_id,
                        "model_key": model_key,
                        "family": model_spec.family,
                        "model": generator.metadata["model"],
                        "model_revision": generator.metadata["model_revision"],
                        "question_id": qid,
                        "source_index": source_index,
                        "condition": "gold",
                        "gold_position": position,
                        "layer": layer,
                        "sink_mass": mass,
                        "prompt_token_count": prompt_tokens,
                        "prompt_variant": prompt_variant,
                        "gold_padded_tokens": gold_padded_tokens,
                        "seed": SEED,
                        "software_versions": software_versions,
                        "data_revision": UPSTREAM_COMMIT,
                        "data_sha256": digest,
                    }

    write_jsonl(output, records())


def sink_mass_summary(records: list[dict]) -> dict:
    """Aggregate per-(model, gold_position, layer) mean sink-mass.

    Returns::

        {
            "models": [str, ...],
            "by_model": {
                model_key: {
                    "layers": [int, ...],
                    "positions": {
                        position: {"mean_sink_mass_per_layer": [float, ...]}
                    },
                    "question_count": int,
                }
            }
        }
    """
    grouped: dict[str, dict[int, dict[int, list[float]]]] = {}
    questions: dict[str, set[str]] = {}
    for record in records:
        model = record["model_key"]
        position = int(record["gold_position"])
        layer = int(record["layer"])
        grouped.setdefault(model, {}).setdefault(position, {}).setdefault(layer, []).append(
            float(record["sink_mass"])
        )
        questions.setdefault(model, set()).add(record["question_id"])
    by_model = {}
    for model, position_map in grouped.items():
        layers = sorted({layer for layer_map in position_map.values() for layer in layer_map})
        positions_out = {}
        for position, layer_map in sorted(position_map.items()):
            per_layer = [
                sum(layer_map.get(layer, [])) / len(layer_map.get(layer, [1]))
                if layer in layer_map
                else 0.0
                for layer in layers
            ]
            positions_out[position] = {"mean_sink_mass_per_layer": per_layer}
        by_model[model] = {
            "layers": layers,
            "positions": positions_out,
            "question_count": len(questions.get(model, set())),
        }
    return {"models": sorted(by_model), "by_model": by_model}
