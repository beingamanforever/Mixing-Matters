"""Hidden-state extraction for the Phase 7 utilisation-vs-storage probe.

The probe question is whether a model *knows* where the gold document
sits even when it fails to *use* it. If a linear probe trained on frozen
hidden states can recover the gold position while QA accuracy is
U-shaped, the model stores the location but does not act on it.

This module runs a forward pass per (question, gold_position), captures
the last-token hidden state at one chosen layer, and writes one JSONL
record per (question, gold_position) carrying that vector. The probe
layer is fixed by the ``--layer`` CLI argument before the QA results
are inspected, honoring the "choose the probe layer before viewing
final results" requirement in the Phase 7 spec.

The heavy training and the shuffled-label control live in ``probe.py``
so the extraction (GPU) and the fit (CPU, seconds) stay decoupled and a
single extraction can be re-probed under different label schemes.
"""

import uuid
from pathlib import Path

from . import UPSTREAM_COMMIT, models
from .build_positions import place_gold
from .data import question_id, read_rows, split_indices
from .download import SHA256
from .io import write_jsonl
from .prompt_variants import build_variant_prompt
from .run import SEED, Generator, file_sha256


def _hidden_state_at_layer(model, tokenizer, prompt: str, layer: int) -> list[float]:
    """Return the last-token hidden state at ``layer`` as a python list.

    ``layer`` indexes ``outputs.hidden_states``: 0 is the embedding
    output, 1..N are the block outputs. The last prompt position is used,
    matching where the model would begin generating the answer.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda") if "attention_mask" in inputs else None
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model.forward did not return hidden_states")
    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(
            f"probe layer {layer} out of range for {len(hidden_states)} hidden-state tensors"
        )
    vector = hidden_states[layer][0, -1, :]
    return [float(value) for value in vector.to(torch.float32).cpu().tolist()]


def run_probe_scan(
    data_path: Path,
    output: Path,
    model_key: str,
    revision: str,
    layer: int,
    questions: int = 200,
) -> None:
    """Write one hidden-state record per (question, gold_position).

    Each record carries::

        {
            "run_id": str,
            "model_key": str,
            "family": str,
            "question_id": str,
            "source_index": int,
            "gold_position": int,
            "layer": int,
            "hidden_state": [float, ...],
            "prompt_token_count": int,
            "seed": int,
            "data_sha256": str,
        }

    ``layer`` is fixed by the caller and recorded on every line so a
    downstream probe cannot silently pick a layer after seeing results.
    """
    from lost_in_the_middle.prompting import Document

    model_spec = models.spec(model_key)
    if output.exists():
        raise FileExistsError(output)
    digest = file_sha256(data_path)
    if digest != SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    rows = read_rows(data_path)
    generator = Generator(model_spec, revision)

    exploratory, _ = split_indices(len(rows), SEED)
    selected = [(index, rows[index]) for index in exploratory[:questions]]
    run_id = str(uuid.uuid4())

    def records():
        for source_index, row in selected:
            qid = question_id(row, source_index)
            for position in range(10):
                documents = place_gold(row, position)["ctxs"]
                prompt = build_variant_prompt(
                    row["question"],
                    [Document.from_dict(document) for document in documents],
                    variant="baseline",
                )
                prompt_tokens = int(
                    generator.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                )
                vector = _hidden_state_at_layer(generator.model, generator.tokenizer, prompt, layer)
                yield {
                    "run_id": run_id,
                    "model_key": model_key,
                    "family": model_spec.family,
                    "model": generator.metadata["model"],
                    "model_revision": generator.metadata["model_revision"],
                    "question_id": qid,
                    "source_index": source_index,
                    "gold_position": position,
                    "layer": layer,
                    "hidden_state": vector,
                    "prompt_token_count": prompt_tokens,
                    "seed": SEED,
                    "data_revision": UPSTREAM_COMMIT,
                    "data_sha256": digest,
                }

    write_jsonl(output, records())
