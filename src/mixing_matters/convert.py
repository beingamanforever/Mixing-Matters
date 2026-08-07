"""Convert an original state-spaces Mamba checkpoint to an HF-format directory.

Phase 5 holds the Mamba architecture fixed and changes only the pretraining
corpus, comparing ``state-spaces/mamba-2.8b-hf`` (the Pile) against
``state-spaces/mamba-2.8b-slimpj`` (SlimPajama). The Pile checkpoint ships in
transformers format, but the SlimPajama one is published only in the original
state-spaces format: its config.json carries ``d_model``/``n_layer`` and no
``model_type``, so ``AutoModelForCausalLM`` cannot load it.

To keep every Mamba run on the one transformers CUDA-kernel path -- so the
only thing that differs across the contrast is the training data, not the
execution path -- this module converts the original checkpoint into an
HF-format ``MambaForCausalLM`` directory ahead of the sweep. The conversion
mirrors the mapping transformers ships in
``convert_mamba_ssm_checkpoint_to_pytorch.py``: the config is rebuilt from the
state-spaces fields and the sole weight rename is ``backbone.embedding`` ->
``backbone.embeddings`` (the LM head is tied to the embedding).

The numerical equivalence of the converted checkpoint to the original run
through the authors' own ``mamba_ssm`` implementation is not assumed here; it
is measured by ``scripts/validate_mamba_slimpj_conversion.py`` before the
converted weights are trusted, exactly as the Phase 2 Mamba-2 conversion was.
"""

import hashlib
import json
import math
import os
from pathlib import Path

from . import models

# The converted checkpoints can be large (a 2.8B model is ~5 GB in bf16), so
# their location is configurable: on a host whose root disk is small, point
# this at a roomier volume. Defaults to ``converted/`` under the repository.
CONVERTED_ROOT_ENV = "MIXING_MATTERS_CONVERTED_DIR"
# The state-spaces Mamba checkpoints carry no tokenizer; the HF conversions use
# the GPT-NeoX-20B tokenizer, the same one every other model in this study
# shares. Pulling it from the matched Pile checkpoint guarantees the SlimPajama
# arm tokenizes prompts identically to the Pile arm.
TOKENIZER_SOURCE_KEY = "mamba-2.8b"


def converted_root() -> Path:
    return Path(os.environ.get(CONVERTED_ROOT_ENV, "converted"))


def converted_dir(spec: models.ModelSpec) -> Path:
    """Where the HF-format conversion of a state-spaces checkpoint lives."""
    return converted_root() / spec.key


def _hf_config(config_ssm: dict):
    """Rebuild an HF MambaConfig from state-spaces config fields.

    Follows transformers' own ``convert_ssm_config_to_hf_config``: the state,
    expand, and convolution sizes keep the MambaConfig defaults the original
    architecture uses, the time-step rank is ceil(d_model / 16), and the vocab
    is padded up to ``pad_vocab_size_multiple`` so it matches the embedding
    matrix stored in the checkpoint.
    """
    from transformers import MambaConfig

    vocab_size = config_ssm["vocab_size"]
    pad = config_ssm["pad_vocab_size_multiple"]
    if vocab_size % pad != 0:
        vocab_size += pad - (vocab_size % pad)

    return MambaConfig(
        vocab_size=vocab_size,
        hidden_size=config_ssm["d_model"],
        intermediate_size=config_ssm["d_model"] * 2,
        num_hidden_layers=config_ssm["n_layer"],
        time_step_rank=math.ceil(config_ssm["d_model"] / 16),
        residual_in_fp32=config_ssm.get("residual_in_fp32", True),
    )


def _rename_state_dict(state_dict: dict) -> dict:
    """Rename original Mamba weights to their HF ``MambaForCausalLM`` names.

    The only difference in the parameter names is the embedding: state-spaces
    stores ``backbone.embedding.weight`` and transformers expects
    ``backbone.embeddings.weight``. Every mixer, norm, and head parameter
    already shares a name across the two implementations.
    """
    return {key.replace("embedding.", "embeddings."): value for key, value in state_dict.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert(spec: models.ModelSpec, output: Path | None = None) -> Path:
    """Convert a ``format="mamba_ssm"`` checkpoint to an HF directory.

    Downloads the original checkpoint at its pinned revision, rebuilds the HF
    config, renames and loads the weights with ``strict=True`` so any missing
    or unexpected parameter is a loud failure rather than a silent one, and
    writes the model, the shared tokenizer, and a manifest recording the exact
    source revision and the converted-weight checksum. Refuses to overwrite an
    existing conversion so a raw run artifact is never clobbered.

    Returns the directory the converted checkpoint was written to.
    """
    import torch
    import transformers
    from huggingface_hub import hf_hub_download, model_info
    from transformers import AutoTokenizer, MambaForCausalLM

    if spec.format != "mamba_ssm":
        raise ValueError(f"{spec.key} is already in HF format; no conversion needed")

    output = output or converted_dir(spec)
    if output.exists():
        raise FileExistsError(output)

    exact_revision = model_info(spec.repo, revision=spec.revision).sha
    if not exact_revision:
        raise ValueError("Hugging Face did not resolve an exact model revision")

    config_path = hf_hub_download(spec.repo, "config.json", revision=exact_revision)
    config_ssm = json.loads(Path(config_path).read_text())
    hf_config = _hf_config(config_ssm)

    weights_path = hf_hub_download(spec.repo, "pytorch_model.bin", revision=exact_revision)
    original_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    renamed = _rename_state_dict(original_state_dict)

    model = MambaForCausalLM(hf_config)
    expected = set(model.state_dict())
    # The LM head is tied to the embedding, so an original checkpoint that omits
    # it is expected; supply the tied weight rather than failing the strict load.
    if "lm_head.weight" in expected and "lm_head.weight" not in renamed:
        renamed["lm_head.weight"] = renamed["backbone.embeddings.weight"]
    missing = expected - set(renamed)
    unexpected = set(renamed) - expected
    if missing or unexpected:
        raise ValueError(
            f"state dict mismatch converting {spec.key}: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(renamed, strict=True)
    model = model.to(torch.bfloat16)

    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)

    tokenizer_spec = models.spec(TOKENIZER_SOURCE_KEY)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec.repo, revision=tokenizer_spec.revision)
    tokenizer.save_pretrained(output)

    weight_files = sorted(path.name for path in output.glob("*.safetensors"))
    manifest = {
        "model_key": spec.key,
        "source_repo": spec.repo,
        "source_revision": exact_revision,
        "source_config": config_ssm,
        "tokenizer_repo": tokenizer_spec.repo,
        "tokenizer_revision": tokenizer_spec.revision,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "torch.bfloat16",
        "hf_config": hf_config.to_dict(),
        "weight_files": weight_files,
        "weight_sha256": {name: _file_sha256(output / name) for name in weight_files},
    }
    (output / "conversion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
