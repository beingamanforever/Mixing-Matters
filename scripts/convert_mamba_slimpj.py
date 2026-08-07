"""Convert the SlimPajama Mamba checkpoint to an HF-format directory.

``state-spaces/mamba-2.8b-slimpj`` is published only in the original
state-spaces format, which transformers cannot load. This converts it once to
an HF ``MambaForCausalLM`` directory so the Phase 5 sweep loads it on the same
transformers CUDA-kernel path as the Pile checkpoint, leaving the training
corpus as the only thing that differs across the contrast.

Usage, from the repository root with the venv built by scripts/setup_gpu.sh:
    PYTHONPATH=src .venv/bin/python scripts/convert_mamba_slimpj.py

Set MIXING_MATTERS_CONVERTED_DIR to place the ~5 GB output on a roomy volume.
Validate it before trusting the weights:
    PYTHONPATH=src .venv/bin/python scripts/validate_mamba_slimpj_conversion.py
"""

import json

from mixing_matters.convert import convert
from mixing_matters.models import spec


def main() -> None:
    model_spec = spec("mamba-2.8b-slimpj")
    output = convert(model_spec)
    manifest = json.loads((output / "conversion-manifest.json").read_text())
    print(f"converted {model_spec.key} -> {output}")
    print(f"source {manifest['source_repo']}@{manifest['source_revision']}")
    print(f"weights {manifest['weight_files']}")
    for name, digest in manifest["weight_sha256"].items():
        print(f"  {name} sha256 {digest}")


if __name__ == "__main__":
    main()
