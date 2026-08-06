#!/usr/bin/env bash
# Provision a GPU host to run the Mixing-Matters sweeps.
# Usage: scripts/setup_gpu.sh [venv-directory]
# Run from the repository root. Safe to rerun: it reinstalls into the same venv.
#
# Two traps this script exists to avoid, both observed on the A10G host:
#
# 1. Installing causal-conv1d or mamba-ssm without --no-deps lets pip resolve
#    torch itself and upgrade it, which replaced a working torch 2.7.1+cu126
#    with torch 2.13.0+cu130 that the installed driver could not run.
# 2. The prebuilt kernel wheels published for "cu12 torch2.7" are compiled
#    against a CUDA 12.9 build of torch and fail to load against a cu126 build
#    with "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib".
#    Building from source against the installed torch avoids the mismatch.
set -euo pipefail

VENV=${VENV:-${1:-.venv}}
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_VERSION=${TORCH_VERSION:-2.7.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.57.1}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-1.5.3.post1}
MAMBA_SSM_VERSION=${MAMBA_SSM_VERSION:-2.2.6.post3}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [ ! -x "$PYTHON" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

echo "== installing pinned runtime"
"$PIP" install -q --no-cache-dir "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
"$PIP" install -q --no-cache-dir \
  "transformers==$TRANSFORMERS_VERSION" "huggingface_hub<1.0" "pydantic<3" \
  regex matplotlib pytest ninja packaging setuptools wheel einops

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA device, stopping before the kernel build")
print(f"torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}")
PY

# The kernel build needs nvcc. Some hosts ship the CUDA toolkit without it on
# PATH, so discover it under the standard install location before building.
if ! command -v nvcc >/dev/null 2>&1; then
  for candidate in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.6 /usr/local/cuda-12; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/nvcc" ]; then
      export CUDA_HOME="$candidate"
      export PATH="$candidate/bin:$PATH"
      break
    fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found, cannot build the Mamba kernels; install the CUDA toolkit" >&2
  exit 1
fi
echo "== using $(command -v nvcc): $(nvcc --version | grep -o 'release [0-9.]*')"

# The kernels are compiled for the exact architecture of this host, so the
# build stays short and cannot silently rely on a cubin for another generation.
ARCH=$("$PYTHON" -c "import torch; print('.'.join(str(part) for part in torch.cuda.get_device_capability()))")
echo "== building kernels for compute capability $ARCH with $BUILD_JOBS jobs"

TORCH_BEFORE=$("$PYTHON" -c "import torch; print(torch.__version__)")
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="$BUILD_JOBS"
CAUSAL_CONV1D_FORCE_BUILD=TRUE "$PIP" install --no-cache-dir --no-build-isolation --no-deps \
  "causal-conv1d==$CAUSAL_CONV1D_VERSION"
MAMBA_FORCE_BUILD=TRUE "$PIP" install --no-cache-dir --no-build-isolation --no-deps \
  "mamba-ssm==$MAMBA_SSM_VERSION"
TORCH_AFTER=$("$PYTHON" -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "torch changed during the kernel build: $TORCH_BEFORE -> $TORCH_AFTER" >&2
  exit 1
fi

echo "== verifying the execution path the sweep requires"
"$PYTHON" - <<'PY'
import causal_conv1d
import mamba_ssm
import torch
from transformers.models.mamba import modeling_mamba
from transformers.models.mamba2 import modeling_mamba2

update, convolution = modeling_mamba._lazy_load_causal_conv1d()
mamba_ready = all(
    (
        modeling_mamba.selective_state_update,
        modeling_mamba.selective_scan_fn,
        modeling_mamba.mamba_inner_fn,
        convolution,
        update,
    )
)
print(f"causal_conv1d {causal_conv1d.__version__} mamba_ssm {mamba_ssm.__version__}")
print(f"mamba kernel path {mamba_ready} mamba2 kernel path {modeling_mamba2.is_fast_path_available}")
print(f"capability {torch.cuda.get_device_capability()} memory {torch.cuda.get_device_properties(0).total_memory // 2**30} GiB")
if not (mamba_ready and modeling_mamba2.is_fast_path_available):
    raise SystemExit("the CUDA kernel path is unavailable, the sweep would refuse to run")
PY

echo "== prefetching the pinned checkpoints"
PYTHONPATH=src "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

from mixing_matters.models import MODELS

for spec in MODELS.values():
    path = snapshot_download(
        spec.repo,
        revision=spec.revision,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
    )
    print(f"{spec.key} {path}")
PY

echo "== setup complete"
