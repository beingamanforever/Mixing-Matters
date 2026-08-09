#!/usr/bin/env bash
# Provision a bare-metal (no container) venv that runs NVIDIA Megatron-LM
# MambaModel inference on Python 3.12, matching the stack the pure Phase 3
# checkpoint was validated against: NGC nvcr.io/nvidia/pytorch:24.01-py3 ships
# torch 2.2 and triton 2.1.0, Megatron-LM at commit df61e60 pins mamba-ssm
# 2.0.3 and causal-conv1d 1.2.2.post1. This reproduces that trio from
# prebuilt wheels instead of the container, then adds a compiled
# transformer-engine (the mamba layer spec hard-requires it for its norm and
# attention-layer modules) and two small compatibility shims Python 3.12
# needs that the container's Python 3.10 does not.
set -euo pipefail

VENV=${VENV:-/root/mm-venv}
MEGATRON=${MEGATRON:-/root/Megatron-LM}
TE_VERSION=${TE_VERSION:-1.11.0}
MAX_JOBS=${MAX_JOBS:-4}

python3.12 -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
"$PIP" install -q -U pip wheel setuptools

echo "== torch 2.2.2 cu121 + triton 2.2.0 (prebuilt)"
"$PIP" install -q torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
"$PIP" install -q triton==2.2.0

echo "== causal-conv1d 1.2.2.post1 + mamba-ssm 2.0.3 (prebuilt cp312/torch2.2 wheels)"
"$PIP" install -q "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.2.post1/causal_conv1d-1.2.2.post1+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
"$PIP" install -q --no-deps "https://github.com/state-spaces/mamba/releases/download/v2.0.3/mamba_ssm-2.0.3+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

echo "== support libraries"
"$PIP" install -q "numpy<2" transformers==4.37.2 "sentencepiece>=0.2.0" flask-restful regex \
  "pydantic<3" six einops packaging "setuptools<81"

echo "== flash-attn (prebuilt cp312/torch2.2 wheel; only needed to satisfy TE's import, not used directly)"
"$PIP" install -q --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

echo "== transformer-engine ${TE_VERSION}: prebuilt core wheel, compiled torch bindings"
export NVTE_FRAMEWORK=pytorch
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
CUDNN_DIR=$("$PY" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CUDNN_PATH="$CUDNN_DIR" CUDNN_INCLUDE_DIR="$CUDNN_DIR/include" CUDNN_LIBRARY_PATH="$CUDNN_DIR/lib"
export CPLUS_INCLUDE_PATH="$CUDNN_DIR/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDNN_DIR/lib:${LIBRARY_PATH:-}"
"$PIP" install -q "transformer-engine==${TE_VERSION}"
nice -n 15 env MAX_JOBS="$MAX_JOBS" NVTE_FRAMEWORK=pytorch \
  "$PIP" install --no-build-isolation --no-deps --no-cache-dir "transformer-engine-torch==${TE_VERSION}"

echo "== apex stub: Megatron imports apex at module load for fused optimizers and"
echo "   fused-norm training kernels, none of which run during greedy inference"
echo "   (inference uses Transformer Engine norms and never builds an optimizer)."
echo "   Real apex needs a heavy source build; these stubs only satisfy the imports"
echo "   and raise if anything actually tries to call the training-only paths."
APEX_DIR="$VENV/lib/python3.12/site-packages/apex"
mkdir -p "$APEX_DIR"/{contrib/layer_norm,multi_tensor_apply,normalization,optimizers,transformer}
cat > "$APEX_DIR/__init__.py" <<'PY'
# Minimal apex stub for Megatron-LM INFERENCE on a bare-metal venv. Real NVIDIA
# apex requires a heavy source build; none of the paths that use apex (fused
# optimizers, apex fused-norm kernels, multi-tensor grad clipping) run during
# greedy inference, which uses Transformer Engine norms and never builds an
# optimizer. These stubs exist only to satisfy imports.
PY
: > "$APEX_DIR/contrib/__init__.py"
cat > "$APEX_DIR/contrib/layer_norm/__init__.py" <<'PY'
from .layer_norm import FastLayerNormFN  # noqa
PY
cat > "$APEX_DIR/contrib/layer_norm/layer_norm.py" <<'PY'
class FastLayerNormFN:  # noqa
    pass
PY
cat > "$APEX_DIR/multi_tensor_apply/__init__.py" <<'PY'
class _MultiTensorApplier:
    available = False

    def __call__(self, *a, **k):
        raise RuntimeError("apex stub multi_tensor_applier not available")


multi_tensor_applier = _MultiTensorApplier()
PY
cat > "$APEX_DIR/normalization/__init__.py" <<'PY'
from .fused_layer_norm import (  # noqa
    FusedLayerNorm,
    FusedLayerNormAffineFunction,
    MixedFusedLayerNorm,
    fused_layer_norm_affine,
)
PY
cat > "$APEX_DIR/normalization/fused_layer_norm.py" <<'PY'
def fused_layer_norm_affine(*a, **k):
    raise RuntimeError("apex stub fused_layer_norm_affine not available")


class FusedLayerNormAffineFunction:  # noqa
    pass


class FusedLayerNorm:  # noqa
    pass


class MixedFusedLayerNorm:  # noqa
    pass
PY
cat > "$APEX_DIR/optimizers/__init__.py" <<'PY'
class FusedAdam:  # never instantiated during inference
    def __init__(self, *a, **k):
        raise RuntimeError("apex stub FusedAdam is inference-only, not usable for training")


class FusedSGD:
    def __init__(self, *a, **k):
        raise RuntimeError("apex stub FusedSGD is inference-only, not usable for training")
PY
: > "$APEX_DIR/transformer/__init__.py"
cat > "$APEX_DIR/transformer/functional.py" <<'PY'
def fused_apply_rotary_pos_emb(*a, **k):
    raise RuntimeError("apex stub not available")


def fused_apply_rotary_pos_emb_thd(*a, **k):
    raise RuntimeError("apex stub not available")
PY

echo "== patch Megatron's jit fuser for Python 3.12"
echo "   megatron/core/jit.py sets jit_fuser = torch.compile for torch >= 2.2, but"
echo "   TorchDynamo does not support Python 3.12 in torch 2.2. The fuser is only an"
echo "   elementwise-fusion optimization the SSM math never depends on (that runs in"
echo "   the mamba-ssm CUDA kernels either way), so it is safe to no-op on 3.12."
cat > "$MEGATRON/megatron/core/jit.py" <<'PY'
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import sys

import torch

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


def _noop_fuser(fn):
    return fn


# torch.compile (TorchDynamo) is unsupported on Python 3.12 in torch 2.2, and
# torch.jit.script cannot script some of the fused helpers there. The fuser is
# only an elementwise-fusion perf optimization (the SSM math runs in mamba-ssm
# CUDA kernels regardless), so fall back to eager on 3.12.
if sys.version_info >= (3, 12):
    jit_fuser = _noop_fuser
elif (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 2):
    jit_fuser = torch.compile
else:
    jit_fuser = torch.jit.script
PY

echo "== verify the full stack"
NVTE_TORCH_COMPILE=0 NVTE_FLASH_ATTN=0 "$PY" -c "
import torch, mamba_ssm, causal_conv1d, transformers
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
print('torch', torch.__version__, 'mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)
import transformer_engine.pytorch as te
print('transformer_engine OK')
"
PYTHONPATH="$MEGATRON" NVTE_TORCH_COMPILE=0 NVTE_FLASH_ATTN=0 "$PY" -c "
from megatron.core.transformer.spec_utils import import_module
spec = import_module(('megatron.core.models.mamba.mamba_layer_specs', 'mamba_stack_spec'))
print('MAMBA_STACK_SPEC_OK', spec is not None)
"
echo "== bare-metal setup complete"
