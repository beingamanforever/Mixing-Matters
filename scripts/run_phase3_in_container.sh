#!/usr/bin/env bash
# Run a Phase 3 stage for one NVIDIA Megatron Mamba-2 checkpoint, inside the
# NVIDIA Megatron-LM container. Mounted layout expected:
#   /workspace/megatron      -> Megatron-LM @ df61e60
#   /workspace/mixing        -> this repository
#   /workspace/checkpoints   -> downloaded checkpoints (per-model subdirs)
#   /workspace/outputs       -> writable output directory
#   /workspace/piqa_valid.jsonl (optional, for validation)
#
# Usage: run_phase3_in_container.sh <model_key> <stage>
#   model_key : mamba2-8b | mamba2-hybrid-8b
#   stage     : validate | kv | sweep
set -euo pipefail

MODEL_KEY="${1:?model_key required}"
STAGE="${2:?stage required}"

case "${MODEL_KEY}" in
  mamba2-8b)
    ATTN_RATIO=0.0; MLP_RATIO=0.0
    REVISION=b915550c63ba9359f88f44d1f6a600d85af27302 ;;
  mamba2-hybrid-8b)
    ATTN_RATIO=0.08; MLP_RATIO=0.5
    REVISION=35e8852e2240b350ac2fe2a3b8aa341b5930018e ;;
  *) echo "unknown model_key ${MODEL_KEY}" >&2; exit 1 ;;
esac

CKPT_DIR="/workspace/checkpoints/${MODEL_KEY}"
TOKENIZER_MODEL="${CKPT_DIR}/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
DATA_FILE="/workspace/mixing/data/nq-open-10_total_documents_gold_at_0.jsonl.gz"
OUT_DIR="/workspace/outputs"
mkdir -p "${OUT_DIR}"

# scoring and the vendored prompt builders import regex and pydantic; install
# once into the container's python if missing.
python -c "import regex" 2>/dev/null || pip install -q regex
python -c "import pydantic" 2>/dev/null || pip install -q "pydantic<3"

export PYTHONPATH="/workspace/megatron:/workspace/mixing/src:/workspace/mixing:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR="/workspace/outputs/triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

MEGATRON_ARGS="--tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --untie-embeddings-and-output-weights \
  --num-layers 56 --hidden-size 4096 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --hybrid-attention-ratio ${ATTN_RATIO} --hybrid-mlp-ratio ${MLP_RATIO} \
  --attention-dropout 0.0 --hidden-dropout 0.0 --disable-bias-linear \
  --normalization RMSNorm --seq-length 4096 --max-position-embeddings 4096 \
  --position-embedding-type none \
  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ${TOKENIZER_MODEL} \
  --distributed-backend nccl --distributed-timeout-minutes 1440 \
  --bf16 --micro-batch-size 1 --use-mcore-models \
  --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
  --seed 42 --load ${CKPT_DIR}"

DIST="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 29500"

cd /workspace/megatron

case "${STAGE}" in
  validate)
    torchrun ${DIST} /workspace/mixing/scripts/megatron_validate.py ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-piqa-file /workspace/piqa_valid.jsonl \
      --mm-piqa-samples "${MM_PIQA_SAMPLES:-1838}"
    ;;
  kv)
    torchrun ${DIST} /workspace/mixing/scripts/megatron_sweep.py ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-revision "${REVISION}" \
      --mm-kv-output "${OUT_DIR}/${MODEL_KEY}-positive-control.jsonl"
    ;;
  sweep)
    torchrun ${DIST} /workspace/mixing/scripts/megatron_sweep.py ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-revision "${REVISION}" \
      --mm-data "${DATA_FILE}" --mm-questions "${MM_QUESTIONS:-800}" \
      --mm-sweep-output "${OUT_DIR}/${MODEL_KEY}-sweep.jsonl"
    ;;
  *) echo "unknown stage ${STAGE}" >&2; exit 1 ;;
esac
