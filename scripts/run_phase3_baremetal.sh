#!/usr/bin/env bash
# Run a Phase 3 stage for one NVIDIA Megatron Mamba-2 checkpoint on the
# bare-metal venv scripts/setup_baremetal_megatron.sh provisions (no
# container). Same Megatron model arguments and stage names as
# run_phase3_in_container.sh; only the paths, interpreter, and the two
# Python-3.12 flags noted below differ.
#
# Env:
#   VENV      python interpreter of the Megatron venv (required)
#   MEGATRON  Megatron-LM checkout (default /root/Megatron-LM)
#   MIXING    this repository (default /root/mixing)
#   CKPT_ROOT checkpoints root, one subdir per model_key (default /root/checkpoints)
#   OUT_DIR   outputs (default /root/outputs)
#   MM_PIQA_SAMPLES / MM_QUESTIONS  optional
#
# Usage: run_phase3_baremetal.sh <mamba2-8b|mamba2-hybrid-8b> <validate|kv|sweep>
set -euo pipefail

MODEL_KEY="${1:?model_key required}"
STAGE="${2:?stage required}"
VENV="${VENV:?set VENV to the megatron venv python}"
MEGATRON="${MEGATRON:-/root/Megatron-LM}"
MIXING="${MIXING:-/root/mixing}"
CKPT_ROOT="${CKPT_ROOT:-/root/checkpoints}"
OUT_DIR="${OUT_DIR:-/root/outputs}"

case "${MODEL_KEY}" in
  mamba2-8b)
    ATTN_RATIO=0.0; MLP_RATIO=0.0
    REVISION=b915550c63ba9359f88f44d1f6a600d85af27302 ;;
  mamba2-hybrid-8b)
    ATTN_RATIO=0.08; MLP_RATIO=0.5
    REVISION=35e8852e2240b350ac2fe2a3b8aa341b5930018e ;;
  *) echo "unknown model_key ${MODEL_KEY}" >&2; exit 1 ;;
esac

CKPT_DIR="${CKPT_ROOT}/${MODEL_KEY}"
TOKENIZER_MODEL="${CKPT_DIR}/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
DATA_FILE="${MIXING}/data/nq-open-10_total_documents_gold_at_0.jsonl.gz"
PIQA_FILE="${MIXING}/data/piqa_valid.jsonl"
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${MEGATRON}:${MIXING}/src:${MIXING}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR="${OUT_DIR}/triton-cache/"
# Transformer Engine's own jit fuser and flash-attn backend hit the same
# Python 3.12 / TorchDynamo incompatibility scripts/setup_baremetal_megatron.sh
# works around in Megatron's jit.py; disable both here too.
export NVTE_TORCH_COMPILE=0
export NVTE_FLASH_ATTN=0

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
  --seed 42 --no-gradient-accumulation-fusion --load ${CKPT_DIR}"

DIST="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 29500"
TORCHRUN="$(dirname "${VENV}")/torchrun"
[ -x "${TORCHRUN}" ] || TORCHRUN="${VENV} -m torch.distributed.run"

cd "${MEGATRON}"

case "${STAGE}" in
  validate)
    ${TORCHRUN} ${DIST} "${MIXING}/scripts/megatron_validate.py" ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-piqa-file "${PIQA_FILE}" \
      --mm-piqa-samples "${MM_PIQA_SAMPLES:-1838}" ;;
  kv)
    ${TORCHRUN} ${DIST} "${MIXING}/scripts/megatron_sweep.py" ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-revision "${REVISION}" \
      --mm-kv-output "${OUT_DIR}/${MODEL_KEY}-positive-control.jsonl" ;;
  sweep)
    ${TORCHRUN} ${DIST} "${MIXING}/scripts/megatron_sweep.py" ${MEGATRON_ARGS} \
      --mm-model-key "${MODEL_KEY}" --mm-revision "${REVISION}" \
      --mm-data "${DATA_FILE}" --mm-questions "${MM_QUESTIONS:-800}" \
      --mm-sweep-output "${OUT_DIR}/${MODEL_KEY}-sweep.jsonl" ;;
  *) echo "unknown stage ${STAGE}" >&2; exit 1 ;;
esac
