#!/usr/bin/env bash

# Load login environment (e.g. CUDA paths) even when invoked from a
# non-login / non-interactive context (ssh "cmd", tmux, sudo, etc.).
# This keeps Triton/Inductor compilation from failing on missing cuda.h/ptxas.
if [[ -f "${HOME}/.profile" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.profile"
fi

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <precision: f16|f8> <optimizer: adam|muon>"
    exit 1
fi

PRECISION_INPUT="$1"
OPTIMIZER_TYPE="$2"

case "${PRECISION_INPUT}" in
    f16)
        PRECISION_CONFIG="bf16_mixed"
        ;;
    f8)
        PRECISION_CONFIG="bf16_with_fp8_subchannel_scaling_mixed"
        ;;
    *)
        echo "Invalid precision: ${PRECISION_INPUT}. Use f16 or f8."
        exit 1
        ;;
esac

case "${OPTIMIZER_TYPE}" in
    adam|dist_muon)
        ;;
    *)
        echo "Invalid optimizer: ${OPTIMIZER_TYPE}. Use adam or dist_muon."
        exit 1
        ;;
esac

MODEL_NAME=Moonlight-16B-A3B

TP_SIZE=1
MICRO_BATCH_SIZE=1
PACK_SEQ=1
EPOCH=1
SAVE_ROOT=nemo_experiments/finetune
PRETRAINED_CHECKPOINT=${SAVE_ROOT}/pretrain/Moonlight-16B-A3B_pt_f16_dist_muon_TP1/checkpoints
EXP_NAME=finetune/${MODEL_NAME}_pt_muon_f16_ft_${PRECISION_INPUT}_${OPTIMIZER_TYPE}

if ((PACK_SEQ == 1)); then
    EXP_NAME=${EXP_NAME}_pack_TP${TP_SIZE}_e${EPOCH}
else
    EXP_NAME=${EXP_NAME}_unpack_TP${TP_SIZE}_e${EPOCH}
fi

LOG_DIR=${SAVE_ROOT}/${EXP_NAME}
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=14400
# export NCCL_TIMEOUT=14400
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 \
    examples/models/moonlight/finetune.py \
    --model-name "${MODEL_NAME}" \
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --tp "${TP_SIZE}" \
    --train-epochs "${EPOCH}" \
    --packed-sequence ${PACK_SEQ} \
    --optimizer-type "${OPTIMIZER_TYPE}" \
    --precision-config "${PRECISION_CONFIG}" \
    --exp-name "${EXP_NAME}" |&
    tee "${LOG_DIR}/train_${TIMESTAMP}.log"
