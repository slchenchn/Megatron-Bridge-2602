#!/usr/bin/env bash

# Load login environment (e.g. CUDA paths) even in non-login shells.
if [[ -f "${HOME}/.profile" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.profile"
fi

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <precision: f16|f8> <optimizer: adam|dist_muon>"
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

# ===== Fixed training config (edit here when needed) =====
MODEL_NAME="Moonlight-16B-A3B"
PRETRAINED_CHECKPOINT="checkpoints/Moonlight-16B-A3B-bridge-mcore"
TOKENIZER_PATH="/home/admin/csl/checkpoints/moonshotai/${MODEL_NAME}"
DATA_PATH="/home/admin/csl/Dataset/tokenized_merged_moonlight/dolma3_dolmino_mix-100B-1125-ingredient1_moonlight_v2_coverage"

TP_SIZE=1
CP_SIZE=1
EP_SIZE=8
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024
TRAIN_ITERS=7500    # 30B / 4M = 7500
SEQ_LEN=4096
LR=1.5e-4
MIN_LR=1.5e-6
LR_WARMUP_ITERS=8
LR_DECAY_STYLE="WSD"
LR_WSD_DECAY_ITERS=750

EXP_NAME="${MODEL_NAME}_pt_${PRECISION_INPUT}_${OPTIMIZER_TYPE}_TP${TP_SIZE}"
CHECKPOINT_SAVE_DIR="nemo_experiments/${EXP_NAME}/checkpoints"

LOG_DIR="nemo_experiments/${EXP_NAME}"
mkdir -p "${LOG_DIR}" "$(dirname "${CHECKPOINT_SAVE_DIR}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 26000 \
    examples/models/moonlight/pretrain.py \
    --model-name "${MODEL_NAME}" \
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --data-path "${DATA_PATH}" \
    --optimizer-type "${OPTIMIZER_TYPE}" \
    --precision-config "${PRECISION_CONFIG}" \
    --train-iters "${TRAIN_ITERS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --seq-length "${SEQ_LEN}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --lr-warmup-iters "${LR_WARMUP_ITERS}" \
    --lr-decay-style "${LR_DECAY_STYLE}" \
    --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS}" \
    --tp "${TP_SIZE}" \
    --cp "${CP_SIZE}" \
    --ep "${EP_SIZE}" \
    --save "${CHECKPOINT_SAVE_DIR}" \
    --exp-name "${EXP_NAME}" |& tee "${LOG_DIR}/train_${TIMESTAMP}.log"

    # --token-drop \
    # --eval-interval 5 \
    # --save-interval 5 \