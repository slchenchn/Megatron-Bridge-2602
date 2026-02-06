#!/usr/bin/env bash

MODEL_NAME=Moonlight-16B-A3B
# MODEL_NAME=Qwen3-0.6B-Base

TP_SIZE=2
MICRO_BATCH_SIZE=1
PACK_SEQ=1
EXP_NAME=${MODEL_NAME}_SFT
EPOCH=1

if ((PACK_SEQ == 1)); then
    EXP_NAME=${EXP_NAME}_pack_TP${TP_SIZE}_e${EPOCH}
else
    EXP_NAME=${EXP_NAME}_unpack_TP${TP_SIZE}_e${EPOCH}
fi

if [[ -d /nfs/FM/chenshuailin ]]; then
    EXP_NAME=${EXP_NAME}_pt_official_ft_bf16
    # PRETRAINED_CHECKPOINT=/nfs/FM/checkpoints/Qwen/${MODEL_NAME}-bridge-mcore
    PRETRAINED_CHECKPOINT=checkpoints/Moonlight-16B-A3B-bridge-mcore
else
    exit 1
    EXP_NAME=${EXP_NAME}_fp8
    # PRETRAINED_CHECKPOINT=/mnt/md0/csl/code/Megatron-LM/checkpoints/Qwen/${MODEL_NAME}-bridge-mcore/TP1_PP1
    PRETRAINED_CHECKPOINT=/mnt/md0/csl/code/Megatron-LM/checkpoints/Qwen/Qwen3-8B-Base-bridge-mcore/TP1_PP1_data_aihub_te_ce_iter7152
fi

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR=nemo_experiments/${EXP_NAME}
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=14400
export NCCL_TIMEOUT=14400
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
torchrun --nproc_per_node=8 \
    examples/models/moonlight/finetune.py \
    --model-name "${MODEL_NAME}" \
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --tp "${TP_SIZE}" \
    --train-epochs "${EPOCH}" \
    --packed-sequence ${PACK_SEQ} \
    --enable-recompute \
    --exp-name "${EXP_NAME}" |&
    tee "${LOG_DIR}/train_${TIMESTAMP}.log"
