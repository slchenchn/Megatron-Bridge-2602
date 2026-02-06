set -xe

HF_PATH=/nfs/FM/chenshuailin/checkpoints/moonshotai/Moonlight-16B-A3B
MEGATRON_PATH=/nfs/FM/chenshuailin/projects/pretrain/Megatron-Bridge-2602/checkpoints/Moonlight-16B-A3B-bridge-mcore

torchrun --nproc_per_node=1 --nnodes=1 --master_port 29500 \
    examples/conversion/convert_checkpoints.py import \
    --hf-model $HF_PATH \
    --megatron-path $MEGATRON_PATH
