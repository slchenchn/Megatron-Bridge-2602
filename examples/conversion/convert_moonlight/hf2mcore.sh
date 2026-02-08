set -xe

if [[ -d /nfs/FM/chenshuailin ]]; then
    # A800
    HF_PATH=/nfs/FM/chenshuailin/checkpoints/moonshotai/Moonlight-16B-A3B
    MEGATRON_PATH=/nfs/FM/chenshuailin/projects/pretrain/Megatron-Bridge-2602/checkpoints/Moonlight-16B-A3B-bridge-mcore
elif [[ -d /home/admin/csl ]]; then
    # H200
    HF_PATH=/home/admin/csl/checkpoints/moonshotai/Moonlight-16B-A3B/
    MEGATRON_PATH=/home/admin/csl/code/Megatron-Bridge-2602/checkpoints/Moonlight-16B-A3B-bridge-mcore
else
    echo "Error: Invalid environment"
    exit 1
fi

torchrun --nproc_per_node=1 --nnodes=1 --master_port 29500 \
    examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_PATH}" \
    --megatron-path "${MEGATRON_PATH}"
