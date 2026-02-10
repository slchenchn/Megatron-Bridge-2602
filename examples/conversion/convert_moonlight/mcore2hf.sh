# set -x


MCORE_DIR=$1

if [[ -d /nfs/FM/chenshuailin ]]; then
  # aihub
  HF_MODEL=/nfs/FM/chenshuailin/checkpoints/moonshotai/Moonlight-16B-A3B
else
  # SJHL
  HF_MODEL=/home/admin/csl/checkpoints/moonshotai/Moonlight-16B-A3B/
fi

HF_OUTPUT_DIR=${MCORE_DIR}/hf_model
if [[ -d "${HF_OUTPUT_DIR}" ]]; then
  echo "Warning: output hf_model dir already exists: ${HF_OUTPUT_DIR}. Remove it or choose another path, then re-run." >&2
  exit 1
fi

python examples/conversion/convert_checkpoints.py export \
  --hf-model ${HF_MODEL} \
  --megatron-path ${MCORE_DIR} \
  --hf-path ${HF_OUTPUT_DIR}