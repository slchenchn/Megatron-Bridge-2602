# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_official_ft_f16_adam_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_official_ft_f8_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_f16_dist_muon_TP1/checkpoints/iter_0007500
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_muon_f16_ft_f16_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_official_ft_f8_adam_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_muon_f8_ft_f8_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_muon_f16_ft_f8_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_adam_f16_ft_f16_adam_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_adam_f8_ft_f8_adam_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_f8_dist_muon_TP1/checkpoints/iter_0007500
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_f16_adam_TP1/checkpoints/iter_0007500
# MCORE_DIR=nemo_experiments/Moonlight-16B-A3B_pt_adam_f16_ft_f8_adam_pack_TP1_e1/checkpoints/iter_0017165
# MCORE_DIR=nemo_experiments/finetune/Moonlight-16B-A3B_pt_adam_f16_ft_f16_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
MCORE_DIR=nemo_experiments/finetune/Moonlight-16B-A3B_pt_adam_f8_ft_f8_dist_muon_pack_TP1_e1/checkpoints/iter_0017165
HF_DIR=${MCORE_DIR}/hf_model
MODEL_NAME=$(basename "$(dirname "$(dirname "${MCORE_DIR}")")")
REPO_ID=chenda/${MODEL_NAME}

echo "=== Step 1: Convert mcore to HuggingFace ==="
bash examples/conversion/convert_moonlight/mcore2hf.sh ${MCORE_DIR}
echo ""
echo "=== Step 2: Validate tokenizer config ==="
python scripts/validate_tokenizer_config.py ${HF_DIR}
echo ""

echo "=== Step 3: Upload to ModelScope ==="
python scripts/modelscope_transit.py ${REPO_ID} ${HF_DIR}
echo ""
