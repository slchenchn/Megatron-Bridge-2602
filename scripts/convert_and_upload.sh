MCORE_dir=nemo_experiments/Moonlight-16B-A3B_pt_official_ft_f16_adam_pack_TP1_e1/checkpoints/iter_0017165
HF_DIR=${MCORE_dir}/hf_model
REPO_ID=chenda/Moonlight-16B-A3B_pt_official_ft_f16_adam_pack_TP1_e1

bash examples/conversion/convert_moonlight/mcore2hf.sh ${MCORE_dir}

python scripts/validate_tokenizer_config.py ${HF_DIR}

python scripts/modelscope_transit.py ${REPO_ID} ${HF_DIR}
