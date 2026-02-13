#!/usr/bin/env bash
set -euo pipefail

# Reproduce v2 ingredient1 coverage with Moonlight tokenizer.
# - Source-level resume: completed sources are skipped.
# - Interrupted source is re-run in full.
# - Merge runs recursively over source outputs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/admin/csl/Dataset}"
INGREDIENT1_ROOT="${INGREDIENT1_ROOT:-${DATASET_ROOT}/allenai_dolma3_dolmino_mix-100B-1125-ingredient1}"
V2_TOKENIZED_ROOT="${V2_TOKENIZED_ROOT:-${DATASET_ROOT}/tokenized}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/home/admin/csl/checkpoints/moonshotai/Moonlight-16B-A3B}"

SOURCE_BASENAMES="${SOURCE_BASENAMES:-}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-dolma3_dolmino_mix-100B-1125-ingredient1_moonlight_v2_coverage}"
MAX_SHARDS="${MAX_SHARDS:-0}"  # unsupported in source-level mode; must remain 0
WORKERS="${WORKERS:-32}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
JSON_KEY="${JSON_KEY:-text}"

TMP_ROOT="${TMP_ROOT:-${DATASET_ROOT}/tokenized_moonlight_staging}"
FINAL_ROOT="${FINAL_ROOT:-${DATASET_ROOT}/tokenized_merged_moonlight}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/tokenization}"
ALLOW_REMERGE="${ALLOW_REMERGE:-1}"

RUN_ID="${RUN_ID:-20260209_v2_moonlight_full}"
RUN_DIR="${TMP_ROOT}/${OUTPUT_BASENAME}_${RUN_ID}"
TOKENIZED_DIR="${RUN_DIR}/tokenized_shards"
LOG_FILE="${LOG_DIR}/moonlight_tokenize_like_v2_${OUTPUT_BASENAME}_${RUN_ID}.log"
MANIFEST_FILE="${RUN_DIR}/source_manifest.txt"
FINAL_PREFIX="${FINAL_ROOT}/${OUTPUT_BASENAME}"

mkdir -p "${LOG_DIR}" "${TOKENIZED_DIR}" "${FINAL_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

FINAL_EXISTS=0
if [[ -f "${FINAL_PREFIX}.idx" || -f "${FINAL_PREFIX}.bin" ]]; then
    FINAL_EXISTS=1
    if [[ "${ALLOW_REMERGE}" != "1" ]]; then
        echo "[ERROR] Final output already exists: ${FINAL_PREFIX}.(idx|bin)"
        echo "        Set ALLOW_REMERGE=1 to replace final output during merge."
        exit 1
    fi
    echo "[INFO] Final output exists and will be replaced at merge step."
fi

collect_source_basenames() {
    if [[ -n "${SOURCE_BASENAMES}" ]]; then
        echo "${SOURCE_BASENAMES}" | tr ',' '\n' | sed '/^$/d' | sort -u
        return
    fi
    python - <<PY
import glob
import os
root = r"${V2_TOKENIZED_ROOT}"
pattern = os.path.join(root, 'allenai_dolma3_dolmino_mix-100B-1125-ingredient1-*_text_document.bin')
out = set()
for path in glob.glob(pattern):
    base = os.path.basename(path)
    if base.endswith('_text_document.bin'):
        out.add(base.replace('allenai_dolma3_dolmino_mix-100B-1125-', '').replace('_text_document.bin', ''))
for name in sorted(out):
    print(name)
PY
}

if [[ "${MAX_SHARDS}" != "0" ]]; then
    echo "[ERROR] MAX_SHARDS is not supported in source-level mode."
    echo "        Set MAX_SHARDS=0 (default) and rerun."
    exit 1
fi

mapfile -t SOURCES < <(collect_source_basenames)
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "[ERROR] No source basenames found. Set SOURCE_BASENAMES explicitly." >&2
    exit 1
fi

: > "${MANIFEST_FILE}"
for src_name in "${SOURCES[@]}"; do
    src_dir="${INGREDIENT1_ROOT}/${src_name}"
    if [[ ! -d "${src_dir}" ]]; then
        alt_dir="${DATASET_ROOT}/allenai_dolma3_dolmino_mix-100B-1125-${src_name}"
        if [[ -d "${alt_dir}" ]]; then
            src_dir="${alt_dir}"
        else
            mapfile -t found_dirs < <(find "${DATASET_ROOT}" -maxdepth 3 -type d -name "${src_name}" 2>/dev/null | sort)
            if [[ ${#found_dirs[@]} -gt 0 ]]; then
                src_dir="${found_dirs[0]}"
            fi
        fi
    fi
    if [[ ! -d "${src_dir}" ]]; then
        echo "[ERROR] Source directory missing for ${src_name}" >&2
        exit 1
    fi
    echo "${src_name}|${src_dir}" >> "${MANIFEST_FILE}"
done

echo "[INFO] Ingredient1 root : ${INGREDIENT1_ROOT}"
echo "[INFO] V2 tokenized root: ${V2_TOKENIZED_ROOT}"
echo "[INFO] Tokenizer model  : ${TOKENIZER_MODEL}"
echo "[INFO] Source count     : ${#SOURCES[@]}"
printf '[INFO] Sources          : %s\n' "${SOURCES[*]}"
echo "[INFO] Run dir          : ${RUN_DIR}"
echo "[INFO] Final prefix     : ${FINAL_PREFIX}"
echo "[INFO] Log file         : ${LOG_FILE}"
echo "[INFO] Manifest         : ${MANIFEST_FILE}"

total_input_files=0
for src_name in "${SOURCES[@]}"; do
    src_dir="$(awk -F'|' -v name="${src_name}" '$1==name {print $2; exit}' "${MANIFEST_FILE}")"
    count="$(find "${src_dir}" -type f \( -name "*.jsonl.zst" -o -name "*.jsonl" \) | wc -l)"
    total_input_files=$((total_input_files + count))
done
echo "[INFO] Input files      : ${total_input_files}"

processed_sources=0
for src_name in "${SOURCES[@]}"; do
    src_dir="$(awk -F'|' -v name="${src_name}" '$1==name {print $2; exit}' "${MANIFEST_FILE}")"
    mapfile -t src_files < <(find "${src_dir}" -type f \( -name "*.jsonl.zst" -o -name "*.jsonl" \) | sort)
    if [[ ${#src_files[@]} -eq 0 ]]; then
        echo "[ERROR] No .jsonl/.jsonl.zst files found for ${src_name}" >&2
        exit 1
    fi

    shard_dir="${TOKENIZED_DIR}/${src_name}"
    mkdir -p "${shard_dir}"
    source_prefix="${shard_dir}/${src_name}"
    source_done="${shard_dir}/.source_done"

    if [[ -f "${source_done}" && -f "${source_prefix}_text_document.idx" && -f "${source_prefix}_text_document.bin" ]]; then
        echo "[SKIP] ${src_name} already tokenized (source-level checkpoint)"
        continue
    fi

    processed_sources=$((processed_sources + 1))
    echo "[STEP 1][${processed_sources}] tokenizing source ${src_name} (${#src_files[@]} files)"
    find "${shard_dir}" -maxdepth 1 -type f \( -name "*_text_document.idx" -o -name "*_text_document.bin" \) -delete
    rm -f "${source_done}"

    python "${REPO_ROOT}/examples/tokenizationi/preprocess_data_hf.py" \
        --input "${src_dir}" \
        --output-prefix "${source_prefix}" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --trust-remote-code \
        --json-keys "${JSON_KEY}" \
        --workers "${WORKERS}" \
        --append-eod \
        --log-interval "${LOG_INTERVAL}" \
        --data-format json

    touch "${source_done}"
done

echo "[STEP 2] merging indexed shards"
if [[ ${FINAL_EXISTS} -eq 1 ]]; then
    echo "[STEP 2] removing existing final output before merge"
    rm -f "${FINAL_PREFIX}.idx" "${FINAL_PREFIX}.bin"
fi

python "${REPO_ROOT}/examples/tokenizationi/merge_indexed_dataset_recursive.py" \
    --input "${TOKENIZED_DIR}" \
    --output-prefix "${FINAL_PREFIX}"

echo "[DONE] Created: ${FINAL_PREFIX}.idx and ${FINAL_PREFIX}.bin"

