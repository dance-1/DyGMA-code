#!/usr/bin/env bash
set -Eeuo pipefail

# Unified pretrained inference entry. Every test exports sensor rankings;
# SWaT and WADI use train-distribution per-sensor Z-score rankings for RCA.

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
DATA_ROOT="${DATA_ROOT:-dataset}"
LOG_DIR="${OUTPUT_DIR}/logs"
RCA_OUTPUT_DIR="${OUTPUT_DIR}/rca"

EPOCHS=15
WIN_SIZE=100
PATCH_LEN=1
BATCH_SIZE=32

declare -A INPUT_DIMS=(
  [MSL]=55
  [SMD]=38
  [PSM]=25
  [SWaT]=51
  [WADI]=123
  [HAI]=86
)

declare -A CHECKPOINT_HASHES=(
  [MSL]=84EF92AFEC3B22F49CA0D160B10A527225BBE532DB42B5E5D6636095FAF12B31
  [SMD]=32E8CC0FBE699F6FA7653C7EA48399E47CA45EF22FAE1EA9F3F74CB3E0D9793B
  [PSM]=DE630874509FCB632A987DE9C8F8DCF44E3EC9F8D397BDB38C51BA0F72DB2B4B
  [SWaT]=7E6355E4DA21732706FDB3D6CED7D3A891573340943A113FF7109B6D2630CA56
  [WADI]=38E05FB132D2816B842F1E42B6001A8EF525466AEA4DDF85A79C61F8A89EA4EE
  [HAI]=4E2D2199F514A941032BCD3AF8FF5701E58354E2611BC1767FE998B5862316C1
)

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  DATASETS=(MSL SMD PSM SWaT WADI HAI)
else
  DATASETS=("$@")
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RCA_OUTPUT_DIR}"

run_dataset() {
  local dataset="$1"
  local input_c="${INPUT_DIMS[$dataset]:-}"
  if [[ -z "${input_c}" ]]; then
    echo "ERROR: unsupported dataset '${dataset}'." >&2
    exit 1
  fi

  local prefix="${dataset}_win${WIN_SIZE}_in${input_c}_out${input_c}_batch${BATCH_SIZE}_patch${PATCH_LEN}_ep${EPOCHS}"
  local checkpoint="${CHECKPOINT_DIR}/${prefix}_checkpoint.pth"
  local artifact="${OUTPUT_DIR}/${prefix}_scores.npy"
  local data_path="${DATA_ROOT}/${dataset}"
  local actual_hash

  if [[ ! -d "${data_path}" ]]; then
    echo "ERROR: dataset directory not found: ${data_path}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
  actual_hash="$(sha256sum "${checkpoint}" | awk '{print toupper($1)}')"
  if [[ "${actual_hash}" != "${CHECKPOINT_HASHES[$dataset]}" ]]; then
    echo "ERROR: checkpoint SHA256 mismatch: ${checkpoint}" >&2
    echo "Expected: ${CHECKPOINT_HASHES[$dataset]}" >&2
    echo "Actual  : ${actual_hash}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Data: ${data_path}"
  echo "Checkpoint: ${checkpoint}"
  echo "Detection: no-Z raw MeanTopK-10%"
  if [[ "${dataset}" == "SWaT" || "${dataset}" == "WADI" ]]; then
    echo "RCA ranking: train-distribution per-sensor Z-score"
  else
    echo "RCA ranking: raw per-sensor score"
  fi
  echo "============================================================"

  "${PYTHON_BIN}" -u main.py \
    --mode test \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --input_c "${input_c}" \
    --output_c "${input_c}" \
    --win_size "${WIN_SIZE}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${EPOCHS}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${dataset}_inference.log"

  if [[ ! -f "${artifact}" ]]; then
    echo "ERROR: expected NPY artifact was not created: ${artifact}" >&2
    exit 1
  fi

  case "${dataset}" in
    SWaT)
      "${PYTHON_BIN}" evaluate_RCA_SWAT.py \
        --analysis event_rca \
        --scores_file "${artifact}" \
        --output_dir "${RCA_OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/SWaT_rca.log"
      ;;
    WADI)
      "${PYTHON_BIN}" evaluate_RCA_WADI.py \
        --analysis event_rca \
        --scores_file "${artifact}" \
        --output_dir "${RCA_OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/WADI_rca.log"
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}"
done

if find "${OUTPUT_DIR}" -type f -name '*.csv' -print -quit | grep -q .; then
  echo "ERROR: CSV output detected under ${OUTPUT_DIR}; only NPY artifacts are allowed." >&2
  exit 1
fi

echo "============================================================"
echo "Inference completed."
echo "NPY artifacts : ${OUTPUT_DIR}/*_scores.npy"
echo "Reports       : ${OUTPUT_DIR}/Report_*.txt"
echo "RCA reports   : ${RCA_OUTPUT_DIR}"
echo "============================================================"
