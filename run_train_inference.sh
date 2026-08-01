#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end retraining and inference. Newly trained checkpoints and outputs
# are isolated from the canonical pretrained release artifacts.

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/retrained}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/retrained}"
LOG_DIR="${OUTPUT_DIR}/logs"
RCA_OUTPUT_DIR="${OUTPUT_DIR}/rca"

EPOCHS="${EPOCHS:-15}"
WIN_SIZE="${WIN_SIZE:-100}"
PATCH_LEN="${PATCH_LEN:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"

declare -A INPUT_DIMS=(
  [MSL]=55
  [SMD]=38
  [PSM]=25
  [SWaT]=51
  [WADI]=123
  [HAI]=86
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

mkdir -p "${CHECKPOINT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}" "${RCA_OUTPUT_DIR}"

run_dataset() {
  local dataset="$1"
  local input_c="${INPUT_DIMS[$dataset]:-}"
  local data_path="${DATA_ROOT}/${dataset}"
  local prefix
  local artifact

  if [[ -z "${input_c}" ]]; then
    echo "ERROR: unsupported dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -d "${data_path}" ]]; then
    echo "ERROR: dataset directory not found: ${data_path}" >&2
    exit 1
  fi

  prefix="${dataset}_win${WIN_SIZE}_in${input_c}_out${input_c}_batch${BATCH_SIZE}_patch${PATCH_LEN}_ep${EPOCHS}"
  artifact="${OUTPUT_DIR}/${prefix}_scores.npy"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Data: ${data_path}"
  echo "Training epochs: ${EPOCHS}"
  echo "Checkpoint output: ${CHECKPOINT_DIR}/${prefix}_checkpoint.pth"
  echo "============================================================"

  "${PYTHON_BIN}" -u main.py \
    --mode train \
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
    2>&1 | tee "${LOG_DIR}/${dataset}_train.log"

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
echo "Training and inference completed."
echo "Checkpoints : ${CHECKPOINT_DIR}"
echo "NPY outputs : ${OUTPUT_DIR}/*_scores.npy"
echo "Reports     : ${OUTPUT_DIR}/Report_*.txt"
echo "RCA reports : ${RCA_OUTPUT_DIR}"
echo "============================================================"
