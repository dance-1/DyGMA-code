#!/usr/bin/env bash
set -euo pipefail

# Run one training epoch and one score-export test pass for all six datasets.
# Usage:
#   bash run_all_dygma_1epoch.sh


cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-1}
WIN_SIZE=${WIN_SIZE:-100}
PATCH_LEN=${PATCH_LEN:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
EXPORT_RCA=1  # Set to 1 to append Timestamp and TopK sensor rankings to scores.csv.

DATASETS=(MSL PSM HAI WADI SWaT SMD)


declare -A INPUT_DIMS=(
  [MSL]=55
  [PSM]=25
  [HAI]=86
  [WADI]=123
  [SWaT]=51
  [SMD]=38
)

run_dataset() {
  local dataset="$1"
  local input_c="${INPUT_DIMS[$dataset]}"
  local data_path="dataset/${dataset}"

  echo "========================================================="
  echo "Dataset: ${dataset} | epochs=${EPOCHS} | win=${WIN_SIZE} | patch=${PATCH_LEN} | batch=${BATCH_SIZE}"
  echo "Export RCA rankings: ${EXPORT_RCA}"
  echo "========================================================="

  local rca_args=()
  if [[ "${EXPORT_RCA}" == "1" ]]; then
    rca_args+=(--export_rca)
  fi

  python main.py \
    --num_epochs "${EPOCHS}" \
    --win_size "${WIN_SIZE}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --mode train \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --input_c "${input_c}" \
    --output_c "${input_c}"

  python main.py \
    --num_epochs "${EPOCHS}" \
    --win_size "${WIN_SIZE}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --mode test \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --input_c "${input_c}" \
    --output_c "${input_c}" \
    "${rca_args[@]}"
}

for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}"
done

echo "All datasets completed. Score CSV files are under checkpoints/."
