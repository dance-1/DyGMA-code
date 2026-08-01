# DyGMA

DyGMA is a multivariate time-series anomaly detection framework for industrial
and spacecraft telemetry. This release provides one reproducible inference
path with pretrained checkpoints, NPY score artifacts, and event-level root
cause analysis (RCA) for SWaT and WADI.

## Unified protocol

- Detection: raw no-Z MeanTopK-10% anomaly score for all six datasets.
- RCA export: always enabled in test mode; there is no optional RCA flag.
- SWaT/WADI RCA: per-sensor scores are standardized with mean and standard
  deviation estimated from the training split, then ranked in Z-score space.
- Other datasets: sensor rankings use raw per-sensor scores.
- Output: one versioned `*_scores.npy` artifact per dataset; inference does not
  export CSV files.

Z standardization changes only the sensor ranking used for SWaT/WADI RCA. It
does not change the anomaly score used by PA, Affiliation, ROC-AUC, or VUS-ROC.

## Repository structure

```text
.
+-- main.py                    # Training and unified test entry point
+-- solver.py                  # Training, raw detection, and mandatory RCA export
+-- artifact_io.py             # Versioned NPY artifact reader/writer
+-- evaluate.py                # Detection evaluation
+-- evaluate_RCA_SWAT.py       # SWaT Z-RCA evaluation
+-- evaluate_RCA_WADI.py       # WADI Z-RCA evaluation
+-- run_inference.sh           # Reproduce inference and RCA
+-- run_train_inference.sh     # Retrain, infer, and run RCA
+-- checkpoints/pretrained/    # Six canonical epoch-15 checkpoints
+-- data_factory/              # Dataset loaders
+-- model/                     # DyGMA model
+-- utils/                     # Utility functions
+-- vus/                       # VUS and Affiliation metrics
```

## Environment

Linux or WSL2 with CUDA-enabled PyTorch is recommended.

```bash
conda create -n dygma python=3.10
conda activate dygma
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA version when GPU
acceleration is required.

## Data preparation

By default, both runners read datasets from `dataset/` in the repository root:

```text
dataset/
+-- MSL/
|   +-- MSL_train.npy
|   +-- MSL_test.npy
|   +-- MSL_test_label.npy
+-- SMD/
|   +-- SMD_train.npy
|   +-- SMD_test.npy
|   +-- SMD_test_label.npy
+-- PSM/
|   +-- train.csv
|   +-- test.csv
|   +-- test_label.csv
+-- SWaT/
|   +-- train.csv
|   +-- test.csv
|   +-- test_label.csv
+-- HAI/
|   +-- train.csv
|   +-- test.csv
|   +-- test_label.csv
+-- WADI/
    +-- train.csv
    +-- test.csv
    +-- test_label.csv
```

For CSV datasets, the first column is the timestamp and the remaining columns
are sensor values. Labels are binary (`1` for anomaly, `0` for normal).

To use another dataset root without changing source code, set `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/dataset bash run_inference.sh SWaT WADI
```

## Pretrained weights

Six canonical epoch-15 checkpoints are included under
`checkpoints/pretrained/`. The inference script contains and verifies the
SHA256 digest of every checkpoint before loading a model.

## Reproduce inference and RCA

Run all six datasets:

```bash
bash run_inference.sh
```

Run only SWaT and WADI:

```bash
bash run_inference.sh SWaT WADI
```

Select a GPU before the command when needed:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh SWaT WADI
```

The script performs inference and detection evaluation for every requested
dataset. For SWaT and WADI it also runs the event-level Z-RCA evaluator.

## Output files

```text
outputs/
+-- {RUN_NAME}_scores.npy
+-- Report_{RUN_NAME}.txt
+-- logs/
|   +-- {DATASET}_inference.log
|   +-- SWaT_rca.log
|   +-- WADI_rca.log
+-- rca/
    +-- SWaT_RCA_Report.txt
    +-- WADI_RCA_Report.txt
```

Each score artifact is a NumPy dictionary with these fields:

- `format_version`
- `dataset`
- `score_protocol`
- `ranking_mode` (`z` for SWaT/WADI)
- `rca_protocol`
- `time` and `timestamp`
- `anomaly_score` and `ground_truth`
- `feature_names`
- `sensor_rankings` as integer feature indices
- `train_score_mean` and `train_score_std` for Z-RCA
- `metadata` with window, patch, batch, epoch, and dimensions

Load an artifact with:

```python
import numpy as np

artifact = np.load("outputs/SWaT_win100_in51_out51_batch32_patch1_ep15_scores.npy",
                   allow_pickle=True).item()
print(artifact["score_protocol"])
print(artifact["ranking_mode"])
print(artifact["sensor_rankings"].shape)
```

The artifact loader in `artifact_io.py` performs stronger schema, dataset, and
ranking-mode validation and is used by all included evaluators.

## Optional training

Training remains available through `main.py --mode train`. Use a separate
checkpoint directory so the provided weights remain unchanged:

```bash
python main.py \
  --mode train \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --win_size 100 \
  --patch_len 1 \
  --batch_size 32 \
  --num_epochs 15 \
  --checkpoint_dir checkpoints/custom \
  --output_dir outputs/custom
```

To retrain and then infer all six datasets in one complete run while keeping
the canonical weights untouched:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_train_inference.sh
```

The end-to-end runner uses 15 epochs by default and writes to
`checkpoints/retrained/` and `outputs/retrained/`. Override settings through
environment variables, for example `EPOCHS=1` for a one-epoch smoke test.

Test mode always exports sensor rankings; no export switch is exposed.

## Reproducibility notes

- Random seeds are fixed in `main.py`.
- The provided script fixes epoch, window, patch, and batch settings to match
  the included checkpoints.
- Checkpoint hashes are verified before inference.
- SWaT and WADI RCA evaluators reject artifacts whose `ranking_mode` is not
  `z`, preventing accidental raw/Z mixing.
- Generated datasets, outputs, logs, and custom checkpoints are excluded by
  `.gitignore`; the six pretrained checkpoints are explicitly included.

## License

This project is released under the MIT License.

## Citation

Citation information will be updated after the paper is released.
