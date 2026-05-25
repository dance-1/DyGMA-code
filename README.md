# DyGMA

DyGMA is a multivariate time-series anomaly detection framework for industrial and spacecraft telemetry data. The repository provides training, score export, threshold-based evaluation, efficiency profiling, and optional root-cause ranking export for RCA analysis.

The current implementation supports six benchmark datasets:

- MSL
- SMD
- PSM
- SWaT
- HAI
- WADI

## Repository Structure

```text
.
+-- main.py                    # Training and test entry point
+-- solver.py                  # DyGMA training, scoring, profiling, and RCA export
+-- evaluate.py                # PA%0, Affiliation, ROC-AUC, and VUS-ROC evaluation
+-- evaluate_hr_mrr_SWAT.py    # SWaT RCA evaluation
+-- evaluate_hr_mrr_WADI.py    # WADI RCA evaluation
+-- run_all_dygma_1epoch.sh    # One-epoch sanity run for all supported datasets
+-- data_factory/              # Dataset loaders
+-- model/                     # DyGMA model
+-- utils/                     # Utility functions
+-- vus/                       # VUS and Affiliation metric utilities
```

## Environment

Create a Python environment with PyTorch and the common scientific stack:

```bash
conda create -n dygma python=3.10
conda activate dygma
pip install numpy pandas scikit-learn scipy torch
```

Install the proper PyTorch build for your CUDA version from the official PyTorch instructions if GPU acceleration is required.

## Data Preparation

Place datasets under `dataset/`:

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

For CSV datasets, the first column is treated as the timestamp column and the remaining columns are sensor values. Labels should be binary, where `1` denotes anomaly and `0` denotes normal.

Large datasets use a larger validation stride to keep validation practical. WADI and HAI also include normalization safeguards for near-constant sensors.

## Train

Example: train DyGMA on MSL for one epoch.

```bash
python main.py \
  --num_epochs 1 \
  --win_size 100 \
  --patch_len 1 \
  --batch_size 32 \
  --mode train \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55
```

The final checkpoint is saved under `checkpoints/`.

## Test And Evaluate

Run test mode after training:

```bash
python main.py \
  --num_epochs 1 \
  --win_size 100 \
  --patch_len 1 \
  --batch_size 32 \
  --mode test \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55
```

Test mode exports anomaly scores and writes an efficiency report. It then automatically calls `evaluate.py` on the generated score file.

Outputs:

```text
checkpoints/{DATASET}_win{WIN}_in{DIM}_out{DIM}_batch{BATCH}_patch{PATCH}_ep{EPOCH}_scores.csv
checkpoints/Report_{DATASET}_win{WIN}_in{DIM}_out{DIM}_batch{BATCH}_patch{PATCH}_ep{EPOCH}.txt
```

The report includes:

- Best PA%0 F1
- Precision and recall at the best PA%0 F1
- Optimal PA anomaly ratio and threshold
- Best Affiliation F1
- Affiliation precision and recall
- Standard ROC-AUC
- VUS-ROC
- Model parameters
- FLOPs
- Inference latency
- GPU peak memory

## Run All Datasets

Use the provided script for a one-epoch sanity run over all six datasets:

```bash
bash run_all_dygma_1epoch.sh
```

You can override common settings:

```bash
EPOCHS=3 BATCH_SIZE=64 bash run_all_dygma_1epoch.sh
```

Edit `DATASETS` inside the script if you only want to run a subset.

## RCA Export

To export per-timestamp sensor rankings for root-cause analysis, enable RCA export in `run_all_dygma_1epoch.sh`:

```bash
EXPORT_RCA=1
```

Or pass `--export_rca` in test mode:

```bash
python main.py \
  --num_epochs 1 \
  --win_size 100 \
  --patch_len 1 \
  --batch_size 32 \
  --mode test \
  --dataset WADI \
  --data_path dataset/WADI \
  --input_c 123 \
  --output_c 123 \
  --export_rca
```

RCA export adds `Timestamp` and `TopK_Sensor` columns to the score CSV. Sensor ranking uses per-sensor reconstruction scores normalized by Z-score statistics estimated from the training set.

## RCA Evaluation

SWaT:

```bash
python evaluate_hr_mrr_SWAT.py \
  --scores_file checkpoints/SWaT_win100_in51_out51_batch32_patch1_ep1_scores.csv
```

WADI:

```bash
python evaluate_hr_mrr_WADI.py \
  --scores_file checkpoints/WADI_win100_in123_out123_batch32_patch1_ep1_scores.csv
```

Each script writes a text report under `checkpoints/` with:

- MRR
- HR@100%
- HR@150%

The SWaT and WADI attack descriptions used by these RCA evaluation scripts are embedded in the scripts to avoid extra spreadsheet dependencies.

## Notes For Reproducibility

- Random seeds are fixed in `main.py`.
- Score export uses non-overlapping windows so the exported score sequence matches the original evaluation protocol.
- Evaluation is decoupled from training: test mode generates scores first, then evaluates the generated score file.
- Dataset files, checkpoints, logs, and generated reports are not intended to be committed to Git.

## License

This project is released under the MIT License.

## Citation

If you use this repository in your research, please cite the DyGMA paper:

```bibtex
@article{dygma,
  title={DyGMA: Dynamic Graph Mamba for Multivariate Time-Series Anomaly Detection},
  author={DyGMA Authors},
  journal={},
  year={2026}
}
```

Update the BibTeX entry with the final publication information before release.
