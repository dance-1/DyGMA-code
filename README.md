# DyGMA

DyGMA is a multivariate time-series anomaly detection framework for industrial
and spacecraft telemetry.

## Environment

Linux or WSL2 with a CUDA-enabled PyTorch installation is recommended.

```bash
conda create -n dygma python=3.10
conda activate dygma
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA version when GPU
acceleration is required.

## Data

Place the six processed datasets under the repository's `dataset/` directory:

```text
dataset/
+-- MSL/
+-- SMD/
+-- PSM/
+-- SWaT/
+-- WADI/
+-- HAI/
```

The expected files for each dataset are defined in
`data_factory/data_loader.py`. To use a different data directory, set
`DATA_ROOT` when running a script:

```bash
DATA_ROOT=/path/to/dataset bash run_inference.sh
```

## Pretrained Inference

Pretrained checkpoints for all six datasets are included in
`checkpoints/pretrained/`.

Run all datasets:

```bash
bash run_inference.sh
```

Run selected datasets:

```bash
bash run_inference.sh SWaT WADI
```

Select a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh
```

Results are written to `outputs/`.

## Training And Inference

Train and evaluate all six datasets:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_train_inference.sh
```

Train and evaluate selected datasets:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_train_inference.sh SWaT WADI
```

The default training length is 15 epochs. Override it with `EPOCHS`:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=1 bash run_train_inference.sh
```

New checkpoints are written to `checkpoints/retrained/`, and results are
written to `outputs/retrained/`.

## License

This project is released under the MIT License.
