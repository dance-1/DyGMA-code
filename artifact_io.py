import os

import numpy as np


FORMAT_VERSION = 1
VALID_RANKING_MODES = {"raw", "standardized"}


def save_score_artifact(
    path,
    *,
    dataset,
    anomaly_score,
    ground_truth,
    timestamps,
    feature_names,
    sensor_rankings,
    ranking_mode,
    train_score_mean=None,
    train_score_std=None,
    metadata=None,
):
    """Save detection scores and RCA rankings as one versioned NPY artifact."""
    if ranking_mode not in VALID_RANKING_MODES:
        raise ValueError(f"Unsupported ranking mode: {ranking_mode}")

    anomaly_score = np.asarray(anomaly_score, dtype=np.float64).reshape(-1)
    ground_truth = np.asarray(ground_truth, dtype=np.int8).reshape(-1)
    timestamps = np.asarray(timestamps).astype(str).reshape(-1)
    feature_names = np.asarray(feature_names).astype(str).reshape(-1)
    sensor_rankings = np.asarray(sensor_rankings, dtype=np.int16)

    length = anomaly_score.shape[0]
    expected_shape = (length, feature_names.shape[0])
    if ground_truth.shape[0] != length or timestamps.shape[0] != length:
        raise ValueError("Score, label, and timestamp lengths must match.")
    if sensor_rankings.shape != expected_shape:
        raise ValueError(
            f"Sensor ranking shape mismatch: {sensor_rankings.shape} != {expected_shape}"
        )
    if sensor_rankings.size and (
        sensor_rankings.min() < 0 or sensor_rankings.max() >= feature_names.shape[0]
    ):
        raise ValueError("Sensor rankings contain an out-of-range feature index.")

    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": str(dataset),
        "score_protocol": "raw MeanTopK-10%",
        "ranking_mode": ranking_mode,
        "rca_protocol": (
            "train-distribution standardized per-sensor score"
            if ranking_mode == "standardized"
            else "raw per-sensor score"
        ),
        "time": np.arange(length, dtype=np.int64),
        "timestamp": timestamps,
        "anomaly_score": anomaly_score,
        "ground_truth": ground_truth,
        "feature_names": feature_names,
        "sensor_rankings": sensor_rankings,
        "train_score_mean": np.asarray(
            [] if train_score_mean is None else train_score_mean, dtype=np.float32
        ),
        "train_score_std": np.asarray(
            [] if train_score_std is None else train_score_std, dtype=np.float32
        ),
        "metadata": dict(metadata or {}),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, payload, allow_pickle=True)
    return path


def load_score_artifact(path, expected_dataset=None, require_ranking_mode=None):
    """Load and validate a DyGMA score artifact created by save_score_artifact."""
    if not str(path).endswith(".npy"):
        raise ValueError("DyGMA score artifacts must use the .npy extension.")

    loaded = np.load(path, allow_pickle=True)
    if loaded.shape != ():
        raise ValueError("Expected a scalar dictionary NPY artifact.")
    payload = loaded.item()
    if not isinstance(payload, dict):
        raise ValueError("The NPY artifact does not contain a dictionary payload.")

    required = {
        "format_version",
        "dataset",
        "score_protocol",
        "ranking_mode",
        "time",
        "timestamp",
        "anomaly_score",
        "ground_truth",
        "feature_names",
        "sensor_rankings",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"NPY artifact is missing required fields: {missing}")
    if int(payload["format_version"]) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported artifact version: {payload['format_version']}"
        )
    if payload["ranking_mode"] not in VALID_RANKING_MODES:
        raise ValueError(f"Invalid ranking mode: {payload['ranking_mode']}")
    if expected_dataset and payload["dataset"] != expected_dataset:
        raise ValueError(
            f"Dataset mismatch: expected {expected_dataset}, got {payload['dataset']}"
        )
    if require_ranking_mode and payload["ranking_mode"] != require_ranking_mode:
        raise ValueError(
            f"RCA requires ranking_mode={require_ranking_mode}, "
            f"but artifact uses {payload['ranking_mode']}."
        )

    length = np.asarray(payload["anomaly_score"]).reshape(-1).shape[0]
    feature_count = np.asarray(payload["feature_names"]).reshape(-1).shape[0]
    rankings = np.asarray(payload["sensor_rankings"])
    if rankings.shape != (length, feature_count):
        raise ValueError(
            f"Invalid sensor ranking shape: {rankings.shape} != {(length, feature_count)}"
        )
    for key in ("time", "timestamp", "ground_truth"):
        if np.asarray(payload[key]).reshape(-1).shape[0] != length:
            raise ValueError(f"Artifact field '{key}' has an inconsistent length.")
    return payload
