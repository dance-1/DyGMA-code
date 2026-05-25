import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


try:
    from vus.affiliation.generics import convert_vector_to_events
    from vus.affiliation.metrics import pr_from_events

    AFFILIATION_AVAILABLE = True
except ImportError as exc:
    print(f"[Warning] Affiliation metrics are unavailable: {exc}")
    AFFILIATION_AVAILABLE = False

try:
    from vus.metrics import get_metrics

    VUS_AVAILABLE = True
except ImportError as exc:
    print(f"[Warning] VUS metrics are unavailable: {exc}")
    VUS_AVAILABLE = False


DATASET_NAMES = ("MSL", "SMD", "SWaT", "WADI", "PSM", "HAI")
EVALUATION_SECTION_TITLE = "Evaluation Metrics"


def apply_point_adjustment(scores, labels, threshold):
    pred = (scores > threshold).astype(int)
    adjusted = pred.copy()

    in_anomaly = False
    start = 0
    for i, label in enumerate(labels):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            if pred[start:i].sum() > 0:
                adjusted[start:i] = 1

    if in_anomaly and pred[start:].sum() > 0:
        adjusted[start:] = 1

    return adjusted


def infer_dataset_name(run_name):
    for dataset in DATASET_NAMES:
        if dataset in run_name:
            return dataset
    return "Unknown"


def extract_affiliation_score(result):
    if isinstance(result, dict):
        if "average" in result:
            return float(result["average"])
        if "score" in result:
            return float(result["score"])
        values = [value for value in result.values() if isinstance(value, (int, float))]
        return float(np.mean(values)) if values else 0.0
    return float(result)


def search_threshold_metrics(scores, labels):
    pa = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "ratio": 0.0,
        "threshold": 0.0,
    }
    affiliation = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    events_gt = convert_vector_to_events(labels) if AFFILIATION_AVAILABLE else None
    time_range = (0, len(labels))
    search_grid = np.arange(0.1, 5.1, 0.05)

    for ratio in search_grid:
        threshold = np.percentile(scores, 100 - ratio)
        pred = (scores > threshold).astype(int)

        pred_pa = apply_point_adjustment(scores, labels, threshold)
        _, _, f1_pa, _ = precision_recall_fscore_support(
            labels, pred_pa, average="binary", zero_division=0
        )
        if f1_pa > pa["f1"]:
            pa.update(
                {
                    "f1": float(f1_pa),
                    "precision": float(precision_score(labels, pred_pa, zero_division=0)),
                    "recall": float(recall_score(labels, pred_pa, zero_division=0)),
                    "ratio": float(ratio),
                    "threshold": float(threshold),
                }
            )

        if AFFILIATION_AVAILABLE:
            try:
                events_pred = convert_vector_to_events(pred)
                precision_result = pr_from_events(events_pred, events_gt, time_range)
                recall_result = pr_from_events(events_gt, events_pred, time_range)
            except Exception:
                continue

            precision = extract_affiliation_score(precision_result)
            recall = extract_affiliation_score(recall_result)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            if f1 > affiliation["f1"]:
                affiliation.update(
                    {"f1": float(f1), "precision": float(precision), "recall": float(recall)}
                )

    return pa, affiliation


def threshold_free_metrics(scores, labels, win_size):
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc_auc = 0.0

    vus_roc = 0.0
    if VUS_AVAILABLE:
        try:
            vus_roc = float(get_metrics(scores, labels, slidingWindow=win_size).get("VUS_ROC", 0.0))
        except Exception as exc:
            print(f"[Warning] Failed to calculate VUS-ROC: {exc}")

    return roc_auc, vus_roc


def build_evaluation_section(metrics):
    return (
        f"{EVALUATION_SECTION_TITLE}\n"
        "------------------------------------------------------------\n"
        f"    -> Best PA%0 F1 Score        : {metrics['Best PA%0 F1 Score']:.4f}\n"
        f"    -> Precision at Best F1      : {metrics['Precision at Best F1']:.4f}\n"
        f"    -> Recall at Best F1         : {metrics['Recall at Best F1']:.4f}\n"
        f"    -> Optimal PA Anomaly Ratio  : {metrics['Optimal PA Anomaly Ratio']:.2f}%\n"
        f"    -> Optimal PA Threshold      : {metrics['Optimal PA Threshold']:.4f}\n"
        f"    -> Best Affiliation F1       : {metrics['Best Affiliation F1']:.4f}\n"
        f"    -> Affiliation Precision     : {metrics['Affiliation Precision']:.4f}\n"
        f"    -> Affiliation Recall        : {metrics['Affiliation Recall']:.4f}\n"
        f"    -> Standard ROC-AUC          : {metrics['Standard ROC-AUC']:.4f}\n"
        f"    -> VUS-ROC                   : {metrics['VUS-ROC']:.4f}\n"
    )


def write_evaluation_section(report_path, run_name, metrics):
    section = build_evaluation_section(metrics)
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as report_file:
            report = report_file.read().rstrip()
    else:
        report = (
            "============================================================\n"
            f"Data from: {run_name}\n"
            "============================================================"
        )

    marker = f"\n\n{EVALUATION_SECTION_TITLE}\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip()

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"{report}\n\n{section}============================================================\n")


def evaluate_file(score_file, base_dir, win_size):
    run_name = os.path.basename(score_file).replace("_scores.csv", "")
    report_path = os.path.join(base_dir, f"Report_{run_name}.txt")

    print(f"[Evaluate] Loading scores: {score_file}", flush=True)
    frame = pd.read_csv(score_file)
    scores = frame["Anomaly_Score"].to_numpy()
    labels = frame["Ground_Truth"].to_numpy().astype(int)

    print("[Evaluate] Searching PA%0 and affiliation thresholds...", flush=True)
    pa, affiliation = search_threshold_metrics(scores, labels)

    print("[Evaluate] Calculating ROC-AUC and VUS-ROC...", flush=True)
    roc_auc, vus_roc = threshold_free_metrics(scores, labels, win_size)

    metrics = {
        "Best PA%0 F1 Score": round(pa["f1"], 4),
        "Precision at Best F1": round(pa["precision"], 4),
        "Recall at Best F1": round(pa["recall"], 4),
        "Optimal PA Anomaly Ratio": round(pa["ratio"], 2),
        "Optimal PA Threshold": pa["threshold"],
        "Best Affiliation F1": round(affiliation["f1"], 4),
        "Affiliation Precision": round(affiliation["precision"], 4),
        "Affiliation Recall": round(affiliation["recall"], 4),
        "Standard ROC-AUC": round(roc_auc, 4),
        "VUS-ROC": round(vus_roc, 4),
    }

    write_evaluation_section(report_path, run_name, metrics)
    print(f"[Evaluate] Saved report: {report_path}", flush=True)
    return run_name, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DyGMA score files.")
    parser.add_argument("--base_dir", type=str, default="checkpoints", help="Directory with score files.")
    parser.add_argument("--score_file", type=str, default=None, help="Evaluate a single score file.")
    parser.add_argument("--win_size", type=int, default=100, help="Window size used by VUS-ROC.")
    args = parser.parse_args()

    if args.score_file is not None:
        score_files = [args.score_file]
    else:
        score_files = sorted(
            glob.glob(os.path.join(args.base_dir, "**", "*_scores.csv"), recursive=True)
        )

    print("=" * 60)
    print(f"Evaluation directory: {args.base_dir}")
    print(f"Detected score files: {len(score_files)}")
    print("=" * 60)

    for score_file in score_files:
        evaluate_file(score_file, args.base_dir, args.win_size)

    if not score_files:
        print("No score files were found.")


if __name__ == "__main__":
    main()
