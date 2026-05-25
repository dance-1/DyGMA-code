import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

WADI_ATTACKS = [
    ("2017/10/9 19:25:00", "2017/10/9 19:50:16", "1_MV_001", "", "", "", "", ""),
    ("2017/10/10 10:24:10", "2017/10/10 10:34:00", "1_FIT_001", "", "", "", "", ""),
    ("2017/10/10 10:55:00", "2017/10/10 11:24:00", "2_LIT_002", "1_AIT_001", "", "", "", ""),
    ("2017/10/10 11:30:40", "2017/10/10 11:44:50", "2_MCV_101", "2_MCV_201", "2_MCV_301", "2_MCV_401", "2_MCV_501", "2_MCV_601"),
    ("2017/10/10 13:39:30", "2017/10/10 13:50:40", "2_MCV_101", "2_MCV_201", "", "", "", ""),
    ("2017/10/10 14:48:17", "2017/10/10 15:00:32", "1_AIT_002", "2_MV_003", "", "", "", ""),
    ("2017/10/10 17:40:00", "2017/10/10 17:49:40", "2_MCV_007", "", "", "", "", ""),
    ("2017/10/11 10:55:00", "2017/10/11 10:56:27", "1_P_005", "1_P_006", "", "", "", ""),
    ("2017/10/11 11:17:54", "2017/10/11 11:31:20", "1_MV_001", "", "", "", "", ""),
    ("2017/10/11 11:36:31", "2017/10/11 11:47:00", "2_MCV_007", "", "", "", "", ""),
    ("2017/10/11 11:59:00", "2017/10/11 12:05:00", "2_MCV_007", "", "", "", "", ""),
    ("2017/10/11 12:07:30", "2017/10/11 12:10:52", "2_PIC_003", "", "", "", "", ""),
    ("2017/10/11 12:16:00", "2017/10/11 12:25:36", "1_P_001", "1_P_003", "", "", "", ""),
    ("2017/10/11 15:26:30", "2017/10/11 15:37:00", "2_LIT_002", "", "", "", "", ""),
]

WADI_PHYSICAL_PAIRS = {
    "1_mv_001": ["1_lt_001", "1_fit_001"],
    "1_mv_002": ["1_lt_001", "1_fit_001"],
    "1_mv_003": ["1_lt_001", "1_fit_001"],
    "1_p_001": ["1_fit_001", "1_pit_001"],
    "1_p_002": ["1_fit_001", "1_pit_001"],
    "1_p_003": ["1_fit_001", "1_pit_001"],
    "1_p_004": ["1_fit_001", "1_pit_001"],
    "1_p_005": ["1_fit_001", "1_pit_001"],
    "1_p_006": ["1_fit_001", "1_pit_001"],
    "2_mcv_101": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mcv_201": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mcv_301": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mcv_401": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mcv_501": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mcv_601": ["2_fit_001", "2_fit_002", "2_fit_003", "2_lt_001", "2_lt_002"],
    "2_mv_003": ["2_lt_001", "2_lt_002"],
    "2_mv_006": ["2_lt_001", "2_lt_002"],
    "2_p_003": ["2_fit_001", "2_fit_002", "2_fit_003", "2_pit_001"],
    "2_p_004": ["2_fit_001", "2_fit_002", "2_fit_003", "2_pit_001"],
    "3_mv_001": ["3_lt_001"],
    "3_mv_002": ["3_lt_001"],
    "3_mv_003": ["3_lt_001"],
    "3_p_001": ["3_fit_001"],
    "3_p_002": ["3_fit_001"],
    "3_p_003": ["3_fit_001"],
    "3_p_004": ["3_fit_001"],
}


def embedded_attacks():
    columns = ["Start Time", "End Time"] + [f"Attack Point{i}" for i in range(1, 7)]
    frame = pd.DataFrame(WADI_ATTACKS, columns=columns)
    frame["Start_Obj"] = pd.to_datetime(frame["Start Time"])
    frame["End_Obj"] = pd.to_datetime(frame["End Time"])
    return frame


def sensor_hit(sensor, targets):
    sensor = str(sensor).strip().lower()
    return any(target in sensor for target in targets)


def expand_targets(base_targets):
    expanded = set(base_targets)
    for target in base_targets:
        expanded.update(WADI_PHYSICAL_PAIRS.get(target, []))
    return expanded


def top_sensor_columns(frame):
    columns = [col for col in frame.columns if col.startswith("Top") and col.endswith("_Sensor")]
    return sorted(columns, key=lambda col: int(col[3:].split("_", 1)[0]))


def aggregate_event_ranking(event_frame, top_cols):
    votes = defaultdict(float)
    for _, row in event_frame.iterrows():
        for rank, col in enumerate(top_cols, start=1):
            sensor = str(row[col]).strip().lower() if pd.notna(row[col]) else ""
            if sensor and sensor != "nan":
                votes[sensor] += 1.0 / rank
    return [sensor for sensor, _ in sorted(votes.items(), key=lambda item: item[1], reverse=True)]


def evaluate(scores_file, output_dir, focus_window):
    scores_df = pd.read_csv(scores_file, low_memory=False)
    attacks_df = embedded_attacks()

    scores_df["Timestamp"] = scores_df["Timestamp"].astype(str).str.strip()
    scores_df["Time_Obj"] = pd.to_datetime(scores_df["Timestamp"], errors="coerce")

    top_cols = top_sensor_columns(scores_df)
    if not top_cols:
        raise ValueError("No TopK sensor columns found. Run test with --export_rca first.")

    mrr_values = []
    hr100_values = []
    hr150_values = []

    for _, attack in attacks_df.iterrows():
        base_targets = []
        for col in [f"Attack Point{i}" for i in range(1, 7)]:
            target = str(attack[col]).strip().lower() if pd.notna(attack[col]) else ""
            if target and target != "nan":
                base_targets.append(target)
        if not base_targets:
            continue

        targets = expand_targets(base_targets)
        mask = (scores_df["Time_Obj"] >= attack["Start_Obj"]) & (
            scores_df["Time_Obj"] <= attack["End_Obj"]
        )
        event_frame = scores_df[mask]
        if event_frame.empty:
            continue

        event_frame = event_frame.head(focus_window)
        predictions = aggregate_event_ranking(event_frame, top_cols)

        reciprocal_rank = 0.0
        for rank, sensor in enumerate(predictions, start=1):
            if sensor_hit(sensor, targets):
                reciprocal_rank = 1.0 / rank
                break

        k100 = int(np.ceil(len(targets) * 1.0))
        k150 = int(np.ceil(len(targets) * 1.5))

        mrr_values.append(reciprocal_rank)
        hr100_values.append(float(any(sensor_hit(sensor, targets) for sensor in predictions[:k100])))
        hr150_values.append(float(any(sensor_hit(sensor, targets) for sensor in predictions[:k150])))

    if not mrr_values:
        raise ValueError("No valid WADI attack events matched the score file timestamps.")

    metrics = {
        "Total_Events": len(mrr_values),
        "MRR": float(np.mean(mrr_values)),
        "HR@100%": float(np.mean(hr100_values)),
        "HR@150%": float(np.mean(hr150_values)),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "WADI_RCA_Report.txt")
    report = (
        "============================================================\n"
        "WADI RCA Evaluation Report\n"
        "============================================================\n"
        f"Valid Events : {metrics['Total_Events']}\n"
        f"MRR          : {metrics['MRR']:.4f}\n"
        f"HR@100%      : {metrics['HR@100%']:.4f}\n"
        f"HR@150%      : {metrics['HR@150%']:.4f}\n"
        "============================================================\n"
    )

    print(report)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved RCA report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WADI root-cause rankings.")
    parser.add_argument(
        "--scores_file",
        default=os.path.join("checkpoints", "WADI_win100_in123_out123_batch32_patch1_ep1_scores.csv"),
    )
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--focus_window", type=int, default=180)
    args = parser.parse_args()

    evaluate(args.scores_file, args.output_dir, args.focus_window)


if __name__ == "__main__":
    main()
