import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd


SWAT_PHYSICAL_PAIRS = {
    "mv101": ["lit101"],
    "p101": ["fit101", "lit101"],
    "p102": ["fit101", "lit101"],
    "mv201": ["fit201"],
    "p201": ["ait201", "ait202", "ait203"],
    "p202": ["ait201", "ait202", "ait203"],
    "p203": ["ait201", "ait202", "ait203"],
    "p204": ["ait201", "ait202", "ait203"],
    "p205": ["ait201", "ait202", "ait203"],
    "p206": ["ait201", "ait202", "ait203"],
    "p301": ["fit301", "dpit301"],
    "p302": ["fit301", "dpit301"],
    "mv301": ["dpit301", "fit301"],
    "mv302": ["dpit301", "fit301"],
    "mv303": ["dpit301", "lit301"],
    "mv304": ["dpit301", "lit301"],
    "p401": ["fit401", "lit401"],
    "p402": ["fit401", "lit401"],
    "uv401": ["ait401", "ait402"],
    "p501": ["fit501", "fit502", "pit501", "pit502", "pit503", "ait501", "ait502"],
    "p502": ["fit501", "fit502", "pit501", "pit502", "pit503", "ait501", "ait502"],
    "p601": ["fit601"],
    "p602": ["fit601"],
    "p603": ["fit601"],
}

SWAT_ATTACKS = [
    ("28/12/2015 10:29:14 AM", "28/12/2015 10:44:53 AM", "MV101", "", ""),
    ("28/12/2015 10:51:08 AM", "28/12/2015 10:58:30 AM", "P102", "", ""),
    ("28/12/2015 11:22:00 AM", "28/12/2015 11:28:22 AM", "LIT101", "", ""),
    ("28/12/2015 11:47:39 AM", "28/12/2015 11:54:08 AM", "MV504", "", ""),
    ("28/12/2015 12:00:55 PM", "28/12/2015 12:04:10 PM", "AIT202", "", ""),
    ("28/12/2015 12:08:25 PM", "28/12/2015 12:15:33 PM", "LIT301", "", ""),
    ("28/12/2015 1:10:10 PM", "28/12/2015 1:26:13 PM", "DPIT301", "", ""),
    ("28/12/2015 2:16:20 PM", "28/12/2015 2:28:20 PM", "FIT401", "", ""),
    ("29/12/2015 6:30:00 AM", "29/12/2015 6:42:00 AM", "FIT401", "", ""),
    ("29/12/2015 11:11:25 AM", "29/12/2015 11:15:17 AM", "MV304", "", ""),
    ("29/12/2015 11:35:40 AM", "29/12/2015 11:42:50 AM", "MV303", "", ""),
    ("29/12/2015 11:57:25 AM", "29/12/2015 12:02:00 PM", "LIT301", "", ""),
    ("29/12/2015 2:38:12 PM", "29/12/2015 2:50:08 PM", "MV303", "", ""),
    ("29/12/2015 6:10:43 PM", "29/12/2015 6:15:01 PM", "AIT504", "", ""),
    ("29/12/2015 6:15:43 PM", "29/12/2015 6:22:17 PM", "AIT504", "", ""),
    ("29/12/2015 6:30:00 PM", "29/12/2015 6:42:00 PM", "MV101", "LIT101", ""),
    ("29/12/2015 10:55:18 PM", "29/12/2015 11:03:00 PM", "UV401", "AIT502", "P501"),
    ("30/12/2015 1:42:34 AM", "30/12/2015 1:54:10 AM", "P602", "DIT301", "MV302"),
    ("30/12/2015 9:51:08 AM", "30/12/2015 9:56:28 AM", "P203", "P205", ""),
    ("30/12/2015 10:01:50 AM", "30/12/2015 10:12:01 AM", "LIT401", "P401", ""),
    ("30/12/2015 5:04:56 PM", "30/12/2015 5:29:00 PM", "P101", "LIT301", ""),
    ("31/12/2015 1:17:08 AM", "31/12/2015 1:45:18 AM", "P302", "LIT401", ""),
    ("31/12/2015 1:45:19 AM", "31/12/2015 11:15:27 AM", "P302", "", ""),
    ("31/12/2015 3:32:00 PM", "31/12/2015 3:34:00 PM", "P201", "P203", "P205"),
    ("31/12/2015 3:47:40 PM", "31/12/2015 4:07:10 PM", "LIT101", "P101", "MV201"),
    ("31/12/2015 10:05:34 PM", "31/12/2015 10:11:40 PM", "LIT401", "", ""),
    ("1/1/2016 10:36:00 AM", "1/1/2016 10:46:00 AM", "LIT301", "", ""),
    ("1/1/2016 2:21:12 PM", "1/1/2016 2:28:35 PM", "LIT101", "", ""),
    ("1/1/2016 5:12:40 PM", "1/1/2016 5:14:20 PM", "P101", "", ""),
    ("1/1/2016 5:18:56 PM", "1/1/2016 5:26:56 PM", "P101", "P102", ""),
    ("1/1/2016 10:16:01 PM", "1/1/2016 10:25:00 PM", "LIT101", "", ""),
    ("2/1/2016 11:17:02 AM", "2/1/2016 11:24:50 AM", "P501", "FIT502", ""),
    ("2/1/2016 11:31:38 AM", "2/1/2016 11:36:18 AM", "AIT402", "AIT502", ""),
    ("2/1/2016 11:43:48 AM", "2/1/2016 11:50:28 AM", "FIT401", "AIT502", ""),
    ("2/1/2016 11:51:42 AM", "2/1/2016 11:56:38 AM", "FIT401", "", ""),
    ("2/1/2016 1:13:02 PM", "2/1/2016 1:41:11 PM", "LIT301", "", ""),
]


def sensor_hit(sensor, targets):
    return str(sensor).strip().lower() in targets


def expand_targets(base_targets):
    expanded = set(base_targets)
    for target in base_targets:
        expanded.update(SWAT_PHYSICAL_PAIRS.get(target, []))
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


def load_attack_table():
    return pd.DataFrame(
        SWAT_ATTACKS,
        columns=["Start Time", "End Time", "Attack Point1", "Attack Point2", "Attack Point3"],
    )


def evaluate(scores_file, output_dir, focus_window):
    scores_df = pd.read_csv(scores_file, low_memory=False)
    attacks_df = load_attack_table()

    scores_df["Timestamp"] = scores_df["Timestamp"].astype(str).str.strip()
    attacks_df["Start Time"] = attacks_df["Start Time"].astype(str).str.strip()
    attacks_df["End Time"] = attacks_df["End Time"].astype(str).str.strip()

    scores_df["Time_Obj"] = pd.to_datetime(scores_df["Timestamp"], dayfirst=True, errors="coerce")
    attacks_df["Start_Obj"] = pd.to_datetime(attacks_df["Start Time"], dayfirst=True, errors="coerce")
    attacks_df["End_Obj"] = pd.to_datetime(attacks_df["End Time"], dayfirst=True, errors="coerce")

    top_cols = top_sensor_columns(scores_df)
    if not top_cols:
        raise ValueError("No TopK sensor columns found. Run test with --export_rca first.")

    mrr_values = []
    hr100_values = []
    hr150_values = []

    for _, attack in attacks_df.iterrows():
        base_targets = []
        for col in ["Attack Point1", "Attack Point2", "Attack Point3"]:
            if col in attacks_df.columns and pd.notna(attack[col]):
                target = str(attack[col]).strip().lower()
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
        raise ValueError("No valid SWaT attack events matched the score file timestamps.")

    metrics = {
        "Total_Events": len(mrr_values),
        "MRR": float(np.mean(mrr_values)),
        "HR@100%": float(np.mean(hr100_values)),
        "HR@150%": float(np.mean(hr150_values)),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "SWaT_RCA_Report.txt")

    report = (
        "============================================================\n"
        "SWaT RCA Evaluation Report\n"
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
    parser = argparse.ArgumentParser(description="Evaluate SWaT root-cause rankings.")
    parser.add_argument(
        "--scores_file",
        default=os.path.join("checkpoints", "SWaT_win100_in51_out51_batch32_patch1_ep1_scores.csv"),
    )
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--focus_window", type=int, default=180)
    args = parser.parse_args()

    evaluate(args.scores_file, args.output_dir, args.focus_window)


if __name__ == "__main__":
    main()
