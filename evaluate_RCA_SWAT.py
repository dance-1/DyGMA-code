import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

# python evaluate_RCA_SWAT.py --scores_file checkpoints/SWaT_win100_in51_out51_batch32_patch1_ep12_scores.csv


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

SWAT_CONTINUOUS_PREFIXES = ("FIT", "LIT", "AIT", "DPIT", "PIT")


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


def load_rca_frame(rca_source):
    frame = pd.read_csv(rca_source, low_memory=False)
    top_cols = top_sensor_columns(frame)
    if not top_cols:
        raise ValueError(
            "No TopK sensor columns found. Run test with --export_rca first, "
            "or pass an RCA table with Top*_Sensor columns."
        )
    if "Timestamp" not in frame.columns:
        raise ValueError("RCA evaluation requires a Timestamp column.")
    return frame, top_cols


def evaluate(
    rca_source,
    output_dir,
    focus_window,
    write_report=True,
):
    scores_df, top_cols = load_rca_frame(rca_source)
    attacks_df = load_attack_table()

    scores_df["Timestamp"] = scores_df["Timestamp"].astype(str).str.strip()
    attacks_df["Start Time"] = attacks_df["Start Time"].astype(str).str.strip()
    attacks_df["End Time"] = attacks_df["End Time"].astype(str).str.strip()

    scores_df["Time_Obj"] = pd.to_datetime(scores_df["Timestamp"], dayfirst=True, errors="coerce")
    attacks_df["Start_Obj"] = pd.to_datetime(attacks_df["Start Time"], dayfirst=True, errors="coerce")
    attacks_df["End_Obj"] = pd.to_datetime(attacks_df["End Time"], dayfirst=True, errors="coerce")

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
        "Source_File": rca_source,
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
    if write_report:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Saved RCA report: {report_path}")
    return metrics


def resolve_rca_source(scores_file, rca_file):
    if rca_file:
        if rca_file.endswith("_RCA.csv") and not rca_file.endswith("_RCA_noZ.csv"):
            raise ValueError("Old Z-normalized *_RCA.csv files are not accepted.")
        return rca_file

    if scores_file.endswith("_scores.csv"):
        candidate = scores_file.replace("_scores.csv", "_RCA_noZ.csv")
        if os.path.exists(candidate):
            return candidate
        return scores_file

    if scores_file.endswith("_RCA_noZ.csv"):
        return scores_file

    raise ValueError("Pass --scores_file or --rca_file pointing to a CSV with Top*_Sensor columns.")


def my_kl_loss_var(p, q):
    import torch

    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


def adaptive_weight(series, prior):
    import torch

    entropy_series = -torch.sum(series * torch.log(series + 1e-5), dim=-1)
    entropy_prior = -torch.sum(prior.detach() * torch.log(prior.detach() + 1e-5), dim=-1)
    confidence = 1.0 - (entropy_series / (entropy_prior + 1e-5))
    confidence = torch.clamp(confidence, min=0.0, max=1.0)
    return torch.exp(torch.mean(confidence).detach())


def read_values(path):
    frame = pd.read_csv(path)
    return frame, np.nan_to_num(frame.values[:, 1:].astype(np.float32))


def load_swat_arrays(data_path):
    from sklearn.preprocessing import StandardScaler

    train_frame, train_values = read_values(os.path.join(data_path, "train.csv"))
    test_frame, test_values = read_values(os.path.join(data_path, "test.csv"))
    labels = pd.read_csv(os.path.join(data_path, "test_label.csv")).iloc[:, 1].to_numpy().astype(int)

    scaler = StandardScaler()
    scaler.fit(train_values)
    scaler.scale_ = np.where(scaler.scale_ < 1e-3, 1.0, scaler.scale_)

    test_scaled = np.clip(scaler.transform(test_values), -10, 10).astype(np.float32)
    feature_names = list(train_frame.columns[1:])
    timestamps = test_frame.iloc[:, 0].to_numpy()
    return test_scaled, labels, timestamps, feature_names


def sample_clean_windows(values, labels, timestamps, win_size, max_windows, seed):
    candidates = []
    for start in range(0, len(values) - win_size + 1, win_size):
        end = start + win_size
        if labels[start:end].sum() == 0:
            candidates.append(start)

    if not candidates:
        raise ValueError("No clean SWaT windows were found for the controlled pilot.")

    rng = random.Random(seed)
    if len(candidates) > max_windows:
        candidates = rng.sample(candidates, max_windows)
    candidates = sorted(candidates)

    windows = np.stack([values[start:start + win_size] for start in candidates]).astype(np.float32)
    window_times = [timestamps[start] for start in candidates]
    return windows, candidates, window_times


def continuous_feature_indices(feature_names):
    return [
        idx for idx, name in enumerate(feature_names)
        if name.startswith(SWAT_CONTINUOUS_PREFIXES)
    ]


def make_benign_drift(windows, feature_names, strength, ratio, seed):
    result = windows.copy()
    cont = continuous_feature_indices(feature_names)
    rng = np.random.default_rng(seed)
    count = max(2, int(len(cont) * ratio))
    chosen = np.array(sorted(rng.choice(cont, size=count, replace=False)))

    ramp = np.linspace(0.0, strength, result.shape[1], dtype=np.float32)
    shared_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32))
    feature_scale = rng.uniform(0.6, 1.0, size=count).astype(np.float32)
    result[:, :, chosen] += ramp[None, :, None] * shared_sign * feature_scale[None, None, :]
    return np.clip(result, -10, 10), chosen


def make_topology_disruption(windows, feature_names, ratio, seed):
    result = windows.copy()
    cont = continuous_feature_indices(feature_names)
    rng = np.random.default_rng(seed)
    count = max(4, int(len(cont) * ratio))
    chosen = np.array(sorted(rng.choice(cont, size=count, replace=False)), dtype=np.int32)

    num_windows = windows.shape[0]
    for feature_idx in chosen:
        perm = rng.permutation(num_windows)
        if np.any(perm == np.arange(num_windows)):
            perm = np.roll(perm, 1)
        result[:, :, feature_idx] = windows[perm, :, feature_idx]

    return np.clip(result, -10, 10), chosen


def build_model(args, device):
    import torch
    from model.DyGMA import DyGMAModel

    model = DyGMAModel(
        win_size=args.win_size,
        num_vars=args.input_c,
        patch_len=args.patch_len,
        d_model=128,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def component_scores(model, windows, batch_size, device):
    import torch
    import torch.nn as nn

    criterion = nn.MSELoss(reduction="none")
    summaries = {
        "full_dygma": [],
        "reconstruction_only": [],
        "raw_graph_only": [],
        "weighted_graph_only": [],
    }

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start:start + batch_size], dtype=torch.float32, device=device)
            output, series, prior = model(batch)

            reconstruction = criterion(batch, output)
            series_loss = my_kl_loss_var(series, prior.detach()) + my_kl_loss_var(prior.detach(), series)
            prior_loss = my_kl_loss_var(prior, series.detach()) + my_kl_loss_var(series.detach(), prior)
            raw_graph = series_loss + prior_loss

            mean_loss = torch.mean(reconstruction, dim=1, keepdim=True)
            temporal_weight = torch.clamp(reconstruction / (mean_loss + 1e-5), min=1.0)
            weighted_graph = raw_graph.unsqueeze(1) * temporal_weight * adaptive_weight(series, prior)
            full = reconstruction + weighted_graph

            components = {
                "full_dygma": full,
                "reconstruction_only": reconstruction,
                "raw_graph_only": raw_graph.unsqueeze(1).expand_as(reconstruction),
                "weighted_graph_only": weighted_graph,
            }

            for name, scores in components.items():
                top_k = max(1, int(0.10 * scores.shape[-1]))
                energy = torch.topk(scores, k=top_k, dim=-1)[0].mean(dim=-1).mean(dim=-1)
                summaries[name].append(energy.detach().cpu().numpy())

    return {name: np.concatenate(parts) for name, parts in summaries.items()}


def summarize_condition(base_scores, condition_scores, condition):
    rows = []
    for variant, values in condition_scores.items():
        base = base_scores[variant]
        delta = values - base
        pooled = np.sqrt((np.var(values) + np.var(base)) / 2.0)
        effect = float(np.mean(delta) / (pooled + 1e-8))
        rows.append({
            "condition": condition,
            "variant": variant,
            "base_mean": float(np.mean(base)),
            "condition_mean": float(np.mean(values)),
            "delta_mean": float(np.mean(delta)),
            "relative_delta": float(np.mean(delta) / (abs(np.mean(base)) + 1e-8)),
            "effect_size": effect,
        })
    return rows


def save_window_npz(path, windows, starts, timestamps, feature_names, perturbed_indices=None):
    if perturbed_indices is None:
        perturbed_indices = np.array([], dtype=np.int32)
    perturbed_indices = np.asarray(perturbed_indices, dtype=np.int32)
    perturbed_names = np.asarray([feature_names[idx] for idx in perturbed_indices], dtype=object)

    np.savez_compressed(
        path,
        windows=windows.astype(np.float32),
        starts=np.asarray(starts, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        perturbed_feature_indices=perturbed_indices,
        perturbed_feature_names=perturbed_names,
    )


def run_controlled_pilot(args):
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    values, labels, timestamps, feature_names = load_swat_arrays(args.data_path)
    windows, starts, window_times = sample_clean_windows(
        values, labels, timestamps, args.win_size, args.max_windows, args.seed
    )

    benign, benign_features = make_benign_drift(
        windows, feature_names, args.drift_strength, args.perturb_ratio, args.seed + 1
    )
    disrupted, disrupted_features = make_topology_disruption(
        windows, feature_names, args.perturb_ratio, args.seed + 2
    )

    model = build_model(args, device)
    base_scores = component_scores(model, windows, args.batch_size, device)
    benign_scores = component_scores(model, benign, args.batch_size, device)
    disrupted_scores = component_scores(model, disrupted, args.batch_size, device)

    rows = []
    rows.extend(summarize_condition(base_scores, benign_scores, "benign_drift"))
    rows.extend(summarize_condition(base_scores, disrupted_scores, "topology_disruption"))
    summary = pd.DataFrame(rows)

    benign_names = ", ".join(feature_names[idx] for idx in benign_features[:12])
    disrupted_names = ", ".join(feature_names[idx] for idx in disrupted_features[:12])
    summary.attrs["meta"] = {
        "checkpoint": args.checkpoint,
        "clean_windows": len(windows),
        "win_size": args.win_size,
        "perturb_ratio": args.perturb_ratio,
        "drift_strength": args.drift_strength,
        "device": str(device),
        "benign_feature_sample": benign_names,
        "topology_feature_sample": disrupted_names,
    }
    return summary


def write_combined_analysis(dataset, output_dir, event_metrics=None, pilot_summary=None):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{dataset}_RCA_Report.txt")

    report_lines = [
        "============================================================",
        f"{dataset} RCA Analysis Report",
        "============================================================",
        "1. Controlled-pilot detailed summary",
        "------------------------------------------------------------",
    ]
    if pilot_summary is not None:
        report_lines.append(pilot_summary.to_string(index=False))
    else:
        report_lines.append("(controlled pilot was not run)")

    report_lines.extend([
        "",
        "2. Table A. Controlled topology perturbation analysis",
        "Values are standardized effect sizes; Gap = Topology - Benign.",
        "------------------------------------------------------------",
    ])

    if pilot_summary is not None:
        meta = pilot_summary.attrs.get("meta", {})
        report_lines.extend([
            f"Checkpoint      : {meta.get('checkpoint', '')}",
            f"Clean windows   : {meta.get('clean_windows', '')}",
            f"Window size     : {meta.get('win_size', '')}",
            f"Perturb ratio   : {meta.get('perturb_ratio', '')}",
            f"Drift strength  : {meta.get('drift_strength', '')}",
            f"Device          : {meta.get('device', '')}",
            f"Benign features : {meta.get('benign_feature_sample', '')}",
            f"Topology features: {meta.get('topology_feature_sample', '')}",
            "",
        ])
        effect_rows = []
        evidence_map = [
            ("Recon", "reconstruction_only"),
            ("Graph", "raw_graph_only"),
            ("Full", "full_dygma"),
        ]
        for label, variant in evidence_map:
            variant_frame = pilot_summary[pilot_summary["variant"] == variant]
            benign = variant_frame.loc[variant_frame["condition"] == "benign_drift", "effect_size"]
            topology = variant_frame.loc[variant_frame["condition"] == "topology_disruption", "effect_size"]
            benign_value = float(benign.iloc[0]) if not benign.empty else np.nan
            topology_value = float(topology.iloc[0]) if not topology.empty else np.nan
            effect_rows.append({
                "Evidence": label,
                "Benign": benign_value,
                "Topology": topology_value,
                "Gap": topology_value - benign_value,
            })
        report_lines.append(pd.DataFrame(effect_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    else:
        report_lines.append("(controlled pilot was not run)")

    report_lines.extend([
        "",
        "3. Table B. Sensor-level root-cause candidate ranking",
        "MRR / HR@100% / HR@150% are computed from exported Top*_Sensor rankings.",
        "------------------------------------------------------------",
    ])

    if event_metrics is not None:
        rca_table = pd.DataFrame([{
            "Dataset": dataset,
            "Events": event_metrics["Total_Events"],
            "MRR": event_metrics["MRR"],
            "HR@100%": event_metrics["HR@100%"],
            "HR@150%": event_metrics["HR@150%"],
        }])
        report_lines.append(rca_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        report_lines.append(f"RCA source file: {event_metrics.get('Source_File', '')}")
    else:
        report_lines.append("(event RCA was not run)")

    report_lines.extend([
        "============================================================",
    ])
    report = "\n".join(report_lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Saved combined RCA report  : {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="SWaT RCA analysis: event HR/MRR and controlled topology pilot.")
    parser.add_argument(
        "--analysis",
        choices=["event_rca", "controlled_pilot", "both"],
        default="event_rca",
        help="Run event-level RCA evaluation, controlled pilot, or both.",
    )
    parser.add_argument(
        "--scores_file",
        default=os.path.join("checkpoints", "SWaT_win100_in51_out51_batch32_patch1_ep10_scores.csv"),
    )
    parser.add_argument(
        "--rca_file",
        default=None,
        help="Optional RCA CSV used for MRR/HR ranking. If omitted, --scores_file is used.",
    )
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--focus_window", type=int, default=180)
    parser.add_argument("--data_path", type=str, default=os.path.join("dataset", "SWaT"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "SWaT_win100_in51_out51_batch32_patch1_ep10_checkpoint.pth"),
    )
    parser.add_argument("--pilot_output_dir", type=str, default=None)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--patch_len", type=int, default=1)
    parser.add_argument("--input_c", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_windows", type=int, default=256)
    parser.add_argument("--perturb_ratio", type=float, default=0.45)
    parser.add_argument("--drift_strength", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    event_metrics = None
    pilot_summary = None

    if args.analysis in {"event_rca", "both"}:
        rca_file = resolve_rca_source(args.scores_file, args.rca_file)
        event_metrics = evaluate(
            rca_file,
            args.output_dir,
            args.focus_window,
            write_report=(args.analysis == "event_rca"),
        )
    if args.analysis in {"controlled_pilot", "both"}:
        pilot_summary = run_controlled_pilot(args)
    if args.analysis in {"controlled_pilot", "both"}:
        combined_output_dir = args.pilot_output_dir or args.output_dir
        write_combined_analysis("SWaT", combined_output_dir, event_metrics, pilot_summary)


if __name__ == "__main__":
    main()
