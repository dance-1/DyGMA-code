import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from model.DyGMA import DyGMAModel
from data_factory.data_loader import get_loader_segment


# KL divergence averaged over sequence positions.
def my_kl_loss(p, q):

    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))

    return torch.mean(torch.sum(res, dim=-1), dim=1)

# Per-variable KL divergence for sensor-level scoring.
def my_kl_loss_var(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

# Step-wise learning-rate decay used by the training loop.
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

# Orchestrates data loading, model training, checkpointing, and evaluation.
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)

        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)

        self.score_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='score',
                                               dataset=self.dataset)

        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.criterion = nn.MSELoss()

    # Build the DyGMA temporal-spatial model.
    def build_model(self):

        self.model = DyGMAModel(
            win_size=self.win_size,
            num_vars=self.input_c,
            patch_len=self.patch_len,
            d_model=128)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Built! Total Parameters: {self.total_params / 1e6:.4f} M")

    def run_name(self, epoch=None):
        ep_str = self.num_epochs if epoch is None else epoch
        return (
            f"{self.dataset}_win{self.win_size}_in{self.input_c}_out{self.output_c}"
            f"_batch{self.batch_size}_patch{self.patch_len}_ep{ep_str}"
        )

    def adaptive_weight(self, series, prior):
        entropy_series = -torch.sum(series * torch.log(series + 1e-5), dim=-1)
        entropy_prior = -torch.sum(prior.detach() * torch.log(prior.detach() + 1e-5), dim=-1)
        confidence = 1.0 - (entropy_series / (entropy_prior + 1e-5))
        confidence = torch.clamp(confidence, min=0.0, max=1.0)
        return torch.exp(torch.mean(confidence).detach())

    def anomaly_energy(self, input_data, criterion, return_sensor_scores=False):
        input = input_data.float().to(self.device)
        output, series, prior = self.model(input)

        loss_raw = criterion(input, output)
        series_loss = my_kl_loss_var(series, prior.detach()) + my_kl_loss_var(prior.detach(), series)
        prior_loss = my_kl_loss_var(prior, series.detach()) + my_kl_loss_var(series.detach(), prior)
        focused_kl = (series_loss + prior_loss).unsqueeze(1)

        mean_loss = torch.mean(loss_raw, dim=1, keepdim=True)
        temporal_weight = torch.clamp(loss_raw / (mean_loss + 1e-5), min=1.0)
        focused_kl = focused_kl * temporal_weight

        cri_var = loss_raw + focused_kl * self.adaptive_weight(series, prior)
        top_k = max(1, int(0.10 * cri_var.shape[-1]))
        energy = torch.topk(cri_var, k=top_k, dim=-1)[0].mean(dim=-1)

        if return_sensor_scores:
            return energy, cri_var
        return energy

    def feature_names(self):
        return getattr(
            self.score_loader.dataset,
            "feature_names",
            [f"Sensor_{idx + 1}" for idx in range(self.input_c)],
        )[:self.input_c]

    def score_timestamps(self, length):
        timestamps = getattr(self.score_loader.dataset, "test_timestamps", None)
        if timestamps is None:
            return None
        return timestamps[:length]

    def measure_efficiency(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print("[Efficiency Profiling] Measuring latency, memory, and FLOPs with batch size 1...")
        try:
            from thop import profile
        except ImportError:
            print("[Warning] thop is not installed. Run 'pip install thop' to calculate FLOPs.")
            profile = None

        dummy_input = torch.randn(1, self.win_size, self.input_c).float().to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                for _ in range(30):
                    _ = self.model(dummy_input)

            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            latencies = []
            with torch.no_grad():
                for _ in range(100):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    torch.cuda.synchronize()
                    latencies.append((time.time() - start_time) * 1000)
            avg_latency = np.mean(latencies)
        else:
            peak_memory_mb = 0.0
            avg_latency = 0.0

        flops_m = 0.0
        if profile is not None:
            macs, _ = profile(self.model, inputs=(dummy_input,), verbose=False)
            flops_m = macs / 1e6

        return {
            "Model Parameters (M)": round(self.total_params / 1e6, 6),
            "Model FLOPs (M)": round(flops_m, 4),
            "Inference Latency (ms)": round(avg_latency, 4),
            "GPU Peak Memory (MB)": round(peak_memory_mb, 4),
        }

    def write_efficiency_report(self, report_path, run_name, metrics):
        report = (
            "============================================================\n"
            f"Data from: {run_name}\n"
            "============================================================\n\n"
            "Efficiency Metrics\n"
            "------------------------------------------------------------\n"
            f"    -> Model Parameters (M)      : {metrics['Model Parameters (M)']}\n"
            f"    -> Model FLOPs (M)           : {metrics['Model FLOPs (M)']}\n"
            f"    -> Inference Latency (ms)    : {metrics['Inference Latency (ms)']}\n"
            f"    -> GPU Peak Memory (MB)      : {metrics['GPU Peak Memory (MB)']}\n"
            "============================================================\n"
        )
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(report)

    # Validation uses reconstruction loss only.
    def vali(self, vali_loader):
        self.model.eval()

        rec_losses = []
        for input_data, _ in vali_loader:
            batch = input_data.float().to(self.device)
            output, _, _ = self.model(batch)
            rec_loss = self.criterion(output, batch)
            rec_losses.append(rec_loss.item())

        return np.average(rec_losses)

    # Train for the requested number of epochs and save one final checkpoint.
    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(self.train_loader)

        prefix = self.run_name()
        log_file = os.path.join(self.model_save_path, f"train_log_{prefix}.txt")

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"Model Parameters : {self.total_params / 1e6:.4f} M\n")
            f.write(f"Window Size      : {self.win_size}\n")
            f.write(f"batch_size      : {self.batch_size}\n")
            f.write(f"input_c      : {self.input_c}\n")
            f.write(f"output_c      : {self.output_c}\n")
            f.write(f"num_epochs      : {self.num_epochs}\n")
            f.write("="*50 + "\n")

            f.write("Epoch,Steps,Train_Loss,Vali_Loss,Epoch_Time(s)\n")

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()

            for i, (input_data, _) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                batch = input_data.float().to(self.device)

                output, series, prior = self.model(batch)

                series_loss = torch.mean(my_kl_loss(series, prior.detach())) + \
                                torch.mean(my_kl_loss(prior.detach(), series))

                prior_loss = torch.mean(my_kl_loss(prior, series.detach())) + \
                                torch.mean(my_kl_loss(series.detach(), prior))

                rec_loss = self.criterion(output, batch)

                adaptive_weight = self.adaptive_weight(series, prior)

                loss1 = rec_loss - adaptive_weight * series_loss
                loss2 = rec_loss + adaptive_weight * prior_loss

                loss1_list.append(loss1.item())

                total_loss = loss1 + loss2

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            cost_time = time.time() - epoch_time

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, cost_time))

            train_loss = np.average(loss1_list)

            vali_rec_loss = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss1: {2:.7f} Vali RecLoss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_rec_loss))

            log_script = f"{epoch + 1},{train_steps},{train_loss:.7f},{vali_rec_loss:.7f},{cost_time:.2f}"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_script + "\n")

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        file_name = f"{self.run_name()}_checkpoint.pth"
        save_path = os.path.join(path, file_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Training finished. Final checkpoint saved to: {save_path}")

    # Evaluate the final checkpoint and export anomaly scores plus efficiency metrics.
    def test(self, load_model=True):
        if load_model:

            model_name = f"{self.run_name()}_checkpoint.pth"
            self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), model_name)), strict=False)

        self.model.eval()

        profiling_metrics = self.measure_efficiency()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        # Optional RCA export appends timestamp and TopK sensor rankings to scores.csv.
        export_rca = bool(getattr(self, "export_rca", False))
        feature_names = self.feature_names()

        if export_rca:
            print("[RCA] Exporting raw per-sensor score rankings...")

        with torch.no_grad():
            attens_energy = []
            test_labels = []
            sensor_rankings = []

            for i, (input_data, labels) in enumerate(self.score_loader):
                if export_rca:
                    cri, sensor_scores = self.anomaly_energy(
                        input_data, criterion, return_sensor_scores=True
                    )
                    _, top_indices = torch.topk(sensor_scores, k=self.input_c, dim=-1)
                    sensor_rankings.append(top_indices.reshape(-1, self.input_c).detach().cpu().numpy())
                else:
                    cri = self.anomaly_energy(input_data, criterion)
                attens_energy.append(cri.detach().cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        gt = np.concatenate(test_labels, axis=0).reshape(-1).astype(int)

        smoothing_window = int(self.patch_len)

        test_energy = np.convolve(test_energy, np.ones(smoothing_window)/smoothing_window, mode='same')

        score_df = pd.DataFrame({
            'Time': range(len(test_energy)),
            'Anomaly_Score': test_energy,
            'Ground_Truth': gt
        })
        if export_rca:
            timestamps = self.score_timestamps(len(score_df))
            if timestamps is not None:
                score_df.insert(1, "Timestamp", timestamps)

            all_rankings = np.concatenate(sensor_rankings, axis=0)[:len(score_df)]
            for rank_idx in range(self.input_c):
                score_df[f"Top{rank_idx + 1}_Sensor"] = [
                    feature_names[row[rank_idx]] if label == 1 else ""
                    for row, label in zip(all_rankings, gt)
                ]

        prefix = self.run_name()
        csv_save_path = os.path.join(self.model_save_path, f'{prefix}_scores.csv')
        report_save_path = os.path.join(self.model_save_path, f"Report_{prefix}.txt")

        score_df.to_csv(csv_save_path, index=False)
        print(f"Anomaly scores saved to: {csv_save_path}")

        self.write_efficiency_report(report_save_path, prefix, profiling_metrics)
        print(f"Profiling report saved to: {report_save_path}")

        return csv_save_path

