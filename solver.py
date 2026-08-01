import torch
import torch.nn as nn
import numpy as np
import os
import time
from artifact_io import save_score_artifact
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

    def rca_train_score_statistics(self, criterion):
        """Estimate train-distribution statistics for standardized ranking."""
        score_sum = torch.zeros(self.input_c, device=self.device)
        score_sq_sum = torch.zeros(self.input_c, device=self.device)
        score_count = 0

        self.model.eval()
        with torch.no_grad():
            for input_data, _ in self.train_loader:
                _, sensor_scores = self.anomaly_energy(
                    input_data, criterion, return_sensor_scores=True
                )
                score_sum += sensor_scores.sum(dim=(0, 1))
                score_sq_sum += (sensor_scores ** 2).sum(dim=(0, 1))
                score_count += sensor_scores.shape[0] * sensor_scores.shape[1]

        if score_count == 0:
            raise ValueError("Cannot compute RCA statistics from an empty train loader.")

        score_mean = score_sum / score_count
        score_var = torch.clamp(score_sq_sum / score_count - score_mean ** 2, min=0.0)
        return score_mean, torch.sqrt(score_var)

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
        path = self.checkpoint_dir
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(self.train_loader)

        prefix = self.run_name()
        log_file = os.path.join(self.output_dir, f"train_log_{prefix}.txt")

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
            checkpoint_path = os.path.join(str(self.checkpoint_dir), model_name)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.endswith("total_ops") and not key.endswith("total_params")
            }
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint/model mismatch for {checkpoint_path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )

        self.model.eval()

        profiling_metrics = self.measure_efficiency()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        ranking_mode = "standardized" if self.dataset in {"SWaT", "WADI"} else "raw"
        feature_names = self.feature_names()

        calibration_mean = None
        calibration_std = None
        if ranking_mode == "standardized":
            print("[RCA] Computing train-distribution statistics...")
            calibration_mean, calibration_std = self.rca_train_score_statistics(criterion)
        print(f"[RCA] Exporting {ranking_mode.upper()} per-sensor rankings...")

        with torch.no_grad():
            attens_energy = []
            test_labels = []
            sensor_rankings = []

            for i, (input_data, labels) in enumerate(self.score_loader):
                cri, sensor_scores = self.anomaly_energy(
                    input_data, criterion, return_sensor_scores=True
                )
                ranking_scores = sensor_scores
                if ranking_mode == "standardized":
                    ranking_scores = (
                        sensor_scores - calibration_mean
                    ) / (calibration_std + 1e-5)
                _, top_indices = torch.topk(
                    ranking_scores, k=self.input_c, dim=-1
                )
                sensor_rankings.append(
                    top_indices.reshape(-1, self.input_c).detach().cpu().numpy()
                )
                attens_energy.append(cri.detach().cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        gt = np.concatenate(test_labels, axis=0).reshape(-1).astype(int)

        smoothing_window = int(self.patch_len)

        test_energy = np.convolve(test_energy, np.ones(smoothing_window)/smoothing_window, mode='same')

        prefix = self.run_name()
        artifact_path = os.path.join(self.output_dir, f'{prefix}_scores.npy')
        report_save_path = os.path.join(self.output_dir, f"Report_{prefix}.txt")
        timestamps = self.score_timestamps(len(test_energy))
        if timestamps is None:
            timestamps = np.arange(len(test_energy))
        all_rankings = np.concatenate(sensor_rankings, axis=0)[:len(test_energy)]

        save_score_artifact(
            artifact_path,
            dataset=self.dataset,
            anomaly_score=test_energy,
            ground_truth=gt,
            timestamps=timestamps,
            feature_names=feature_names,
            sensor_rankings=all_rankings,
            ranking_mode=ranking_mode,
            train_score_mean=(
                None
                if calibration_mean is None
                else calibration_mean.detach().cpu().numpy()
            ),
            train_score_std=(
                None
                if calibration_std is None
                else calibration_std.detach().cpu().numpy()
            ),
            metadata={
                "win_size": self.win_size,
                "patch_len": self.patch_len,
                "batch_size": self.batch_size,
                "epoch": self.num_epochs,
                "input_c": self.input_c,
                "output_c": self.output_c,
            },
        )
        print(f"Scores and RCA rankings saved to: {artifact_path}")

        self.write_efficiency_report(report_save_path, prefix, profiling_metrics)
        print(f"Profiling report saved to: {report_save_path}")

        return artifact_path

