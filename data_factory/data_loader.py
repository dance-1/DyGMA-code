import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def read_csv_values(path):
    frame = pd.read_csv(path)
    values = frame.values[:, 1:]
    return np.nan_to_num(values)


def read_first_column(path):
    return pd.read_csv(path, usecols=[0]).iloc[:, 0].to_numpy()


def default_feature_names(num_features):
    return [f"Sensor_{idx + 1}" for idx in range(num_features)]


class BaseSegLoader:
    def __init__(self, win_size, step, mode):
        self.win_size = win_size
        self.step = step
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.mode == "score":
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        raise ValueError(f"Unsupported loader mode: {self.mode}")

    def __getitem__(self, index):
        start = index * self.step

        if self.mode == "train":
            data = self.train[start:start + self.win_size]
            label = self.test_labels[:self.win_size]
        elif self.mode == "val":
            data = self.val[start:start + self.win_size]
            label = self.test_labels[:self.win_size]
        elif self.mode == "score":
            start = index * self.win_size
            data = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            raise ValueError(f"Unsupported loader mode: {self.mode}")

        return np.float32(data), np.float32(label)


class CSVSegLoader(BaseSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)
        self.scaler = StandardScaler()

        train_data = read_csv_values(os.path.join(data_path, "train.csv"))
        self.feature_names = list(pd.read_csv(os.path.join(data_path, "train.csv"), nrows=0).columns[1:])
        self.scaler.fit(train_data)
        self.adjust_scaler()

        self.train = self.postprocess(self.scaler.transform(train_data))

        test_data = read_csv_values(os.path.join(data_path, "test.csv"))
        self.test_timestamps = read_first_column(os.path.join(data_path, "test.csv"))
        self.test = self.postprocess(self.scaler.transform(test_data))
        self.val = self.test
        self.test_labels = read_csv_values(os.path.join(data_path, "test_label.csv"))

        print(f"{self.dataset_name} train:", self.train.shape)
        print(f"{self.dataset_name} test:", self.test.shape)

    def adjust_scaler(self):
        return None

    def postprocess(self, data):
        return data


class PSMSegLoader(CSVSegLoader):
    dataset_name = "PSM"


class SWaTSegLoader(CSVSegLoader):
    dataset_name = "SWaT"


class HAISegLoader(CSVSegLoader):
    dataset_name = "HAI"

    def adjust_scaler(self):
        # Avoid division by zero for constant sensors.
        self.scaler.scale_ = np.where(self.scaler.scale_ == 0.0, 1e-8, self.scaler.scale_)


class WADISegLoader(CSVSegLoader):
    dataset_name = "WADI"

    def adjust_scaler(self):
        # Stabilize near-constant sensors before standardization.
        self.scaler.scale_ = np.where(self.scaler.scale_ < 1e-3, 1.0, self.scaler.scale_)

    def postprocess(self, data):
        # Clip extreme standardized values caused by unstable sensor scales.
        return np.clip(data, -10, 10)


class NpySegLoader(BaseSegLoader):
    dataset_name = None

    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(data_path, f"{self.dataset_name}_train.npy"))
        self.feature_names = default_feature_names(train_data.shape[1])
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, f"{self.dataset_name}_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.test_timestamps = np.arange(len(test_data))
        self.test_labels = np.load(os.path.join(data_path, f"{self.dataset_name}_test_label.npy"))

        self.set_train_val(train_data)

        print(f"{self.dataset_name} train:", self.train.shape)
        print(f"{self.dataset_name} test:", self.test.shape)
        print(f"{self.dataset_name} val:", self.val.shape)

    def set_train_val(self, train_data):
        self.train = train_data
        self.val = self.test


class MSLSegLoader(NpySegLoader):
    dataset_name = "MSL"


class SMDSegLoader(NpySegLoader):
    dataset_name = "SMD"

    def set_train_val(self, train_data):
        split_idx = int(len(train_data) * 0.8)
        self.train = train_data[:split_idx]
        self.val = train_data[split_idx:]


DATASET_LOADERS = {
    "MSL": MSLSegLoader,
    "SMD": SMDSegLoader,
    "PSM": PSMSegLoader,
    "SWaT": SWaTSegLoader,
    "HAI": HAISegLoader,
    "WADI": WADISegLoader,
}


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode="train", dataset="MSL"):
    if dataset not in DATASET_LOADERS:
        supported = ", ".join(DATASET_LOADERS)
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}.")

    if mode == "score":
        actual_step = win_size
    elif dataset in {"SMD", "SWaT", "HAI", "WADI"}:
        actual_step = 10
    else:
        actual_step = 1

    segment_dataset = DATASET_LOADERS[dataset](data_path, win_size, actual_step, mode)
    return DataLoader(
        dataset=segment_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
    )
