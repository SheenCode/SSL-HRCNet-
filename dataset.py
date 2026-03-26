import os
import pickle
import random

import numpy as np
import torch
from scipy.interpolate import splrep, splev


def scaler(x):
    """
    Standardize 1D signal.
    """
    x = np.asarray(x, dtype=np.float32)
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


class UnifiedApneaDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for:
        1) Apnea-ECG
        2) UCDDB

    Args:
        dataset_name (str): 'apnea-ecg' or 'ucddb'
        split (str): 'train', 'val', or 'test'
        base_dir (str): directory containing pkl files
        interp_fs (int|float): interpolation frequency, default = 3 Hz
        val_ratio (float): validation split ratio for training data
        seed (int): random seed for reproducible split
        return_domain (bool): whether to return domain/group label
    """

    def __init__(
        self,
        dataset_name="apnea-ecg",
        split="train",
        base_dir="./",
        interp_fs=3,
        val_ratio=0.2,
        seed=42,
        return_domain=False,
    ):
        super().__init__()

        self.dataset_name = dataset_name.lower()
        self.split = split.lower()
        self.base_dir = base_dir
        self.interp_fs = interp_fs
        self.val_ratio = val_ratio
        self.seed = seed
        self.return_domain = return_domain

        if self.dataset_name not in ["apnea-ecg", "ucddb"]:
            raise ValueError("dataset_name must be 'apnea-ecg' or 'ucddb'")

        if self.split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Interpolation timeline: 5 minutes with 3 Hz by default
        self.tm = np.arange(0, 5 * 60, step=1 / float(self.interp_fs))

        # Load dataset
        if self.dataset_name == "apnea-ecg":
            self._load_apnea_ecg()
        else:
            self._load_ucddb()

    def _interpolate_sample(self, sample):
        """
        Interpolate one sample:
            sample = [(rri_tm, rri_signal), (ampl_tm, ampl_signal)]
        Returns:
            np.ndarray with shape [2, T]
        """
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = sample

        rri_interp_signal = splev(
            self.tm,
            splrep(rri_tm, scaler(rri_signal), k=3),
            ext=1
        )
        ampl_interp_signal = splev(
            self.tm,
            splrep(ampl_tm, scaler(ampl_signal), k=3),
            ext=1
        )

        return np.array([rri_interp_signal, ampl_interp_signal], dtype=np.float32)

    def _make_train_val_split(self, n):
        """
        Create reproducible train/val split.
        """
        indices = list(range(n))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        train_size = int(n * (1 - self.val_ratio))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        return train_idx, val_idx

    def _load_apnea_ecg(self):
        """
        Load Apnea-ECG dataset from apnea-ecg.pkl
        Expected keys:
            o_train, y_train, groups_train
            o_test, y_test, groups_test
        """
        path = os.path.join(self.base_dir, "apnea-ecg.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)

        o_train = data["o_train"]
        y_train = np.array(data["y_train"])
        groups_train = np.array(data["groups_train"], dtype=object)

        o_test = data["o_test"]
        y_test = np.array(data["y_test"])
        groups_test = np.array(data["groups_test"], dtype=object)

        x_train = np.array([self._interpolate_sample(sample) for sample in o_train], dtype=np.float32)
        x_test = np.array([self._interpolate_sample(sample) for sample in o_test], dtype=np.float32)

        train_idx, val_idx = self._make_train_val_split(len(y_train))

        if self.split == "train":
            self.x = torch.tensor(x_train[train_idx], dtype=torch.float32)
            self.y = torch.tensor(y_train[train_idx], dtype=torch.long)
            self.domain_y = groups_train[train_idx]

        elif self.split == "val":
            self.x = torch.tensor(x_train[val_idx], dtype=torch.float32)
            self.y = torch.tensor(y_train[val_idx], dtype=torch.long)
            self.domain_y = groups_train[val_idx]

        elif self.split == "test":
            self.x = torch.tensor(x_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.long)
            self.domain_y = groups_test

    def _load_ucddb(self):
        """
        Load UCDDB dataset from UCDDB.pkl
        Expected keys:
            X_train, y_train
        candidate_files = ["UCDDB.pkl", "UCCDB.pkl"]
        path = None
        for fname in candidate_files:
            full_path = os.path.join(self.base_dir, fname)
            if os.path.exists(full_path):
                path = full_path
                break

        if path is None:
            raise FileNotFoundError("Neither 'UCDDB.pkl' nor 'UCCDB.pkl' was found.")

        with open(path, "rb") as f:
            data = pickle.load(f)

        o_train = data["X_train"]
        y_train = np.array(data["y_train"])

        x_train = np.array([self._interpolate_sample(sample) for sample in o_train], dtype=np.float32)

        train_idx, val_idx = self._make_train_val_split(len(y_train))

        if self.split == "train":
            self.x = torch.tensor(x_train[train_idx], dtype=torch.float32)
            self.y = torch.tensor(y_train[train_idx], dtype=torch.long)
            self.domain_y = None

        elif self.split == "val":
            self.x = torch.tensor(x_train[val_idx], dtype=torch.float32)
            self.y = torch.tensor(y_train[val_idx], dtype=torch.long)
            self.domain_y = None

        elif self.split == "test":
            raise ValueError("UCDDB does not provide a predefined test split in this pkl file.")

    def __getitem__(self, index):
        input_data = self.x[index]
        target = self.y[index]

        if self.return_domain and self.domain_y is not None:
            domain = self.domain_y[index]
            return input_data, target, domain

        return input_data, target

    def __len__(self):
        return len(self.x)
