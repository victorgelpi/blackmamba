# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SequenceDataset(Dataset):
    """
    Takes a 1D time series and turns it into (sequence -> next value) pairs.
    """
    def __init__(self, series: np.ndarray, seq_len: int):
        super().__init__()
        if series.ndim != 1:
            raise ValueError("series must be a 1D array")

        self.series = torch.tensor(series, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]        # (seq_len,)
        y = self.series[idx + self.seq_len]              # scalar
        x = x.unsqueeze(-1)                              # (seq_len, 1)
        return x, y


def build_dataloaders(
    series: np.ndarray,
    seq_len: int,
    batch_size: int = 128,
    train_frac: float = 0.8,
):
    """
    Splits the series into train/val loaders for next-step prediction.
    """
    dataset = SequenceDataset(series, seq_len)

    train_size = int(train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader
