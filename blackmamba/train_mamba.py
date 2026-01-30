# train_mamba.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import build_dataloaders
from model import MambaForecaster


def create_dummy_series(n_points: int = 20000) -> np.ndarray:
    """
    Dummy time series: replace this with real Polymarket data.
    E.g. df["price"].values.astype(float)
    """
    x = np.linspace(0, 200, n_points)
    series = np.sin(x) + 0.05 * np.random.randn(n_points)
    return series


def normalize_series(series: np.ndarray):
    mean = series.mean()
    std = series.std()
    norm = (series - mean) / std
    return norm, mean, std


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)      # (batch, seq_len, 1)
        y = y.to(device)      # (batch,)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load or create your time series
    # Replace this with real Polymarket data:
    #   e.g. series = df["price"].values.astype(float)
    raw_series = create_dummy_series(n_points=20000)

    # 2. Normalize
    series, mean, std = normalize_series(raw_series)

    # 3. Build dataloaders
    SEQ_LEN = 64
    BATCH_SIZE = 128
    train_loader, val_loader = build_dataloaders(
        series=series,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        train_frac=0.8,
    )

    # 4. Build model, loss, optimizer
    model = MambaForecaster(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    # 5. Train
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")

    # 6. Make a next-step prediction on the latest window
    model.eval()
    with torch.no_grad():
        # last SEQ_LEN points from normalized series
        last_window = torch.tensor(
            series[-SEQ_LEN:], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        last_window = last_window.to(device)
        pred_norm = model(last_window).item()

    # un-normalize
    pred = pred_norm * std + mean
    print("Next-step prediction (original scale):", pred)


if __name__ == "__main__":
    main()
