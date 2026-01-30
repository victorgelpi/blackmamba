#used after training
# predict.py

import numpy as np
import torch
from model import MambaForecaster
from utils import load_model, normalize
from evaluate import predict_next

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load your data
    series = np.load("my_series.npy")
    series, mean, std = normalize(series)

    # Rebuild model
    model = MambaForecaster().to(device)
    model = load_model(model, "checkpoints/mamba_forecaster.pt", device)

    # Predict next value
    seq_len = 64
    next_val = predict_next(model, series[-seq_len:], device)

    print("Next prediction:", next_val * std + mean)

if __name__ == "__main__":
    main()
