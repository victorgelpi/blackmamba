# evaluate.py

import torch
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def predict_next(model, window, device="cpu"):
    """
    window: (seq_len,) numpy array
    """
    model.eval()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    x = x.to(device)

    with torch.no_grad():
        pred = model(x).item()

    return pred

def predict_n_steps(model, series, n_steps, seq_len, device="cpu"):
    """
    Autoregressive multi-step forecast.
    """
    window = series[-seq_len:].copy()
    preds = []

    for _ in range(n_steps):
        next_val = predict_next(model, window, device)
        preds.append(next_val)
        window = np.append(window[1:], next_val)

    return preds
