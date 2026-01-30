# utils.py

import torch
import numpy as np
import os

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def normalize(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std, mean, std
