# experiment.py

from train_mamba import train_model_with_config
from config import Config

def run_experiment():
    configs = [
        {"d_model": 32},
        {"d_model": 64},
        {"d_model": 128},
    ]

    for cfg in configs:
        print("Running config:", cfg)
        train_model_with_config(cfg)

if __name__ == "__main__":
    run_experiment()
