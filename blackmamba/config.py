# config.py

class Config:
    SEQ_LEN = 64
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-3

    D_MODEL = 64
    D_STATE = 16
    D_CONV = 4
    EXPAND = 2

    TRAIN_FRAC = 0.8

    # Optional: path to saved model
    MODEL_PATH = "checkpoints/mamba_forecaster.pt"
