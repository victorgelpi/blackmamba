# data_loader.py

import pandas as pd
import numpy as np

def load_polymarket_prices(path):
    """
    Expects a JSON or CSV with a 'price' column.
    """
    if path.endswith(".json"):
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    assert "price" in df.columns, "Data must have a 'price' column"
    series = df["price"].values.astype(float)
    return series
