````markdown
# mamba-forecast

Mamba-based time series forecasting for Polymarket.

This project is a small, modular PyTorch package that uses a **Mamba state-space model** to forecast the next value (or next few values) in a time series.  
It is designed to plug nicely into workflows where you already have a data pipeline (e.g., a Polymarket data collection toolkit) and want to experiment with **modern sequence models** on top of it.

---

## What is Mamba (in this context)?

**Mamba** is a next-generation **state-space neural network** architecture that can model long sequences efficiently, similar to Transformers but with:

- Recurrent-style dynamics (state updates) instead of full attention
- Better scaling to long contexts
- Strong empirical performance on sequence tasks

In this project, you don’t need to know the internal math of Mamba.  
You use it like any other PyTorch layer (similar to an LSTM or Transformer block), via the `mamba-ssm` library:

- We wrap a single `Mamba` block inside a forecasting model (`MambaForecaster`)
- You pass a sequence of past values, and it predicts the **next value** in the sequence

---

## Goal of this Package

The main goal is to provide a **clean, research-friendly codebase** that lets you:

- Take a  time series (e.g., Polymarket prices, volume, or other numeric signals)
- Turn it into supervised learning data (sequence → next step)
- Train a Mamba-based model to forecast the next point
- Evaluate performance and generate multi-step forecasts
- Easily swap the synthetic “toy” data with real Polymarket data

You can use this as:

- A **teaching/demo project** for Mamba on time series
- A **prototype** for more serious forecasting
- A **submodule** inside a larger project (e.g., your IEOR4212 Polymarket calibration analysis)

---

## Features

- **MambaForecaster model**
  - Built on top of `mamba-ssm`
  - Input: `(batch, seq_len, 1)` → Output: `(batch,)` next-step prediction
  - Uses a simple input projection → Mamba block → output head

- **Reusable SequenceDataset**
  - Converts a raw `numpy` array into supervised pairs:
    - Input: last `seq_len` values
    - Target: the next value
  - Works with any numeric time series, not just Polymarket

- **Training & evaluation utilities**
  - Standard PyTorch training loop with:
    - MSE loss
    - Train / validation split
    - Mini-batch loading
  - Runs on GPU if available (CUDA)

- **Utility helpers**
  - Normalization (mean/std)
  - Saving/loading model checkpoints
  - Multi-step forecasting helpers (iterative next-step prediction)

- **Modular design**
  - Clean separation between:
    - Dataset logic
    - Model architecture
    - Training code
    - Evaluation & prediction scripts
    - Configuration (hyperparameters, paths)
  - Easy to extend or integrate into larger pipelines

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd <your-repo-name>
````

Replace `<your-repo-url>` and `<your-repo-name>` with your actual GitHub repo URL and directory name.

### 2. Install the package in editable mode

Editable mode is recommended during development so that code changes take effect immediately.

```bash
pip install --upgrade pip
pip install -e .
```

This will install the `mamba-forecast` package and all declared dependencies.

> **Note:** You must have a working PyTorch installation.
> On GPU machines (like Google Colab), `mamba-ssm` will automatically use CUDA if available.

---

## Requirements

These are handled via `setup.py`, but conceptually the project depends on:

* **Python**: 3.9+
* **PyTorch**: for model training and GPU support
* **mamba-ssm**: the Mamba state-space model implementation
* **numpy**: numerical operations
* **pandas**: convenient loading of CSV/JSON data (optional but recommended)

On **Google Colab**, a typical setup looks like:

```bash
pip install --upgrade pip
pip install "mamba-ssm[causal-conv]" --no-build-isolation
pip install -e .
```

The first line ensures you have a modern `pip`.
The second installs `mamba-ssm` (and optional causal-conv kernels).
The third installs this project locally in editable mode.

---

## Project Structure

A recommended structure for this project is:

```text
your-repo/
│
├── setup.py
├── README.md
│
├── mamba_forecast/
│   ├── __init__.py
│   ├── config.py          # central hyperparameters & paths
│   ├── dataset.py         # SequenceDataset + dataloader builder
│   ├── model.py           # MambaForecaster architecture
│   ├── train_utils.py     # epoch-level train/eval helpers
│   ├── evaluate.py        # metrics & multi-step forecasting functions
│   ├── data_loader.py     # helpers to load real time series (e.g., Polymarket)
│   ├── utils.py           # saving/loading models, seeding, normalization
│   ├── train_mamba.py     # main training script / entrypoint
│   └── predict.py         # script to load a model and make predictions
│
├── data/
│   └── polymarket_prices.csv   # example or real Polymarket data (optional)
│
└── checkpoints/
    └── mamba_forecaster.pt     # saved model weights after training
```

**Key idea:**
Everything under `mamba_forecast/` is importable as a Python package, so you can do:

```python
from mamba_forecast.model import MambaForecaster
from mamba_forecast.dataset import build_dataloaders
from mamba_forecast.utils import save_model, load_model
```

---

## Data Assumptions

The core code assumes you have:

* A ** time series**: for example, a list or array of prices over time.

Formally:

* `series`: a `numpy.ndarray` of shape `(N,)` containing floats.

You can generate synthetic data (e.g., sine waves) or load real data from CSV/JSON via `pandas`.

Typical Polymarket-style data structure for this package:

* A CSV file with at least one column named `price`, e.g.:

```text
timestamp,price
2025-01-01T00:00:00Z,0.42
2025-01-01T01:00:00Z,0.45
...
```

In that case, `data_loader.py` can load the series as:

```python
import pandas as pd

def load_polymarket_prices(path):
    df = pd.read_csv(path)
    series = df["price"].values.astype(float)
    return series
```

---

## Core Components

### 1. SequenceDataset (`dataset.py`)

* Converts a  time series into supervised learning data.
* For each index `i`, it constructs:

  * Input `x`: `series[i : i + seq_len]` (a window of past values)
  * Target `y`: `series[i + seq_len]` (the next value)
* Outputs tensors shaped as:

  * `x`: `(seq_len, 1)`
  * `y`: scalar

`build_dataloaders()` then splits the dataset into train/validation sets and wraps them in PyTorch `DataLoader` objects.

---

### 2. MambaForecaster (`model.py`)

* Neural network model for next-step forecasting.

* Structure:

  * **Input projection**: `Linear(1 → d_model)`
    Converts scalar inputs into a higher-dimensional representation.
  * **Mamba block**: a single `Mamba` layer from `mamba-ssm`
    Processes the entire sequence and captures temporal dependencies.
  * **Output head**: `Linear(d_model → 1)` applied to the **last token**
    Produces the predicted next value.

* Forward pass:

  * Input: `(batch, seq_len, 1)`
  * Output: `(batch,)`

---

### 3. Training Utilities (`train_utils.py`)

* Contains functions like:

  * `train_epoch(model, loader, optimizer, criterion, device)`
  * `eval_epoch(model, loader, criterion, device)`

These handle the standard PyTorch training loop details:

* Moving data to GPU/CPU
* Forward pass
* Backpropagation
* Loss accumulation

---

### 4. Evaluation Tools (`evaluate.py`)

* Functions for:

  * Computing metrics (MSE, MAE)
  * Conducting **multi-step forecasts** by iteratively predicting and feeding back predictions
* Example: `predict_n_steps(model, series, n_steps, seq_len, device)`:

  * Takes the last `seq_len` points of the series
  * Predicts one step ahead, appends the prediction, slides the window, and repeats

---

### 5. Utilities (`utils.py`)

Typical helpers include:

* `set_seed(seed)` – for reproducible experiments
* `save_model(model, path)` – to save model weights
* `load_model(model, path, device)` – to load a saved checkpoint
* `normalize(series)` – returns normalized series + mean/std for later de-normalization

---

### 6. Configuration (`config.py`)

Centralizes key hyperparameters and paths:

* Sequence length (`SEQ_LEN`)
* Batch size (`BATCH_SIZE`)
* Learning rate (`LR`)
* Number of epochs (`EPOCHS`)
* Mamba architecture parameters (`D_MODEL`, `D_STATE`, etc.)
* Train/validation split fraction
* Default model checkpoint path

This allows you to adjust experiments in one place.

---

## Usage

### 1. Train a Mamba Forecaster

From the repo root:

```bash
python -m mamba_forecast.train_mamba
```

The `train_mamba` script typically does the following:

1. Loads or creates a time series:

   * By default, possibly a synthetic sine wave (for quick testing)
   * Optionally replaced with real Polymarket data via `data_loader.py`
2. Normalizes the series (mean 0, std 1)
3. Builds train/validation dataloaders from the series
4. Initializes a `MambaForecaster` model
5. Trains for a specified number of epochs
6. Prints:

   * Training and validation losses per epoch
   * A final next-step prediction at the end of training

You can customize behavior by editing:

* `config.py` for hyperparameters
* `train_mamba.py` for data loading logic and training specifics

---

### 2. Using Real Polymarket Data

In `mamba_forecast/data_loader.py`, implement a loader like:

```python
def load_polymarket_prices(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df["price"].values.astype(float)
```

Then in `train_mamba.py`, replace the dummy series generator with:

```python
from mamba_forecast.data_loader import load_polymarket_prices

series = load_polymarket_prices("data/polymarket_prices.csv")
```

Now, when you run:

```bash
python -m mamba_forecast.train_mamba
```

you will be training a Mamba model directly on your Polymarket price history.

---

### 3. Making Predictions After Training

Once you’ve trained and saved a model (e.g., to `checkpoints/mamba_forecaster.pt`), you can use:

```bash
python -m mamba_forecast.predict
```

Typical behavior of `predict.py`:

1. Load the **same** time series used for training (or a continuation of it)
2. Normalize it using stored or recomputed mean/std
3. Rebuild the `MambaForecaster` with the same architecture
4. Load the saved model weights from `checkpoints/mamba_forecaster.pt`
5. Print:

   * A next-step prediction (de-normalized back to the original scale)
   * Optionally, multi-step forecasts using the helper functions in `evaluate.py`

---

## How This Fits Into a Polymarket Pipeline

A typical Polymarket research workflow might look like:

1. **Data ingestion**

   * Use a separate toolkit (e.g., `datacollection`) to pull Polymarket events and price history.
   * Save clean, processed time series to a CSV/JSON file.

2. **Exploratory analysis & calibration**

   * Use separate code (e.g., `IEOR4212` project) to analyze calibration, bias, category differences, etc.

3. **Forecasting with Mamba (this package)**

   * Load the price history (or derived features) as a  time series.
   * Use `mamba-forecast` to:

     * Train a Mamba model
     * Forecast future price movement
     * Experiment with different hyperparameters and settings

Because this package is **modular**, you can:

* swap in different time series,
* plug in multiple Mamba layers,
* or replace the output head with classification (e.g., predict “up vs down” instead of a raw price).

---

## Extending the Project

Some natural extensions:

* Use **multi-feature inputs** (price, volume, open interest, etc.)

  * Change input dimension from 1 to `d_in`, adjust input projection accordingly.
* Predict **multi-step outputs** directly instead of just the next value.
* Use **multiple Mamba blocks** stacked for deeper models.
* Integrate with **hyperparameter search** tools to tune architecture and training parameters.
* Add **visualization scripts** to plot:

  * true vs predicted series
  * residuals
  * error metrics over time

---

## License

MIT

---

## Author

Internal project using Mamba for time series forecasting on Polymarket-style data (or any other  numeric sequences).

```
::contentReference[oaicite:0]{index=0}
```

