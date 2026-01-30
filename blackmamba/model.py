# model.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaForecaster(nn.Module):
    """
    Simple forecaster:
      input:  (batch, seq_len, 1)   -> price or any 1D feature
      output: (batch,)              -> next value prediction
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 1)
        returns: (batch,)
        """
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.mamba(x)            # (batch, seq_len, d_model)
        last_token = x[:, -1, :]     # (batch, d_model)
        y_hat = self.output_head(last_token)  # (batch, 1)
        return y_hat.squeeze(-1)     # (batch,)
