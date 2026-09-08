import torch
import torch.nn as nn

HORIZONS = [15, 30, 60, 90, 120]
NATIVE_STEP_MIN = 5

# ─────────────────────────── Recurrent baselines ───────────────────────────

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(cgm.unsqueeze(-1))
        return self.head(out[:, -1, :])


class GRUPredictor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(cgm.unsqueeze(-1))
        return self.head(out[:, -1, :])


# ─────────────────────────── Forecasting-architecture baselines ───────────────────────────

def _horizon_step_indices(horizons: list = HORIZONS, native_step_min: int = NATIVE_STEP_MIN) -> list:
    return [h // native_step_min - 1 for h in horizons]


class NBEATSPredictor(nn.Module):
    def __init__(self, context_length: int = 24, hidden_dim: int = 128,
                 num_block_layers: int = 4, dropout: float = 0.1, horizons: list = HORIZONS):
        super().__init__()
        from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock
        self._horizon_idx = _horizon_step_indices(horizons)
        max_horizon_steps = max(horizons) // NATIVE_STEP_MIN
        self.block = NBEATSGenericBlock(
            units=hidden_dim, thetas_dim=hidden_dim, num_block_layers=num_block_layers,
            dropout=dropout, backcast_length=context_length, forecast_length=max_horizon_steps,
        )

    def forward(self, cgm: torch.Tensor) -> torch.Tensor:
        _, forecast = self.block(cgm)          # [B, max_horizon_steps]
        return forecast[:, self._horizon_idx]  # [B, num_horizons]


class NHiTSPredictor(nn.Module):
    def __init__(self, context_length: int = 24, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.1, horizons: list = HORIZONS):
        super().__init__()
        from pytorch_forecasting.models.nhits.sub_modules import NHiTS
        self._horizon_idx = _horizon_step_indices(horizons)
        max_horizon_steps = max(horizons) // NATIVE_STEP_MIN
        pooling_sizes = [4, 2, 1]  # coarse -> fine; also used as n_freq_downsample per stack
        self.nhits = NHiTS(
            context_length=context_length, prediction_length=max_horizon_steps,
            output_size=[1], static_size=0, encoder_covariate_size=0, decoder_covariate_size=0,
            static_hidden_size=0, n_blocks=[1, 1, 1], n_layers=[n_layers] * 3,
            hidden_size=[[hidden_dim] * n_layers] * 3, pooling_sizes=pooling_sizes,
            downsample_frequencies=pooling_sizes, pooling_mode='max', interpolation_mode='linear',
            dropout=dropout, activation='ReLU', initialization='orthogonal',
            batch_normalization=False, shared_weights=False, naive_level=True,
        )

    def forward(self, cgm: torch.Tensor) -> torch.Tensor:
        encoder_y = cgm.unsqueeze(-1)                       # [B, T, 1]
        encoder_mask = torch.ones_like(cgm)                 # [B, T]
        forecast, _, _, _ = self.nhits(encoder_y, encoder_mask, None, None, None)
        return forecast.squeeze(-1)[:, self._horizon_idx]   # [B, num_horizons]
