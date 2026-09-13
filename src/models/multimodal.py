import torch
import torch.nn as nn

from src.models.classical_baseline import HORIZONS


class CGMTimePredictor(nn.Module):
    def __init__(self, gru_hidden_size: int = 256, gru_num_layers: int = 3,
                 gru_dropout: float = 0.2, time_proj_dim: int = 16, head_dropout: float = 0.2,
                 num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(1, gru_hidden_size, gru_num_layers, batch_first=True,
                           dropout=gru_dropout if gru_num_layers > 1 else 0.0)
        self.time_proj = nn.Sequential(nn.Linear(1, time_proj_dim), nn.ReLU())

        fusion_dim = gru_hidden_size + time_proj_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(fusion_dim // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor,
                has_meal: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(cgm.unsqueeze(-1))
        cgm_feat = h_n[-1]                                    # [B, gru_hidden_size], last layer
        time_feat = self.time_proj(time_since_meal.unsqueeze(-1))  # [B, time_proj_dim]
        gate = has_meal.unsqueeze(-1)                          # [B, 1]
        time_feat = time_feat * gate                           # no meal nearby -> exactly zero
        fused = torch.cat([cgm_feat, time_feat], dim=-1)
        return self.head(fused)


class MultimodalGRUPredictor(nn.Module):
    def __init__(self, img_feat_dim: int, gru_hidden_size: int = 256, gru_num_layers: int = 3,
                 gru_dropout: float = 0.2, time_proj_dim: int = 16, img_proj_dim: int = 256,
                 head_dropout: float = 0.2, num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(1, gru_hidden_size, gru_num_layers, batch_first=True,
                           dropout=gru_dropout if gru_num_layers > 1 else 0.0)
        self.time_proj = nn.Sequential(nn.Linear(1, time_proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(img_feat_dim, img_proj_dim), nn.Dropout(0.5), nn.ReLU())

        fusion_dim = gru_hidden_size + time_proj_dim + img_proj_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(fusion_dim // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor,
                img_feat: torch.Tensor, has_meal: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(cgm.unsqueeze(-1))
        cgm_feat = h_n[-1]                                          # [B, gru_hidden_size]
        time_feat = self.time_proj(time_since_meal.unsqueeze(-1))   # [B, time_proj_dim]
        img_feat_proj = self.img_proj(img_feat)                     # [B, img_proj_dim]
        gate = has_meal.unsqueeze(-1)                                # [B, 1]
        time_feat = time_feat * gate                                 # no meal nearby -> zero
        img_feat_proj = img_feat_proj * gate
        fused = torch.cat([cgm_feat, time_feat, img_feat_proj], dim=-1)
        return self.head(fused)