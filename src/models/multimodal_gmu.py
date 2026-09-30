

import torch
import torch.nn as nn

from src.models.classical_baseline import HORIZONS


class CGMTimeGMUPredictor(nn.Module):

    def __init__(self, gru_hidden_size: int = 256, gru_num_layers: int = 3,
                 gru_dropout: float = 0.2, fusion_dim: int = 256, head_dropout: float = 0.2,
                 num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(1, gru_hidden_size, gru_num_layers, batch_first=True,
                           dropout=gru_dropout if gru_num_layers > 1 else 0.0)
        # Per-modality transforms into the shared fusion_dim -- both required for a weighted sum
        # to be well-defined, unlike concatenation which tolerates mismatched per-modality sizes.
        self.cgm_transform = nn.Sequential(nn.Linear(gru_hidden_size, fusion_dim), nn.Tanh())
        self.time_transform = nn.Sequential(nn.Linear(1, fusion_dim), nn.Tanh())
        # Gate network: learns per-sample mixing weights from the *transformed* h_cgm/h_time
        # (see module docstring) -- not the raw pre-transform features.
        self.gate_net = nn.Linear(fusion_dim * 2, 2)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(fusion_dim // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor,
                has_meal: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(cgm.unsqueeze(-1))
        cgm_feat = h_n[-1]                                            # [B, gru_hidden_size]
        h_cgm = self.cgm_transform(cgm_feat)                          # [B, fusion_dim]
        h_time = self.time_transform(time_since_meal.unsqueeze(-1))   # [B, fusion_dim]

        gate_input = torch.cat([h_cgm, h_time], dim=-1)
        learned_z = torch.softmax(self.gate_net(gate_input), dim=-1)  # [B, 2], learned mix

        forced_z = torch.zeros_like(learned_z)
        forced_z[:, 0] = 1.0                                          # no meal -> 100% CGM

        m = has_meal.unsqueeze(-1)                                    # [B, 1], 1.0 or 0.0
        z = m * learned_z + (1 - m) * forced_z                        # [B, 2]

        fused = z[:, 0:1] * h_cgm + z[:, 1:2] * h_time                # [B, fusion_dim]
        return self.head(fused)


class MultimodalGMUPredictor(nn.Module):

    def __init__(self, img_feat_dim: int, gru_hidden_size: int = 256, gru_num_layers: int = 3,
                 gru_dropout: float = 0.2, fusion_dim: int = 256, img_bottleneck_dim: int = 16,
                 head_dropout: float = 0.2, num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(1, gru_hidden_size, gru_num_layers, batch_first=True,
                           dropout=gru_dropout if gru_num_layers > 1 else 0.0)
        self.cgm_transform = nn.Sequential(nn.Linear(gru_hidden_size, fusion_dim), nn.Tanh())
        self.time_transform = nn.Sequential(nn.Linear(1, fusion_dim), nn.Tanh())

        self.img_transform = nn.Sequential(
            nn.Linear(img_feat_dim, img_bottleneck_dim), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(img_bottleneck_dim, fusion_dim), nn.Tanh(),
        )
        self.gate_net = nn.Linear(fusion_dim * 3, 3)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(fusion_dim // 2, num_horizons),
        )

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor,
                img_feat: torch.Tensor, has_meal: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(cgm.unsqueeze(-1))
        cgm_feat = h_n[-1]                                            # [B, gru_hidden_size]
        h_cgm = self.cgm_transform(cgm_feat)                          # [B, fusion_dim]
        h_time = self.time_transform(time_since_meal.unsqueeze(-1))   # [B, fusion_dim]
        h_img = self.img_transform(img_feat)                          # [B, fusion_dim]

        gate_input = torch.cat([h_cgm, h_time, h_img], dim=-1)
        learned_z = torch.softmax(self.gate_net(gate_input), dim=-1)  # [B, 3], learned mix

        forced_z = torch.zeros_like(learned_z)
        forced_z[:, 0] = 1.0                                          # no meal -> 100% CGM

        m = has_meal.unsqueeze(-1)                                    # [B, 1], 1.0 or 0.0
        z = m * learned_z + (1 - m) * forced_z                        # [B, 3]

        fused = z[:, 0:1] * h_cgm + z[:, 1:2] * h_time + z[:, 2:3] * h_img   # [B, fusion_dim]
        return self.head(fused)
