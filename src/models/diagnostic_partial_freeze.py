import torch
import torch.nn as nn

from src.models.classical_baseline import HORIZONS


class PartiallyFrozenSharedHeadPredictor(nn.Module):


    def __init__(self, base_model, img_feat_dim: int, img_proj_dim: int = 256,
                 head_dropout: float = 0.2, num_horizons: int = len(HORIZONS)):
        super().__init__()

        self.gru = base_model.gru
        self.time_proj = base_model.time_proj
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.time_proj.parameters():
            p.requires_grad = False

        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, img_proj_dim), nn.Dropout(0.5), nn.ReLU(),
        )

        gru_hidden_size = self.gru.hidden_size
        time_proj_dim = self.time_proj[0].out_features
        fusion_dim = gru_hidden_size + time_proj_dim + img_proj_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(fusion_dim // 2, num_horizons),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.gru.eval()
        self.time_proj.eval()
        return self

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor,
                img_feat: torch.Tensor, has_meal: torch.Tensor) -> torch.Tensor:
        gate = has_meal.unsqueeze(-1)

        with torch.no_grad():
            _, h_n = self.gru(cgm.unsqueeze(-1))
            cgm_feat = h_n[-1]
            time_feat = self.time_proj(time_since_meal.unsqueeze(-1))
            time_feat = time_feat * gate

        img_feat_proj = self.img_proj(img_feat)
        img_feat_proj = img_feat_proj * gate

        fused = torch.cat([cgm_feat, time_feat, img_feat_proj], dim=-1)
        return self.head(fused)
