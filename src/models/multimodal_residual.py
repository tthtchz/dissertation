from pathlib import Path

import torch
import torch.nn as nn

from src.models.classical_baseline import HORIZONS
from src.models.multimodal import CGMTimePredictor


def load_frozen_baseline(seed: int, exp2_checkpoint_dir: Path, device: str = 'cpu',
                          gru_hidden_size: int = 256, gru_num_layers: int = 3,
                          gru_dropout: float = 0.2, time_proj_dim: int = 16,
                          head_dropout: float = 0.2) -> CGMTimePredictor:

    checkpoint_path = Path(exp2_checkpoint_dir) / f'gated_cgm_time_seed{seed}' / 'best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Exp2 checkpoint not found: {checkpoint_path}\n"
            f"The residual models in this module require experiment 2 (CGM+time_since_meal) to "
            f"have already been trained for this seed -- run "
            f"experiments/3_multimodal/multimodal/run_exp2_cgm_time.ipynb first.")

    model = CGMTimePredictor(gru_hidden_size=gru_hidden_size, gru_num_layers=gru_num_layers,
                              gru_dropout=gru_dropout, time_proj_dim=time_proj_dim,
                              head_dropout=head_dropout, num_horizons=len(HORIZONS))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class FrozenImageOnlyResidualPredictor(nn.Module):
    def __init__(self, base_model: CGMTimePredictor, img_feat_dim: int, img_proj_dim: int = 16,
                 residual_hidden_dim: int = 16, img_dropout: float = 0.3,
                 image_scale: float = 1.0, num_horizons: int = len(HORIZONS)):
        super().__init__()
        self.image_scale = image_scale

        self.gru = base_model.gru
        self.time_proj = base_model.time_proj
        self.base_head = base_model.head
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.time_proj.parameters():
            p.requires_grad = False
        for p in self.base_head.parameters():
            p.requires_grad = False

        # Trainable image-only residual branch.
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, img_proj_dim), nn.ReLU(), nn.Dropout(img_dropout),
        )
        self.image_head = nn.Sequential(
            nn.Linear(img_proj_dim, residual_hidden_dim), nn.ReLU(), nn.Dropout(img_dropout),
            nn.Linear(residual_hidden_dim, num_horizons),
        )
        # y_final == base_pred at initialization (see module docstring).
        nn.init.zeros_(self.image_head[-1].weight)
        nn.init.zeros_(self.image_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.gru.eval()
        self.time_proj.eval()
        self.base_head.eval()
        return self

    def forward(self, cgm: torch.Tensor, time_since_meal: torch.Tensor, img_feat: torch.Tensor,
                has_meal: torch.Tensor) -> torch.Tensor:
        gate = has_meal.unsqueeze(-1)

        with torch.no_grad():
            _, h_n = self.gru(cgm.unsqueeze(-1))
            cgm_feat = h_n[-1]
            time_feat = self.time_proj(time_since_meal.unsqueeze(-1))
            time_feat = time_feat * gate
            base_feat = torch.cat([cgm_feat, time_feat], dim=-1)
            base_pred = self.base_head(base_feat)

        # IMAGE-ONLY residual
        img_feat_proj = self.img_proj(img_feat)
        img_feat_proj = img_feat_proj * gate                 # no meal -> no image contribution
        image_residual = self.image_head(img_feat_proj)
        image_residual = image_residual * gate

        return base_pred + self.image_scale * image_residual
