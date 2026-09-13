"""
Frozen image encoders for the multimodal (CGM + food photograph) fusion experiments
(experiments/3_*). Four candidates, chosen to cover four different pretraining paradigms rather
than four architectures for their own sake -- a "backbone screen" for the image side, in the same
spirit as the CGM backbone screen (Experimental Design: backbone screen):

  resnet18  -- CNN, supervised (ImageNet-1k classification labels)
  vit       -- ViT-B/16, supervised (ImageNet-1k classification labels)
  clip      -- CLIP ViT-B/16 vision tower, contrastive image-text pretraining
  dinov2    -- DINOv2 ViT-B/14, self-supervised (no labels, no text)

All four are used **frozen** (`eval()`, `requires_grad_(False)`) and never fine-tuned on this
dataset, consistent with the "lightweight fusion given limited data" requirement from the
multimodal architecture design (Methodology: Model Architecture) -- with roughly 1,500 unique
tagged photographs total, fine-tuning any of these from this dataset alone would overfit badly.

Deliberately not reusing the pilot phase's conclusion that CLIP was the best encoder: the sample
set, resolution, and lookback window have all changed since then (dataset/meal_analysis.ipynb,
cgm_preprocessing.ipynb), so that conclusion is re-established from scratch, not assumed.

Preprocessing recipe (`SquarePad` + `IMG_TRANSFORM`) is identical across all four candidates and
reused from the pilot phase's `preprocess/dataset.py` / `preprocess/precompute_features.py` for
continuity: pad the short side to a square, then resize to 224x224, so the full photo is preserved
with no cropping, followed by ImageNet mean/std normalization (DINOv2's own training recipe uses
the same statistics, so this is not a CLIP/ImageNet-only convention forced onto it).
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ENCODER_NAMES = ['resnet18', 'vit', 'clip', 'dinov2']


class SquarePad:
    """Pad the short side to make the image square, then resize -- preserves the entire photo,
    unlike a center-crop, at the cost of some black padding on one axis."""

    def __init__(self, size: int = IMG_SIZE, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        diff = abs(w - h)
        if w < h:
            pad = (diff // 2, 0, diff - diff // 2, 0)
        else:
            pad = (0, diff // 2, 0, diff - diff // 2)
        img = TF.pad(img, pad, fill=self.fill)
        return TF.resize(img, [self.size, self.size])


IMG_TRANSFORM = transforms.Compose([
    SquarePad(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_encoder(name: str, device: str = 'cpu'):
    """Returns (model, feat_dim, forward_fn). forward_fn(model, batch_tensor) -> [B, feat_dim].

    batch_tensor is the output of IMG_TRANSFORM stacked over a batch -- [B, 3, 224, 224].
    """
    if name == 'resnet18':
        import torchvision.models as tv_models
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        feat_dim = 512
        forward_fn = lambda m, x: m(x)

    elif name == 'vit':
        import torchvision.models as tv_models
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        feat_dim = 768
        forward_fn = lambda m, x: m(x)

    elif name == 'clip':
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
        feat_dim = 768
        forward_fn = lambda m, x: m(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token

    elif name == 'dinov2':
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        feat_dim = 768
        forward_fn = lambda m, x: m(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token

    else:
        raise ValueError(f'unknown encoder: {name!r}; choose from {ENCODER_NAMES}')

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, feat_dim, forward_fn
