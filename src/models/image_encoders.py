import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ENCODER_NAMES = ['resnet18', 'vit', 'clip', 'dinov2']


class SquarePad:
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
