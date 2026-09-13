from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.models.classical_baseline import HORIZONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / 'processed_data' / 'processed_5min'

DEFAULT_IMAGE_DIR = PROJECT_ROOT / 'processed_data' / 'img_preprocessing'

def load_cgm_stats(processed_dir=DEFAULT_PROCESSED_DIR) -> dict:
    with open(Path(processed_dir) / 'cgm_stats.json') as f:
        return json.load(f)


def load_samples(split: str, processed_dir=DEFAULT_PROCESSED_DIR) -> list:
    with open(Path(processed_dir) / f'{split}.pkl', 'rb') as f:
        return pickle.load(f)


def _to_arrays(samples: list, cgm_mean: float, cgm_std: float):
    X = np.stack([s['cgm_sequence'] for s in samples]).astype(np.float32)
    y = np.stack([[s[f'target_{h}'] for h in HORIZONS] for s in samples]).astype(np.float32)
    # Use training-set CGM statistics for both inputs and prediction targets.
    X = (X - cgm_mean) / cgm_std
    y = (y - cgm_mean) / cgm_std
    return X, y

def load_tensors(split: str, batch_size: int = 256, shuffle: bool | None = None,
                  processed_dir=DEFAULT_PROCESSED_DIR) -> 'torch.utils.data.DataLoader':
    import torch
    from torch.utils.data import Dataset, DataLoader

    class CGMDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    stats = load_cgm_stats(processed_dir)
    X, y = _to_arrays(load_samples(split, processed_dir), stats['cgm_mean'], stats['cgm_std'])
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(CGMDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def load_arrays(split: str, processed_dir=DEFAULT_PROCESSED_DIR):
    stats = load_cgm_stats(processed_dir)
    return _to_arrays(load_samples(split, processed_dir), stats['cgm_mean'], stats['cgm_std'])


def load_participant_ids(split: str, processed_dir=DEFAULT_PROCESSED_DIR) -> np.ndarray:
    return np.array([s['participant_id'] for s in load_samples(split, processed_dir)])


# --- --- --- Multimodal Data loader --- --- ---
# Keep the same eligible population across all multimodal ablations.
TIME_SINCE_MEAL_LOOKBACK_MIN = 120.0

def _exclude_missing_image(samples: list) -> list:
    # Exclude meal windows with missing photographs before constructing any modality condition.
    return [s for s in samples if not (s['has_meal'] and not s['has_image'])]

def _image_key(sample: dict) -> str:
    return f"{sample['participant_id']}/{Path(sample['image_path']).name}"

def load_img_features(encoder_name: str, image_dir=DEFAULT_IMAGE_DIR) -> dict:
    path = Path(image_dir) / 'img_features' / f'img_features_{encoder_name}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_multimodal_samples(split: str, processed_dir=DEFAULT_PROCESSED_DIR) -> list:
    return _exclude_missing_image(load_samples(split, processed_dir))

def load_multimodal_participant_ids(split: str, processed_dir=DEFAULT_PROCESSED_DIR) -> np.ndarray:
    return np.array([s['participant_id']
                      for s in load_multimodal_samples(split, processed_dir)])


def load_multimodal_tensors(split: str, encoder_name: str | None = None, batch_size: int = 256,
                            shuffle: bool | None = None,
                            processed_dir=DEFAULT_PROCESSED_DIR,
                            image_dir=DEFAULT_IMAGE_DIR) -> 'torch.utils.data.DataLoader':
    import torch
    from torch.utils.data import Dataset, DataLoader

    samples = load_multimodal_samples(split, processed_dir)
    stats = load_cgm_stats(processed_dir)
    X, y = _to_arrays(samples, stats['cgm_mean'], stats['cgm_std'])

    has_meal = np.array([1.0 if s['has_meal'] else 0.0 for s in samples], dtype=np.float32)

    # Scale meal timing to the fixed 120-minute lookback range.
    t_raw = np.array([s['time_since_meal'] if s['has_meal'] else 0.0 for s in samples],
                     dtype=np.float32)
    t = t_raw / TIME_SINCE_MEAL_LOOKBACK_MIN

    img_feats = None
    if encoder_name is not None:
        feature_dict = load_img_features(encoder_name, image_dir)
        feat_dim = len(next(iter(feature_dict.values())))

        # Non-meal windows have no image modality, so represent them with a zero feature vector.
        zero_feat = np.zeros(feat_dim, dtype=np.float32)
        img_feats = np.stack([
            feature_dict[_image_key(s)] if s['has_image'] else zero_feat for s in samples
        ]).astype(np.float32)

    class MultimodalDataset(Dataset):
        def __len__(self) -> int:
            return len(X)

        def __getitem__(self, idx):
            item = (torch.from_numpy(X[idx]), torch.tensor(t[idx]), torch.tensor(has_meal[idx]))
            if img_feats is not None:
                item = item + (torch.from_numpy(img_feats[idx]),)
            return item + (torch.from_numpy(y[idx]),)

    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(MultimodalDataset(), batch_size=batch_size, shuffle=shuffle)