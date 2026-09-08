from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.models.classical_baseline import HORIZONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / 'preprocessed_data' / 'processed_5min'


def load_cgm_stats(processed_dir=DEFAULT_PROCESSED_DIR) -> dict:
    with open(Path(processed_dir) / 'cgm_stats.json') as f:
        return json.load(f)


def load_samples(split: str, processed_dir=DEFAULT_PROCESSED_DIR) -> list:
    with open(Path(processed_dir) / f'{split}.pkl', 'rb') as f:
        return pickle.load(f)


def _to_arrays(samples: list, cgm_mean: float, cgm_std: float):
    X = np.stack([s['cgm_sequence'] for s in samples]).astype(np.float32)
    y = np.stack([[s[f'target_{h}'] for h in HORIZONS] for s in samples]).astype(np.float32)
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
