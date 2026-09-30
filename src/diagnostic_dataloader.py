import numpy as np

from src.dataloader import (
    DEFAULT_PROCESSED_DIR, DEFAULT_IMAGE_DIR, TIME_SINCE_MEAL_LOOKBACK_MIN,
    load_cgm_stats, load_multimodal_samples, load_img_features, _to_arrays, _image_key,
)


def _make_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation with no fixed points when n > 1."""
    if n <= 1:
        return np.arange(n)

    order = rng.permutation(n)
    shift = int(rng.integers(1, n))
    reassigned = np.empty(n, dtype=int)
    reassigned[order] = np.roll(order, shift)
    return reassigned


def load_multimodal_tensors_shuffled_image(
    split: str,
    encoder_name: str,
    batch_size: int = 256,
    shuffle: bool | None = None,
    seed: int | None = None,
    processed_dir=DEFAULT_PROCESSED_DIR,
    image_dir=DEFAULT_IMAGE_DIR,
) -> 'torch.utils.data.DataLoader':
    """Standard multimodal loader with fixed unique-image-level feature reassignment."""
    import torch
    from torch.utils.data import Dataset, DataLoader

    samples = load_multimodal_samples(split, processed_dir)
    stats = load_cgm_stats(processed_dir)
    X, y = _to_arrays(samples, stats['cgm_mean'], stats['cgm_std'])

    has_meal = np.array(
        [1.0 if s['has_meal'] else 0.0 for s in samples], dtype=np.float32
    )
    t_raw = np.array(
        [s['time_since_meal'] if s['has_meal'] else 0.0 for s in samples],
        dtype=np.float32,
    )
    t = t_raw / TIME_SINCE_MEAL_LOOKBACK_MIN

    feature_dict = load_img_features(encoder_name, image_dir)
    feat_dim = len(next(iter(feature_dict.values())))
    zero_feat = np.zeros(feat_dim, dtype=np.float32)

    # First-occurrence order keeps this deterministic before applying the seeded reassignment.
    unique_keys = list(dict.fromkeys(
        _image_key(s) for s in samples if s['has_image']
    ))

    missing_keys = [k for k in unique_keys if k not in feature_dict]
    if missing_keys:
        raise KeyError(
            f"{len(missing_keys)} image keys are missing from {encoder_name} features; "
            f"first missing key: {missing_keys[0]}"
        )

    rng = np.random.default_rng(seed)
    permutation = _make_derangement(len(unique_keys), rng)

    reassigned_feature = {
        src_key: np.asarray(feature_dict[unique_keys[dst_idx]], dtype=np.float32)
        for src_key, dst_idx in zip(unique_keys, permutation)
    }

    img_feats = []
    for s in samples:
        if s['has_image']:
            img_feats.append(reassigned_feature[_image_key(s)])
        else:
            img_feats.append(zero_feat)
    img_feats = np.stack(img_feats).astype(np.float32)

    class ShuffledImageDataset(Dataset):
        def __len__(self):
            return len(X)

        def __getitem__(self, idx):
            return (
                torch.from_numpy(X[idx]),
                torch.tensor(t[idx]),
                torch.tensor(has_meal[idx]),
                torch.from_numpy(img_feats[idx]),
                torch.from_numpy(y[idx]),
            )

    if shuffle is None:
        shuffle = (split == 'train')

    return DataLoader(
        ShuffledImageDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
    )
