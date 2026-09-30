import numpy as np
from src.dataloader import (
    DEFAULT_PROCESSED_DIR, DEFAULT_IMAGE_DIR, TIME_SINCE_MEAL_LOOKBACK_MIN,
    load_cgm_stats, load_multimodal_samples, load_img_features, _to_arrays, _image_key,
)

def load_multimodal_tensors_random_image(
    split, encoder_name, batch_size=256, shuffle=None, seed=None,
    processed_dir=DEFAULT_PROCESSED_DIR, image_dir=DEFAULT_IMAGE_DIR,
):
    import torch
    from torch.utils.data import Dataset, DataLoader

    samples = load_multimodal_samples(split, processed_dir)
    stats = load_cgm_stats(processed_dir)
    X, y = _to_arrays(samples, stats["cgm_mean"], stats["cgm_std"])

    has_meal = np.asarray([float(s["has_meal"]) for s in samples], dtype=np.float32)
    t_raw = np.asarray(
        [s["time_since_meal"] if s["has_meal"] else 0.0 for s in samples],
        dtype=np.float32,
    )
    t = t_raw / TIME_SINCE_MEAL_LOOKBACK_MIN

    feature_dict = load_img_features(encoder_name, image_dir)
    feat_dim = len(next(iter(feature_dict.values())))
    zero = np.zeros(feat_dim, dtype=np.float32)

    unique_keys = list(dict.fromkeys(_image_key(s) for s in samples if s["has_image"]))
    missing = [k for k in unique_keys if k not in feature_dict]
    if missing:
        raise KeyError(f"Missing {len(missing)} image features; first: {missing[0]}")

    real_unique = np.stack(
        [np.asarray(feature_dict[k], dtype=np.float32) for k in unique_keys]
    )
    mu = real_unique.mean(axis=0)
    sigma = real_unique.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)

    rng = np.random.default_rng(seed)
    random_unique = rng.normal(
        loc=mu, scale=sigma, size=(len(unique_keys), feat_dim)
    ).astype(np.float32)
    random_map = {k: random_unique[i] for i, k in enumerate(unique_keys)}

    img = np.stack([
        random_map[_image_key(s)] if s["has_image"] else zero
        for s in samples
    ]).astype(np.float32)

    class DS(Dataset):
        def __len__(self): return len(X)
        def __getitem__(self, i):
            return (
                torch.from_numpy(X[i]), torch.tensor(t[i]), torch.tensor(has_meal[i]),
                torch.from_numpy(img[i]), torch.from_numpy(y[i]),
            )

    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(DS(), batch_size=batch_size, shuffle=shuffle)
