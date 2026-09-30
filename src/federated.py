from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src import dataloader as data
from src import metrics as metrics_module
from src.train import _default_forward_fn, predict, set_seed  # reused, not reimplemented


def load_federated_clients(split: str, encoder_name: str | None = None, batch_size: int = 256,
                            processed_dir=data.DEFAULT_PROCESSED_DIR,
                            image_dir=data.DEFAULT_IMAGE_DIR) -> dict[str, 'torch.utils.data.DataLoader']:

    from torch.utils.data import Dataset, DataLoader

    samples = data.load_multimodal_samples(split, processed_dir)
    stats = data.load_cgm_stats(processed_dir)
    cgm_mean, cgm_std = stats['cgm_mean'], stats['cgm_std']

    img_feats_by_key = None
    feat_dim = None
    if encoder_name is not None:
        img_feats_by_key = data.load_img_features(encoder_name, image_dir)
        feat_dim = len(next(iter(img_feats_by_key.values())))

    class MultimodalDataset(Dataset):
        def __init__(self, participant_samples: list):
            X = np.stack([s['cgm_sequence'] for s in participant_samples]).astype(np.float32)
            y = np.stack([[s[f'target_{h}'] for h in metrics_module.HORIZONS]
                          for s in participant_samples]).astype(np.float32)
            self.X = (X - cgm_mean) / cgm_std
            self.y = (y - cgm_mean) / cgm_std
            self.has_meal = np.array([1.0 if s['has_meal'] else 0.0
                                       for s in participant_samples], dtype=np.float32)
            t_raw = np.array([s['time_since_meal'] if s['has_meal'] else 0.0
                               for s in participant_samples], dtype=np.float32)
            self.t = t_raw / data.TIME_SINCE_MEAL_LOOKBACK_MIN
            self.img_feats = None
            if img_feats_by_key is not None:
                zero_feat = np.zeros(feat_dim, dtype=np.float32)
                self.img_feats = np.stack([
                    img_feats_by_key[data._image_key(s)] if s['has_image'] else zero_feat
                    for s in participant_samples
                ]).astype(np.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            item = (torch.from_numpy(self.X[idx]), torch.tensor(self.t[idx]),
                     torch.tensor(self.has_meal[idx]))
            if self.img_feats is not None:
                item = item + (torch.from_numpy(self.img_feats[idx]),)
            return item + (torch.from_numpy(self.y[idx]),)

    by_participant: dict[str, list] = {}
    for s in samples:
        by_participant.setdefault(s['participant_id'], []).append(s)

    return {
        pid: DataLoader(MultimodalDataset(pid_samples), batch_size=batch_size, shuffle=True)
        for pid, pid_samples in sorted(by_participant.items())
    }


def _state_dict_to_cpu(model: nn.Module) -> dict:
    return {k: v.clone().cpu() for k, v in model.state_dict().items()}


def _weighted_average_state_dicts(state_dicts: list[dict], weights: list[int]) -> dict:
    total = float(sum(weights))
    keys = state_dicts[0].keys()
    return {
        k: sum(w * sd[k].float() for w, sd in zip(weights, state_dicts)) / total
        for k in keys
    }


def federated_train(model_fn, client_loaders: dict, val_loader, cgm_mean: float, cgm_std: float,
                     device=None, lr: float = 1e-4, weight_decay: float = 1e-5,
                     max_grad_norm: float = 1.0, local_epochs: int = 1, max_rounds: int = 50,
                     clients_per_round: int | None = None,
                     early_stop_patience: int = 10, lr_patience: int = 5, lr_factor: float = 0.5,
                     checkpoint_dir=None, verbose: bool = True, progress_bar: bool = False,
                     progress_desc: str = 'federated training', forward_fn=None):

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    forward_fn = forward_fn or _default_forward_fn

    global_model = model_fn().to(device)
    all_client_ids = list(client_loaders.keys())
    k = min(clients_per_round, len(all_client_ids)) if clients_per_round is not None else None
    if verbose or progress_bar:
        n_params = sum(p.numel() for p in global_model.parameters())
        n_trainable = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        participation = f"{k}/{len(all_client_ids)} sampled" if k is not None else "full"
        print(f"device: {device}  model: {global_model.__class__.__name__}  "
              f"params: {n_params:,}  trainable: {n_trainable:,}  "
              f"clients: {len(all_client_ids)} ({participation} per round)")
    global_state = _state_dict_to_cpu(global_model)

    criterion = nn.MSELoss()
    current_lr = lr
    best_val_loss = float('inf')
    best_state, final_state = None, None
    rounds_without_improvement = 0
    lr_plateau_counter = 0
    history = {'val_loss': [], 'val_rmse': [], 'val_mae': [], 'lr': []}

    round_iter = range(max_rounds)
    if progress_bar:
        from tqdm.auto import tqdm
        round_iter = tqdm(round_iter, desc=progress_desc)

    for round_idx in round_iter:
        client_states, client_weights = [], []

        round_client_ids = random.sample(all_client_ids, k) if k is not None else all_client_ids
        for pid in round_client_ids:
            loader = client_loaders[pid]
            local_model = model_fn().to(device)
            local_model.load_state_dict(global_state)
            local_model.train()
            optimizer = torch.optim.Adam(
                (p for p in local_model.parameters() if p.requires_grad),
                lr=current_lr, weight_decay=weight_decay)

            for _ in range(local_epochs):
                for batch in loader:
                    y_pred, target = forward_fn(local_model, batch, device)
                    optimizer.zero_grad()
                    loss = criterion(y_pred, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
                    optimizer.step()


            n_samples = len(loader.dataset)
            client_states.append(_state_dict_to_cpu(local_model))
            client_weights.append(n_samples)

        global_state = _weighted_average_state_dicts(client_states, client_weights)
        global_model.load_state_dict(global_state)

        y_pred_norm, y_true_norm = predict(global_model, val_loader, device=device,
                                            forward_fn=forward_fn)
        val_loss = float(np.mean((y_pred_norm - y_true_norm) ** 2))
        y_pred_mgdl = y_pred_norm * cgm_std + cgm_mean
        y_true_mgdl = y_true_norm * cgm_std + cgm_mean
        val_rmse = [metrics_module.rmse(y_pred_mgdl[:, i], y_true_mgdl[:, i])
                    for i in range(y_pred_mgdl.shape[1])]
        val_mae = [metrics_module.mae(y_pred_mgdl[:, i], y_true_mgdl[:, i])
                   for i in range(y_pred_mgdl.shape[1])]

        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        if verbose:
            print(f"round {round_idx + 1:>3}/{max_rounds}  val_loss={val_loss:.4f}  "
                  f"val_rmse(mg/dL)={[round(r, 1) for r in val_rmse]}  lr={current_lr:.2e}")
        if progress_bar:
            round_iter.set_postfix(val_loss=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}",
                                    lr=f"{current_lr:.1e}")

        final_state = {k: v.clone() for k, v in global_state.items()}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = final_state
            rounds_without_improvement = 0
            lr_plateau_counter = 0
        else:
            rounds_without_improvement += 1
            lr_plateau_counter += 1
            if lr_plateau_counter >= lr_patience:
                current_lr *= lr_factor
                lr_plateau_counter = 0
            if rounds_without_improvement >= early_stop_patience:
                if verbose:
                    print(f"early stopping at round {round_idx + 1} "
                          f"(no improvement for {early_stop_patience} rounds)")
                if progress_bar:
                    round_iter.set_description(f"{progress_desc} (stopped @ {round_idx + 1})")
                    round_iter.close()
                break

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / 'best.pt')
        torch.save(final_state, checkpoint_dir / 'final.pt')
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    global_model.load_state_dict(best_state)
    return global_model, history


def load_federated_frozen_baseline(seed: int, fed_cgm_time_checkpoint_dir, device: str = 'cpu',
                                    gru_hidden_size: int = 256, gru_num_layers: int = 3,
                                    gru_dropout: float = 0.2, time_proj_dim: int = 16,
                                    head_dropout: float = 0.2):

    from src.models.classical_baseline import HORIZONS
    from src.models.multimodal import CGMTimePredictor

    checkpoint_path = Path(fed_cgm_time_checkpoint_dir) / f'seed{seed}' / 'best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Federated CGM+Time checkpoint not found: {checkpoint_path}\n"
            f"run_fed3_cgm_time_dinov2.ipynb requires run_fed2_cgm_time.ipynb to have already "
            f"been trained for this seed.")

    model = CGMTimePredictor(gru_hidden_size=gru_hidden_size, gru_num_layers=gru_num_layers,
                              gru_dropout=gru_dropout, time_proj_dim=time_proj_dim,
                              head_dropout=head_dropout, num_horizons=len(HORIZONS))
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
