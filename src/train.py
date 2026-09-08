import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    preds, trues = [], []
    for cgm, target in loader:
        preds.append(model(cgm.to(device)).cpu().numpy())
        trues.append(target.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_model(model: nn.Module, train_loader, val_loader, cgm_mean: float, cgm_std: float,
                 device=None, lr: float = 1e-3, weight_decay: float = 1e-4,
                 max_grad_norm: float = 1.0, max_epochs: int = 50, early_stop_patience: int = 10,
                 lr_patience: int = 5, lr_factor: float = 0.5, checkpoint_dir=None,
                 verbose: bool = True, progress_bar: bool = False, progress_desc: str = 'training'):

    from src import metrics as metrics_module

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if verbose or progress_bar:
        device_label = device
        if device == 'cuda':
            device_label += f" ({torch.cuda.get_device_name(torch.cuda.current_device())})"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"device: {device_label}  model: {model.__class__.__name__}  "
              f"params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state, final_state = None, None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}

    epoch_iter = range(max_epochs)
    if progress_bar:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(epoch_iter, desc=progress_desc)

    for epoch in epoch_iter:
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for cgm, target in train_loader:
            cgm, target = cgm.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(cgm), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            # weight each batch's contribution by its actual sample count, not by batch index --
            # the last batch of an epoch is usually smaller than batch_size (e.g. 76 of 256 here)
            # and would otherwise be counted as equally important as a full batch.
            batch_n = cgm.size(0)
            train_loss_sum += loss.item() * batch_n
            train_n += batch_n

        y_pred_norm, y_true_norm = predict(model, val_loader, device=device)
        val_loss = float(np.mean((y_pred_norm - y_true_norm) ** 2))

        y_pred_mgdl = y_pred_norm * cgm_std + cgm_mean
        y_true_mgdl = y_true_norm * cgm_std + cgm_mean
        val_rmse = [metrics_module.rmse(y_pred_mgdl[:, i], y_true_mgdl[:, i])
                    for i in range(y_pred_mgdl.shape[1])]
        val_mae = [metrics_module.mae(y_pred_mgdl[:, i], y_true_mgdl[:, i])
                   for i in range(y_pred_mgdl.shape[1])]

        history['train_loss'].append(train_loss_sum / train_n)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        scheduler.step(val_loss)

        if verbose:
            print(f"epoch {epoch + 1:>3}/{max_epochs}  train_loss={history['train_loss'][-1]:.4f}"
                  f"  val_loss={val_loss:.4f}  val_rmse(mg/dL)={[round(r, 1) for r in val_rmse]}")
        if progress_bar:
            epoch_iter.set_postfix(train_loss=f"{history['train_loss'][-1]:.4f}",
                                    val_loss=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}")

        final_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = final_state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                if verbose:
                    print(f"early stopping at epoch {epoch + 1} "
                          f"(no improvement for {early_stop_patience} epochs)")
                if progress_bar:
                    epoch_iter.set_description(f"{progress_desc} (stopped @ {epoch + 1})")
                    epoch_iter.close()
                break

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / 'best.pt')
        torch.save(final_state, checkpoint_dir / 'final.pt')
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    return model, history
