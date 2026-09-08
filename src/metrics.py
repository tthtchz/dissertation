import numpy as np

from src.models.classical_baseline import HORIZONS


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# ─────────────────────────── RMSE / MAE ───────────────────────────

def rmse(y_pred, y_true) -> float:
    y_pred, y_true = _to_numpy(y_pred), _to_numpy(y_true)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred, y_true) -> float:
    y_pred, y_true = _to_numpy(y_pred), _to_numpy(y_true)
    return float(np.mean(np.abs(y_pred - y_true)))


# ─────────────────────────── gRMSE (Del Favero et al. 2012; sqrt per Zhu et al. 2022) ───────────

def _sigma_ge(x, a, eps):

    x, a = np.asarray(x, dtype=np.float64), np.asarray(a, dtype=np.float64)
    xi = (2.0 / eps) * (x - a - eps / 2.0)
    mid1 = -0.5 * xi**4 - xi**3 + xi + 0.5   # a <= x <= a + eps/2
    mid2 = 0.5 * xi**4 - xi**3 + xi + 0.5    # a + eps/2 <= x <= a + eps
    return np.select([x <= a, x <= a + eps / 2, x <= a + eps, x > a + eps],
                      [0.0, mid1, mid2, 1.0])


def _sigma_le(x, a, eps):

    x, a = np.asarray(x, dtype=np.float64), np.asarray(a, dtype=np.float64)
    xi_bar = -(2.0 / eps) * (x - a + eps / 2.0)
    mid1 = 0.5 * xi_bar**4 - xi_bar**3 + xi_bar + 0.5    # a-eps <= x <= a-eps/2
    mid2 = -0.5 * xi_bar**4 - xi_bar**3 + xi_bar + 0.5   # a-eps/2 <= x <= a
    return np.select([x <= a - eps, x <= a - eps / 2, x <= a, x > a],
                      [1.0, mid1, mid2, 0.0])


# Table I, Del Favero et al. 2012. TL/TH are not the clinical hypo/hyperglycemia thresholds
# themselves (70/180 mg/dL) -- they are set further out so the smooth penalty ramp is already
# under way before the clinical threshold and fully saturated (2.5x / 2x) only well past it
# (g<55 severe hypo, g>255 severe hyper).
GRMSE_PARAMS = dict(alpha_L=1.5, beta_L=30.0, gamma_L=10.0, T_L=85.0,
                     alpha_H=1.0, beta_H=100.0, gamma_H=20.0, T_H=155.0)


def pen(g, ghat, **params) -> np.ndarray:

    p = {**GRMSE_PARAMS, **params}
    g, ghat = _to_numpy(g), _to_numpy(ghat)
    hypo = p['alpha_L'] * _sigma_le(g, p['T_L'], p['beta_L']) * _sigma_ge(ghat, g, p['gamma_L'])
    hyper = p['alpha_H'] * _sigma_ge(g, p['T_H'], p['beta_H']) * _sigma_le(ghat, g, p['gamma_H'])
    return 1.0 + hypo + hyper


def grmse(y_pred, y_true, **params) -> float:

    y_pred, y_true = _to_numpy(y_pred), _to_numpy(y_true)
    weight = pen(y_true, y_pred, **params)
    return float(np.sqrt(np.mean(weight * (y_pred - y_true) ** 2)))


# ─────────────────────────── Clarke Error Grid (Clarke et al. 1987) ───────────────────────────

def clarke_error_grid_zones(y_true, y_pred) -> dict:
    ref, pred = _to_numpy(y_true), _to_numpy(y_pred)

    is_A = ((ref <= 70) & (pred <= 70)) | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    is_E = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    is_C = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
            | ((ref >= 130) & (ref <= 180) & (pred <= (7 / 5) * ref - 182)))
    is_D = (((ref >= 240) & (pred >= 70) & (pred <= 180))
            | ((ref <= 175 / 3) & (pred <= 180) & (pred >= 70))
            | ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref)))

    zone = np.full(ref.shape, 'B', dtype='<U1')
    zone[is_D] = 'D'   # applied lowest-priority-first so a later line can overwrite it --
    zone[is_C] = 'C'   # this reproduces "first matching elif wins" (A checked first) using
    zone[is_E] = 'E'   # "last assignment wins" instead.
    zone[is_A] = 'A'

    return {z: int(np.sum(zone == z)) for z in ['A', 'B', 'C', 'D', 'E']}


def plot_clarke_error_grid(y_true, y_pred, title: str = ''):
    import matplotlib.pyplot as plt

    ref, pred = _to_numpy(y_true), _to_numpy(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(ref, pred, marker='o', color='black', s=8)
    ax.set_title(f'{title} Clarke Error Grid'.strip())
    ax.set_xlabel('Reference Concentration (mg/dl)')
    ax.set_ylabel('Prediction Concentration (mg/dl)')
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect(1.0)

    ax.plot([0, 400], [0, 400], ':', c='black')
    ax.plot([0, 175 / 3], [70, 70], '-', c='black')
    ax.plot([175 / 3, 400 / 1.2], [70, 400], '-', c='black')
    ax.plot([70, 70], [84, 400], '-', c='black')
    ax.plot([0, 70], [180, 180], '-', c='black')
    ax.plot([70, 290], [180, 400], '-', c='black')
    ax.plot([70, 70], [0, 56], '-', c='black')
    ax.plot([70, 400], [56, 320], '-', c='black')
    ax.plot([180, 180], [0, 70], '-', c='black')
    ax.plot([180, 400], [70, 70], '-', c='black')
    ax.plot([240, 240], [70, 180], '-', c='black')
    ax.plot([240, 400], [180, 180], '-', c='black')
    ax.plot([130, 180], [0, 70], '-', c='black')

    for x, y, label in [(30, 15, 'A'), (370, 260, 'B'), (280, 370, 'B'), (160, 370, 'C'),
                         (160, 15, 'C'), (30, 140, 'D'), (370, 120, 'D'), (30, 370, 'E'),
                         (370, 15, 'E')]:
        ax.text(x, y, label, fontsize=15)

    return fig


# ─────────────────────────── Orchestration ───────────────────────────

def evaluate(y_pred_norm, y_true_norm, cgm_mean: float, cgm_std: float,
             horizons: list = HORIZONS) -> dict:

    y_pred = _to_numpy(y_pred_norm) * cgm_std + cgm_mean
    y_true = _to_numpy(y_true_norm) * cgm_std + cgm_mean

    results = {}
    for i, h in enumerate(horizons):
        yp, yt = y_pred[:, i], y_true[:, i]
        zone_counts = clarke_error_grid_zones(yt, yp)
        n = sum(zone_counts.values())
        results[h] = {
            'rmse': rmse(yp, yt),
            'mae': mae(yp, yt),
            'grmse': grmse(yp, yt),
            'ceg': {z: 100.0 * c / n for z, c in zone_counts.items()},
        }
    return results


def evaluate_per_participant(y_pred_norm, y_true_norm, participant_ids, cgm_mean: float,
                              cgm_std: float, horizons: list = HORIZONS) -> dict:

    y_pred_norm = _to_numpy(y_pred_norm)
    y_true_norm = _to_numpy(y_true_norm)
    participant_ids = np.asarray(participant_ids)
    return {
        pid: evaluate(y_pred_norm[participant_ids == pid], y_true_norm[participant_ids == pid],
                      cgm_mean, cgm_std, horizons)
        for pid in sorted(set(participant_ids))
    }


def format_results_table(results: dict) -> str:

    header = (f"{'Horizon (min)':>13} | {'RMSE':>8} | {'MAE':>8} | {'gRMSE':>8} | "
              f"{'Zone A %':>9} | {'Zone A+B %':>10}")
    lines = [header, '-' * len(header)]
    for h, m in results.items():
        a = m['ceg']['A']
        ab = a + m['ceg']['B']
        lines.append(f"{h:>13} | {m['rmse']:>8.2f} | {m['mae']:>8.2f} | {m['grmse']:>8.2f} | "
                      f"{a:>9.2f} | {ab:>10.2f}")
    return '\n'.join(lines)
