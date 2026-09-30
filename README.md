# Multimodal Blood Glucose Prediction using CGM and Food Photographs

Companion code for the dissertation of the same name, built on the
[CGMacros](https://physionet.org/content/cgmacros/) dataset.

The models forecast blood glucose 15, 30, 60, 90 and 120 minutes ahead from a 2-hour window of
continuous glucose monitor (CGM) readings. The experiments test whether adding the time since the
last meal and features from the meal photograph improves the forecast, first with centralised
training and then with federated learning.

## Usage

### 1. Install dependencies

Python 3.10 or later, with:

```
pip install torch torchvision transformers numpy pandas scikit-learn xgboost \
            pytorch-forecasting optuna matplotlib tqdm Pillow
```

### 2. Get the data

The processed data is committed in `processed_data/`, so the experiments run without the raw
dataset:

| Path | Contents |
|------|----------|
| `processed_data/processed_5min/` | `train.pkl`, `val.pkl`, `test.pkl` (CGM windows, targets and meal information), `cgm_stats.json` (normalisation statistics), `split_participants.json` (participant split) |
| `processed_data/img_preprocessing/` | `resolved_image_keys.json` (verified meal photographs), `missing_image_samples.json` (meals without a usable photograph) |
| `processed_data/img_preprocessing/img_features/` | `img_features_{resnet18,vit,clip,dinov2}.pkl`, the image features from each encoder |

The raw data is not committed. It is only needed to rebuild `processed_data/` from scratch: in
that case download CGMacros and place it at the repository root as `CGMacros/` (it contains the
`CGMacros-0xx/` folders and `bio.csv`).

### 3. Run the notebooks

Every notebook works both on Google Colab and locally. The first cell sets `PROJECT_ROOT`: on
Colab it mounts Google Drive and expects the project at `MyDrive/dissertation`; locally it uses
the repository root. No other configuration is needed.

Run the notebooks in this order, because later ones load checkpoints saved by earlier ones.
Steps 1 and 4 need the raw dataset and can be skipped, since their outputs are already in
`processed_data/`.

| Step | Notebooks | What it does |
|------|-----------|--------------|
| 1 | `preprocessing/cgm_preprocessing.ipynb`<br>`preprocessing/images_preprocessing.ipynb` | Builds the samples and the verified image list. |
| 2 | `experiments/1_baseline/` | CGM-only baselines. |
| 3 | `experiments/2_best_cgm_encoders/` | Compares the baselines and tunes the GRU. |
| 4 | `experiments/3_initial_multimodal/data_generation/` | Extracts image features with each encoder. |
| 5 | `experiments/3_initial_multimodal/multimodal/` | Trains CGM-only, CGM+Time and CGM+Time+Image. |
| 6 | `experiments/4_diagnostic/` | Diagnostic experiments on the image branch. |
| 7 | `experiments/5_residual_hyper/` | Run `create_multimodal_tuning_configurations.ipynb` first, then the four `tune_*` notebooks. |
| 8 | `experiments/6_final_multimodal/` | Final evaluation on the test split. |
| 9 | `experiments/7_federated_learning/` | `fed0` → `fed1` → `fed2_cgm_only` → `fed2_cgm_time` → `fed2_cgm_time_image`. |

All experiments in this repository were run on Google Colab with the **G4 GPU** runtime (NVIDIA
RTX PRO 6000 Blackwell Server Edition). A GPU is recommended for reproducing them, because each
neural experiment trains 10 seeds.

### 4. Find the results

Each experiment writes to a `results/` folder next to its notebook: one JSON per seed, plus
summary files averaged over seeds. The final test tables are in
`experiments/6_final_multimodal/summary/`. Model checkpoints go to `checkpoints/` folders, which
are not committed.

## Layout

- `preprocessing/` — turns the raw dataset into model-ready samples
- `processed_data/` — the processed samples and image features used by every experiment
- `src/` — shared code imported by the notebooks: data loaders, training loops, metrics, models
- `experiments/` — one notebook per experiment, with its results
