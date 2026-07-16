# Dissertation Code

Companion code repository for the dissertation "Multimodal Blood Glucose Prediction using CGM and Food Photographs" (CGMacros dataset).

## Structure

- `dataset/` — dataset analysis / EDA notebooks
- `preprocessing/` — preprocessing pipeline notebooks (TBD)
- `modelling/` — model training/evaluation notebooks (TBD)
- reusable code (model definitions, shared preprocessing utilities) lives in top-level `.py` modules, imported by the notebooks — not inlined.

Convention: **one notebook = one experiment.**

Raw data (`CGMacros/`, `bio.csv`) is not committed; it is expected either locally at the repo root or mounted from Google Drive when run in Colab.
