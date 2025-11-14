# Neutron Stars Pulsars Classification

## Overview
This repository explores a pulsar dataset to build a baseline model that distinguishes real pulsars from radio frequency interference and other astrophysical sources. The work is organized as a notebook-driven workflow that covers exploratory data analysis, feature engineering, dimensionality reduction, and a first supervised learning model (logistic regression with class balancing).

This is a Data Science challenge project proposed by RocketSeat.

## Dataset
- 17,898 observations and 8 predictive attributes derived from pulsar radio signals.
- Binary target `target_class`, where `1` denotes true pulsars and `0` captures non-pulsar detections.
- Severe class imbalance (16,259 negatives vs. 1,639 positives) motivates stratified sampling and the use of class weights during modelling.

### Feature Glossary
- `mean_integrated_profile` – Average intensity of the integrated pulse profile.
- `std_integrated_profile` – Standard deviation of the integrated profile.
- `kurtosis_integrated_profile` – Excess kurtosis of the integrated profile, highlighting heavy tails.
- `skewness_integrated_profile` – Skewness of the integrated profile, capturing asymmetry.
- `mean_dm_snr_curve` – Mean signal-to-noise ratio across dispersion measures.
- `std_dm_snr_curve` – Standard deviation of the DM-SNR curve.
- `kurtosis_dm_snr_curve` – Excess kurtosis of the DM-SNR curve.
- `skewness_dm_snr_curve` – Skewness of the DM-SNR curve.

Raw data lives in `data/raw/pulsar.csv`; the processed version exported by the preprocessing notebook is stored in `data/processed/pulsar_processed.csv`.

## Repository Layout
```text
.
├── data/
│   ├── raw/                # Original HTRU2 dataset (CSV)
│   └── processed/          # PCA-transformed dataset saved by the notebooks
├── notebooks/              # Jupyter notebooks for EDA, preprocessing, and modeling
├── main.py                 # Placeholder entry point
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions for uv
└── README.md
```

## Environment Setup
1. Install Python 3.12 or later.
2. (Recommended) Install uv and run `uv sync` to create the virtual environment from `pyproject.toml` and `uv.lock`.
3. Alternatively, create a virtual environment manually and install dependencies with `pip install .` or `pip install -r <(uv pip compile pyproject.toml)`.
4. Activate the environment and launch Jupyter with `uv run jupyter lab` (or `pip install jupyterlab` followed by `jupyter lab`).

## Workflow
### 1. Exploratory Data Analysis (`notebooks/1.0.0-dataset-loading-and-eda.ipynb`)
- Loads the raw dataset, inspects schema, and quantifies class imbalance.
- Highlights skewed distributions, heavy tails, and numerous outliers in both integrated-profile and DM-SNR features.
- Uses correlation heatmaps and VIF calculations to confirm substantial multicollinearity across the eight predictors.

### 2. Data Preprocessing (`notebooks/1.1.0-data-preprocessing.ipynb`)
- Applies the Yeo-Johnson power transform (grouped by class) to reduce skewness while handling negative values.
- Performs winsorization (1st–99th percentiles) to limit the influence of extreme outliers.
- Standardizes features and runs PCA retaining 95% of the variance, which compresses the feature space from 8 raw attributes to 5 principal components.
- Saves the resulting feature matrix plus target into `data/processed/pulsar_processed.csv` for downstream modeling.

### 3. Model Training (`notebooks/[2.1.0-lr-model-training.ipynb, 2.2.1-rf-model-training.ipynb, 2.2.2-gb-model-training.ipynb, 2.2.3-xgb-model-training.ipynb]`)
- Loads processed dataset.
- Splits the processed dataset with stratified 80/20 train-test splits.
- Trains, considering class imbalance, the following classifiers:
  - a logistic regression classifier;
  - a random forest classifier;
  - a gradient boosting classifier;
  - an XGBoost classifier.
- Tunes hyperparameters via random search with 5-fold stratified cross-validation, optimizing for F1-score.
- Evaluates each model using accuracy, precision, recall, F1-score, ROC curves, and confusion matrix visualizations.
- Selects the best model based on cross-validated F1-score.

## Models Performance
All scores reported on the held-out test set derived from `train_test_split` (random_state=42).

For LR, I performed bayesian hyperparameter optimization. Also, I applied `TunedThresholdClassifierCV` to find the optimal decision threshold that maximizes F1-score on cross-validated folds.

To explore different approaches, I applied Random Search CV for the tree-based models (RF, GB, XGB).

### Logistic Regression
```text
              precision    recall  f1-score   support

           0      0.985     0.992     0.988      3252
           1      0.914     0.848     0.880       328

    accuracy                          0.979      3580
   macro avg      0.950     0.920     0.934      3580
weighted avg      0.978     0.979     0.978      3580
```

### Random Forest
```text
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      3252
           1       0.90      0.88      0.89       328

    accuracy                           0.98      3580
   macro avg       0.95      0.93      0.94      3580
weighted avg       0.98      0.98      0.98      3580
```

### Gradient Boosting
```text
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.99      0.99      3252
           1       0.93      0.83      0.88       328

    accuracy                           0.98      3580
   macro avg       0.95      0.91      0.93      3580
weighted avg       0.98      0.98      0.98      3580
```

### XGBoost
```text
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.98      0.99      3252
           1       0.84      0.89      0.87       328

    accuracy                           0.97      3580
   macro avg       0.92      0.94      0.93      3580
weighted avg       0.98      0.97      0.98      3580
```

> Therefore, the Random Forest model slightly outperforms the others in terms of F1-score, achieving a good balance between precision and recall.

## Reproducing Results
1. Follow the setup instructions and launch Jupyter Lab.
2. Execute the notebooks in numerical order.
3. Inspect intermediate outputs and final model evaluation metrics.
4. Adjust hyperparameters or preprocessing steps as desired to experiment with model performance.
5. Save any modified notebooks or export results for reporting.

## Roadmap Ideas
- Package the preprocessing steps into a reproducible `scikit-learn` pipeline and expose it via a CLI or web API for batch scoring.
- Add Python scripts to modularize data loading, preprocessing, model training, and evaluation.
- Automate evaluation with unit tests and CI workflows to track performance regressions.
