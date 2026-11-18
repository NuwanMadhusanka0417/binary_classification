# NCI experiment sweep

The sweep compares molecular representations, model choices and class
balancing across all datasets. Only XGBoost is fine-tuned; Random Forest and
Logistic Regression are fixed baselines.

## Experiment grid

| Factor | Values tested | Purpose |
|---|---|---|
| Dataset group | ED, tox21, HTSFP10 | Test generalisation across datasets |
| Features | gvfa, concat, bundle, rdkit | Compare molecular representations |
| GVFA dimension | 100, 500, 1000, 2000, 5000, 10000 | Find the useful embedding size |
| Classifier | XGBoost, Random Forest, Logistic Regression | Compare model families |
| Class balancing | Off, on | Measure the effect of imbalance correction |
| XGBoost tuning | 0 or 50 Optuna trials | Compare untuned and tuned XGBoost |

## XGBoost Optuna search

| Parameter | Search range | Sampling |
|---|---:|---|
| learning rate | 0.01–0.30 | Log continuous |
| maximum depth | 3–10 | Integer |
| minimum child weight | 1–20 | Integer |
| row subsampling | 0.60–1.00 | Continuous |
| feature sampling per node | 0.20–1.00 | Continuous |
| L2 regularisation | 0.01–100 | Log continuous |
| L1 regularisation | 0.001–10 | Log continuous |
| gamma | 0–5 | Continuous |

Optuna runs 50 trials per XGBoost configuration. It selects parameters that
maximise singleton prediction efficiency at 80% confidence while requiring
valid conformal error rates for both classes.