# Etch VM Challenge

SiO₂ plasma etching virtual metrology competition.

## Task

Predict etch rate (nm/min) from 49 sensor features (OES, RF, pressure, temperature, gas flow) using data from a CCP etch chamber simulation.

## Data

| File | Description |
|------|-------------|
| `data/train.csv` | 1,500 runs — Chamber_1, lots 1-60, with `etch_rate` label |
| `data/test_features.csv` | 1,000 runs — Chamber_1 + Chamber_2, lots 61-80, unlabeled |
| `data/feature_description.csv` | 49 feature descriptions |

The test set introduces cross-chamber and temporal distribution shift by design.

## Scoring

| Metric | Weight |
|--------|--------|
| R² | 60% |
| MAE | 20% |
| Methodology | 20% |

## Quick Start

```bash
jupyter notebook starter_notebook.ipynb
```

## Methodology Ideas

- **Physics-informed features**: reconstruct RF power from `power_pressure_product`, model non-monotonic break-in aging curve, estimate wall polymer state from OES F/Ar ratio
- **Structured residual modeling**: PLS for linear sensor mapping + GP on `runs_in_pm_cycle` to capture non-linear aging that PLS misses
- **Cross-chamber calibration**: physics-based correction for tool-to-tool variation (rate_scale, coupling_strength differences)
- **Ensemble stacking**: blend PLS, GP-corrected BVM, physics-residual hybrid, and gradient boosting with a meta-learner

## Requirements

- Python >= 3.10
- numpy, pandas, scikit-learn, matplotlib, scipy
- Optional: xgboost, torch
