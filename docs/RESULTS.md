# Results summary

The numbers below were regenerated on the reduced public sample master dataset (`2023-2026 snippet`) together with the MOEX control block.

## 1. Realized volatility forecast

| Period | MAE | RMSE |
|---|---:|---:|
| Test (2025) | 0.6436 | 0.8947 |
| Control (2026) | 0.7444 | 1.1107 |
| All OOS | 0.6640 | 0.9424 |

## 2. Volume forecast comparison

### Naive benchmark

| Period | MAE | RMSE |
|---|---:|---:|
| Test (2025) | 0.6459 | 1.2011 |
| Control (2026) | 0.9473 | 1.5629 |
| All OOS | 0.7069 | 1.2825 |

### Base model

| Period | MAE | RMSE | OOS R² vs naive |
|---|---:|---:|---:|
| Test (2025) | 0.5856 | 1.0318 | 0.2621 |
| Control (2026) | 0.8285 | 1.3322 | 0.2734 |
| All OOS | 0.6348 | 1.0992 | 0.2655 |

### Full model

| Period | MAE | RMSE | OOS R² vs naive |
|---|---:|---:|---:|
| Test (2025) | 0.5761 | 1.0268 | 0.2692 |
| Control (2026) | 0.8316 | 1.3241 | 0.2823 |
| All OOS | 0.6278 | 1.0935 | 0.2731 |

All values come from `rts_activity_outputs/metrics.json`.

## 3. Interpretation

Against the naive benchmark, the full model improves RMSE by approximately:

- `14.51%` on the 2025 test period
- `15.28%` on the 2026 control period
- `14.74%` on all out-of-sample observations

Against the base model, the full model still wins, but the gain is modest:

- `0.48%` lower RMSE on the 2025 test period
- `0.61%` lower RMSE on the 2026 control period
- `0.52%` lower RMSE on all out-of-sample observations

The main takeaway remains the same: predicted volatility adds useful information for volume forecasting, but the incremental gain over an inertia-only volume model is moderate rather than dramatic.

## 4. Activity regime accuracy

| Period | Accuracy |
|---|---:|
| Test (2025) | 0.5704 |
| Control (2026) | 0.6389 |
| All OOS | 0.5843 |
