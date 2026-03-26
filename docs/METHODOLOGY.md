# Methodology

## Objective

The goal is to forecast expected market activity for RTS index futures.

Instead of predicting price direction, the project focuses on two observable quantities:

- realized volatility,
- trading volume.

## Public sample setup

The GitHub version uses a reduced master dataset:

- period: `2023-01-03` to `2026-01-06`
- columns: `date`, `log_RV`, `volume_day`

This is enough for the current baseline because no extra dummy or macro columns are required.

## Targets

### Stage 1 target

- `log_RV`

This is the logarithm of daily realized volatility.

### Stage 2 target

- `log_volume`

This is created inside the pipeline from `volume_day`.

## Time split

The forecasting setup uses a strict chronological split:

- Train: up to `2024-12-31`
- Test: `2025-01-01` to `2025-12-31`
- Control: from `2026-01-01` onward

## Feature construction

The current baseline uses lagged features at horizons:

- 1 day
- 2 days
- 5 days

Stage 1 uses:

- `log_RV_lag1`
- `log_RV_lag2`
- `log_RV_lag5`

Stage 2 uses:

- `log_volume_lag1`
- `log_volume_lag2`
- `log_volume_lag5`
- `pred_log_RV_oos`

## Model design

### Stage 1: RV forecast

An expanding-window linear regression model predicts one-step-ahead `log_RV`.

### Stage 2: volume forecast

Three alternatives are compared:

- naive benchmark: yesterday's `log_volume`
- base model: lagged `log_volume` only
- full model: lagged `log_volume` plus predicted volatility

## Warm-up window

The public sample configuration uses an initial warm-up window of `252` trading days.

This is shorter than the original research setup, but it is the practical choice for the reduced GitHub sample because it preserves a meaningful test/control evaluation window.

## Evaluation

The project reports:

- MAE
- RMSE
- out-of-sample R² relative to the naive benchmark

## Activity regimes

The volume forecast is also converted into an activity regime using rolling quantile thresholds:

- lookback: `126`
- minimum periods: `60`
- low quantile: `0.25`
- high quantile: `0.75`

The labels are:

- `low`
- `normal`
- `high`

## Main limitation

The GitHub sample is intentionally compact. It is good enough to demonstrate the full analytical workflow, but it is not the same as distributing the full private research history.
