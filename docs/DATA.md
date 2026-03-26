# Data description

## Overview

The public repository uses a compact RTS sample dataset plus a MOEX control block.

This is deliberate: the goal is to keep the GitHub version lightweight while preserving the full modeling flow of the prototype.

## Master dataset used in the public repo

File: `RTS_daily_RV_sample.csv`

Public sample characteristics:

- period: `2023-01-03` to `2026-01-06`
- rows: `796`
- columns:
  - `date`
  - `log_RV`
  - `volume_day`

Only these columns are retained because they are the only fields required by the current baseline pipeline.

## Why the dataset is reduced

The original research workflow used a wider historical file with extra diagnostics, dummy variables, and macro columns.

The public GitHub version strips that down for two reasons:

- faster cloning and easier inspection,
- only the fields actually consumed by the current code are kept.

In the baseline pipeline:

- `log_RV` is the stage-1 prediction target,
- `volume_day` is transformed into `log_volume` for stage 2,
- `date` defines the chronological split and plotting axis.

## External control sample

File: `moex_control_daily.csv`

This file is refreshed by `update_data.py` from MOEX ISS candles and is used as the external control block from `2026-01-01` onward.

Default contract segments:

- `RIH6` from `2026-01-01` to `2026-03-19`
- `RIM6` from `2026-03-20` onward

Required columns in the control file:

- `date`
- `log_RV`
- `volume_day`

## Produced outputs

The forecasting pipeline writes:

- `rts_activity_outputs/predictions.csv`
- `rts_activity_outputs/metrics.json`

These files store out-of-sample forecasts, model-comparison metrics, and activity-regime diagnostics.

## Important caveat

The public sample master dataset and the MOEX control dataset are constructed through different technical routes. Because of that, the control period should be treated as an external practical validation block, not as a perfectly identical continuation of the master sample.
