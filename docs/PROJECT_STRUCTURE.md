# Project structure

## Current public-facing layout

The repository is organized around a compact, publishable workflow:

```text
.
├── README.md
├── app.py
├── moex_api.py
├── moex_contract_segments.json
├── requirements.txt
├── rts_activity_pipeline.py
├── update_data.py
├── docs/
├── images/
├── RTS_daily_RV_sample.csv
├── moex_control_daily.csv
└── rts_activity_outputs/
```

## Main roles

- `RTS_daily_RV_sample.csv`
  Public sample master dataset with only `date`, `log_RV`, and `volume_day`.

- `moex_control_daily.csv`
  External control block refreshed from MOEX ISS.

- `rts_activity_pipeline.py`
  Forecasting pipeline and metric generation.

- `app.py`
  Streamlit dashboard for inspection of forecasts and diagnostics.

- `update_data.py`
  Loader for the MOEX control dataset.

- `moex_api.py`
  Minimal MOEX ISS client.

- `docs/`
  Supporting methodology, data, and result notes.

- `images/`
  Dashboard screenshots used in GitHub documentation.

## Why the layout is intentionally simple

This repository is meant to be easy to open on GitHub and easy to run locally. The public version therefore keeps:

- the working application code,
- a lightweight sample dataset,
- generated documentation,
- and dashboard screenshots.

It avoids turning the repository into a dump of raw market history or old experimental scripts.
