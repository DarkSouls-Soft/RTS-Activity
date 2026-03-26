from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    """Configuration for the RTS activity prototype pipeline."""

    base_data_path: str = str(BASE_DIR / "RTS_daily_RV_sample.csv")
    control_data_path: str = str(BASE_DIR / "moex_control_daily.csv")
    output_dir: str = str(BASE_DIR / "rts_activity_outputs")

    date_col: str = "date"
    rv_col: str = "log_RV"
    volume_col: str = "volume_day"
    activity_col: str = "log_volume"

    lags: tuple[int, ...] = (1, 2, 5)
    init_window: int = 252

    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    control_start: str = "2026-01-01"
    control_end: str | None = None

    use_macro: bool = False
    macro_cols: tuple[str, ...] = (
        "brent_ret",
        "usdrub_ret",
        "key_rate_level",
        "key_rate_change",
    )

    regime_lookback: int = 126
    regime_min_periods: int = 60
    regime_low_q: float = 0.25
    regime_high_q: float = 0.75
    low_label: str = "low"
    normal_label: str = "normal"
    high_label: str = "high"


class DataError(ValueError):
    """Raised when the input data does not meet expected constraints."""


# -----------------------------
# Data loading and validation
# -----------------------------

def load_dataset(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise DataError(f"Missing required date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def normalize_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.init_window <= 0:
        raise DataError("init_window must be positive.")

    if not 0 < cfg.regime_low_q < cfg.regime_high_q < 1:
        raise DataError("Regime quantiles must satisfy 0 < low_q < high_q < 1.")

    if cfg.regime_min_periods > cfg.regime_lookback:
        raise DataError("regime_min_periods cannot exceed regime_lookback.")

    train_end = pd.Timestamp(cfg.train_end)
    test_start = pd.Timestamp(cfg.test_start)
    test_end = pd.Timestamp(cfg.test_end)
    control_start = pd.Timestamp(cfg.control_start)

    if not (train_end < test_start <= test_end < control_start):
        raise DataError(
            "Date split must satisfy train_end < test_start <= test_end < control_start."
        )


def validate_dataset(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    required_cols = {cfg.date_col, cfg.rv_col, cfg.volume_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = [cfg.rv_col, cfg.volume_col]
    bad_numeric = [col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if bad_numeric:
        raise DataError(
            "These columns are not numeric after loading/merging: "
            f"{bad_numeric}. Check CSV formatting or numeric coercion."
        )

    null_counts = {col: int(df[col].isna().sum()) for col in numeric_cols if df[col].isna().any()}
    if null_counts:
        raise DataError(
            "Numeric coercion produced NaNs in required columns: "
            f"{null_counts}. Check malformed values in the source CSV files."
        )

    non_positive_volume = int((df[cfg.volume_col] <= 0).sum())
    if non_positive_volume > 0:
        raise DataError(
            f"Column '{cfg.volume_col}' contains {non_positive_volume} non-positive values. "
            "Cannot take logarithm of volume."
        )


# -----------------------------
# Merge base and control data
# -----------------------------

def merge_control_dataset(base_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Keep train/test only from master, control only from MOEX control file."""
    base = base_df.copy()
    base = normalize_numeric_columns(base, [cfg.rv_col, cfg.volume_col, *cfg.macro_cols])
    base["data_source"] = "master"

    # Strict rule: master is used only before control_start.
    base = base[base[cfg.date_col] < pd.Timestamp(cfg.control_start)].copy()

    if not cfg.control_data_path:
        return base.reset_index(drop=True)

    path = Path(cfg.control_data_path)
    if not path.exists():
        return base.reset_index(drop=True)

    control_df = load_dataset(str(path), cfg.date_col)
    needed = [cfg.date_col, cfg.rv_col, cfg.volume_col]
    missing = [c for c in needed if c not in control_df.columns]
    if missing:
        raise DataError(f"Control dataset is missing columns: {missing}")

    control_df = control_df[needed].copy()
    control_df = normalize_numeric_columns(control_df, [cfg.rv_col, cfg.volume_col])
    control_df["data_source"] = "moex_control"

    merged = pd.concat([base, control_df], ignore_index=True, sort=False)
    merged = merged.sort_values([cfg.date_col, "data_source"]).drop_duplicates(
        subset=[cfg.date_col], keep="last"
    )
    merged = merged.sort_values(cfg.date_col).reset_index(drop=True)
    merged = normalize_numeric_columns(merged, [cfg.rv_col, cfg.volume_col, *cfg.macro_cols])
    return merged


# -----------------------------
# Feature engineering
# -----------------------------

def add_base_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out = normalize_numeric_columns(out, [cfg.rv_col, cfg.volume_col, *cfg.macro_cols])
    out[cfg.activity_col] = np.log(out[cfg.volume_col].astype(float))

    for lag in cfg.lags:
        out[f"{cfg.rv_col}_lag{lag}"] = out[cfg.rv_col].shift(lag)
        out[f"{cfg.activity_col}_lag{lag}"] = out[cfg.activity_col].shift(lag)

    if cfg.use_macro:
        for col in cfg.macro_cols:
            if col not in out.columns:
                raise DataError(f"Macro column '{col}' is missing from the dataset.")

    return out


def build_feature_lists(cfg: PipelineConfig) -> tuple[list[str], list[str], list[str]]:
    rv_features = [f"{cfg.rv_col}_lag{lag}" for lag in cfg.lags]
    volume_features = [f"{cfg.activity_col}_lag{lag}" for lag in cfg.lags]
    macro_features = list(cfg.macro_cols) if cfg.use_macro else []
    return rv_features, volume_features, macro_features


def prepare_model_frame(df: pd.DataFrame, required_cols: Sequence[str]) -> pd.DataFrame:
    return df.dropna(subset=list(required_cols)).reset_index(drop=True)


def assign_periods(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out["period"] = "unused"

    train_mask = (
        (out["data_source"] == "master")
        & (out[cfg.date_col] <= pd.Timestamp(cfg.train_end))
    )
    out.loc[train_mask, "period"] = "train"

    test_mask = (
        (out["data_source"] == "master")
        & (out[cfg.date_col] >= pd.Timestamp(cfg.test_start))
        & (out[cfg.date_col] <= pd.Timestamp(cfg.test_end))
    )
    out.loc[test_mask, "period"] = "test"

    control_end = pd.Timestamp.max if cfg.control_end is None else pd.Timestamp(cfg.control_end)
    control_mask = (
        (out["data_source"] == "moex_control")
        & (out[cfg.date_col] >= pd.Timestamp(cfg.control_start))
        & (out[cfg.date_col] <= control_end)
    )
    out.loc[control_mask, "period"] = "control"

    return out


# -----------------------------
# Forecasting helpers
# -----------------------------

def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_benchmark: pd.Series | None = None,
) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics: dict[str, float] = {
        "mae": float(mae),
        "rmse": float(rmse),
    }

    if y_benchmark is not None:
        sse_model = float(np.sum((y_true - y_pred) ** 2))
        sse_benchmark = float(np.sum((y_true - y_benchmark) ** 2))
        metrics["oos_r2_vs_benchmark"] = np.nan if sse_benchmark == 0 else 1 - sse_model / sse_benchmark

    return metrics


def summarize_metrics_by_period(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    benchmark_col: str | None = None,
    periods: Sequence[str] = ("test", "control"),
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    for period in periods:
        part = df[df["period"] == period].dropna(subset=[actual_col, pred_col]).copy()
        if len(part) == 0:
            out[period] = {}
            continue

        y_bench = None if benchmark_col is None else part[benchmark_col]
        out[period] = evaluate_forecast(part[actual_col], part[pred_col], y_benchmark=y_bench)

    all_oos = df[df["period"].isin(periods)].dropna(subset=[actual_col, pred_col]).copy()
    if len(all_oos) == 0:
        out["oos_all"] = {}
    else:
        y_bench = None if benchmark_col is None else all_oos[benchmark_col]
        out["oos_all"] = evaluate_forecast(all_oos[actual_col], all_oos[pred_col], y_benchmark=y_bench)

    return out


# -----------------------------
# Stage 1: RV forecast
# -----------------------------

def run_stage_1_rv_forecast(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rv_features, _, macro_features = build_feature_lists(cfg)
    rv_required = [cfg.rv_col, *rv_features, *macro_features]
    rv_df = prepare_model_frame(df, rv_required)

    if len(rv_df) <= cfg.init_window:
        raise DataError(
            f"Not enough rows for RV forecast after lagging: {len(rv_df)} rows, init_window={cfg.init_window}."
        )

    rv_stage_features = [*rv_features, *macro_features]
    rv_df["pred_log_RV_oos"] = np.nan

    for t in range(cfg.init_window, len(rv_df)):
        train = rv_df.iloc[:t]
        test = rv_df.iloc[[t]]

        model = LinearRegression()
        model.fit(train[rv_stage_features], train[cfg.rv_col])
        rv_df.loc[rv_df.index[t], "pred_log_RV_oos"] = model.predict(test[rv_stage_features])[0]

    rv_metrics = summarize_metrics_by_period(
        rv_df,
        actual_col=cfg.rv_col,
        pred_col="pred_log_RV_oos",
    )
    return rv_df, rv_metrics


# -----------------------------
# Stage 2: Volume forecast
# -----------------------------

def run_stage_2_volume_forecast(
    rv_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    _, volume_features, _ = build_feature_lists(cfg)

    vol_required = [cfg.activity_col, *volume_features, "pred_log_RV_oos", "period"]
    volume_df = prepare_model_frame(rv_df, vol_required)

    volume_df["pred_log_volume_base"] = np.nan
    volume_df["pred_log_volume_full"] = np.nan
    volume_df["naive_log_volume"] = volume_df[f"{cfg.activity_col}_lag1"]

    base_feature_cols = list(volume_features)
    full_feature_cols = [*volume_features, "pred_log_RV_oos"]

    oos_targets = volume_df["period"].isin(["test", "control"])
    oos_idx = volume_df.index[oos_targets]

    if len(oos_idx) == 0:
        raise DataError("No test/control rows available for stage-2 volume forecasting.")

    for idx in oos_idx:
        train = volume_df.loc[: idx - 1]
        if len(train) < cfg.init_window:
            continue

        test = volume_df.loc[[idx]]

        model_base = LinearRegression()
        model_base.fit(train[base_feature_cols], train[cfg.activity_col])
        volume_df.loc[idx, "pred_log_volume_base"] = model_base.predict(test[base_feature_cols])[0]

        model_full = LinearRegression()
        model_full.fit(train[full_feature_cols], train[cfg.activity_col])
        volume_df.loc[idx, "pred_log_volume_full"] = model_full.predict(test[full_feature_cols])[0]

    benchmark_metrics = summarize_metrics_by_period(
        volume_df,
        actual_col=cfg.activity_col,
        pred_col="naive_log_volume",
    )
    base_metrics = summarize_metrics_by_period(
        volume_df,
        actual_col=cfg.activity_col,
        pred_col="pred_log_volume_base",
        benchmark_col="naive_log_volume",
    )
    full_metrics = summarize_metrics_by_period(
        volume_df,
        actual_col=cfg.activity_col,
        pred_col="pred_log_volume_full",
        benchmark_col="naive_log_volume",
    )

    return volume_df, benchmark_metrics, base_metrics, full_metrics


# -----------------------------
# Dynamic regimes
# -----------------------------

def rolling_regime_thresholds(
    series: pd.Series,
    lookback: int,
    min_periods: int,
    low_q: float,
    high_q: float,
) -> pd.DataFrame:
    history = series.shift(1)
    low = history.rolling(lookback, min_periods=min_periods).quantile(low_q)
    high = history.rolling(lookback, min_periods=min_periods).quantile(high_q)
    return pd.DataFrame({"regime_q_low": low, "regime_q_high": high}, index=series.index)


def classify_regime_series(
    values: pd.Series,
    low_thresholds: pd.Series,
    high_thresholds: pd.Series,
    cfg: PipelineConfig,
) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype="object")
    valid = values.notna() & low_thresholds.notna() & high_thresholds.notna()
    if not valid.any():
        return out

    out.loc[valid & (values <= low_thresholds)] = cfg.low_label
    out.loc[valid & (values >= high_thresholds)] = cfg.high_label
    out.loc[valid & (values > low_thresholds) & (values < high_thresholds)] = cfg.normal_label
    return out


def add_dynamic_activity_regimes(volume_df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    out = volume_df.copy()
    thresholds = rolling_regime_thresholds(
        series=out[cfg.activity_col],
        lookback=cfg.regime_lookback,
        min_periods=cfg.regime_min_periods,
        low_q=cfg.regime_low_q,
        high_q=cfg.regime_high_q,
    )
    out = pd.concat([out, thresholds], axis=1)

    out["actual_regime"] = classify_regime_series(
        out[cfg.activity_col],
        out["regime_q_low"],
        out["regime_q_high"],
        cfg,
    )
    out["predicted_regime"] = classify_regime_series(
        out["pred_log_volume_full"],
        out["regime_q_low"],
        out["regime_q_high"],
        cfg,
    )
    out["pred_activity_regime"] = out["predicted_regime"]
    out["actual_activity_regime"] = out["actual_regime"]

    regime_mask = (
        out["period"].isin(["test", "control"])
        & out["actual_regime"].notna()
        & out["predicted_regime"].notna()
    )
    out["regime_hit"] = np.where(
        regime_mask,
        (out["actual_regime"] == out["predicted_regime"]).astype(int),
        np.nan,
    )

    acc_by_period: dict[str, float] = {}
    for period in ["test", "control"]:
        part = out[(out["period"] == period) & out["regime_hit"].notna()]
        acc_by_period[period] = float(part["regime_hit"].mean()) if len(part) > 0 else np.nan

    all_part = out[out["regime_hit"].notna()]
    acc_by_period["oos_all"] = float(all_part["regime_hit"].mean()) if len(all_part) > 0 else np.nan
    return out, acc_by_period


# -----------------------------
# Output formatting
# -----------------------------

def add_human_readable_volume_columns(results: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = results.copy()
    out["actual_volume"] = np.exp(out[cfg.activity_col])
    out["naive_volume"] = np.exp(out["naive_log_volume"])
    out["pred_volume_base"] = np.exp(out["pred_log_volume_base"])
    out["pred_volume_full"] = np.exp(out["pred_log_volume_full"])
    return out


def build_result_frame(volume_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    keep_cols = [
        cfg.date_col,
        "data_source",
        "period",
        cfg.rv_col,
        cfg.activity_col,
        "pred_log_RV_oos",
        "naive_log_volume",
        "pred_log_volume_base",
        "pred_log_volume_full",
        "regime_q_low",
        "regime_q_high",
        "actual_regime",
        "predicted_regime",
        "pred_activity_regime",
        "actual_activity_regime",
        "regime_hit",
    ]
    result_df = volume_df[keep_cols].copy()
    return add_human_readable_volume_columns(result_df, cfg)


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, dict]]:
    validate_config(cfg)

    base_df = load_dataset(cfg.base_data_path, cfg.date_col)
    df = merge_control_dataset(base_df, cfg)
    validate_dataset(df, cfg)
    df = add_base_features(df, cfg)
    df = assign_periods(df, cfg)

    print("\nRows by source:")
    print(df["data_source"].value_counts(dropna=False))
    print("\nRows by period:")
    print(df["period"].value_counts(dropna=False))
    print("\nDate range by source:")
    print(df.groupby("data_source")[cfg.date_col].agg(["min", "max", "count"]))

    rv_df, rv_metrics = run_stage_1_rv_forecast(df, cfg)
    volume_df, benchmark_metrics, base_metrics, full_metrics = run_stage_2_volume_forecast(rv_df, cfg)
    volume_df, regime_accuracy = add_dynamic_activity_regimes(volume_df, cfg)

    metrics: dict[str, dict] = {
        "rv_model": rv_metrics,
        "volume_naive": benchmark_metrics,
        "volume_base": base_metrics,
        "volume_full": full_metrics,
        "activity_regime": {
            "accuracy": regime_accuracy,
            "lookback": cfg.regime_lookback,
            "min_periods": cfg.regime_min_periods,
            "low_q": cfg.regime_low_q,
            "high_q": cfg.regime_high_q,
        },
    }

    result_df = build_result_frame(volume_df, cfg)
    return result_df, metrics


def save_outputs(results: pd.DataFrame, metrics: dict[str, dict], output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def print_report(metrics: dict[str, dict]) -> None:
    print("\n=== RTS activity prototype report ===")
    for block, values in metrics.items():
        print(f"\n[{block}]")
        if isinstance(values, dict):
            for metric_name, metric_value in values.items():
                if isinstance(metric_value, dict):
                    print(f"  {metric_name}:")
                    for sub_name, sub_value in metric_value.items():
                        if isinstance(sub_value, float):
                            print(f"    {sub_name:>20}: {sub_value:.6f}")
                        else:
                            print(f"    {sub_name:>20}: {sub_value}")
                elif isinstance(metric_value, float):
                    print(f"{metric_name:>24}: {metric_value:.6f}")
                else:
                    print(f"{metric_name:>24}: {metric_value}")
        else:
            print(values)


if __name__ == "__main__":
    cfg = PipelineConfig()
    results, metrics = run_pipeline(cfg)
    save_outputs(results, metrics, cfg.output_dir)
    print_report(metrics)
    print(f"\nSaved outputs to: {cfg.output_dir}")
