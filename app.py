from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from update_data import (
    UpdateConfig,
    append_segment,
    ensure_segments_file,
    load_segments,
    update_control_daily_csv,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'rts_activity_outputs'
METRICS_PATH = OUTPUT_DIR / 'metrics.json'
PREDICTIONS_PATH = OUTPUT_DIR / 'predictions.csv'
MASTER_DATA_PATH = BASE_DIR / 'RTS_daily_RV_sample.csv'
PIPELINE_SCRIPT = BASE_DIR / 'rts_activity_pipeline.py'
SEGMENTS_PATH = BASE_DIR / 'moex_contract_segments.json'
CONTROL_DATA_PATH = BASE_DIR / 'moex_control_daily.csv'

PERIOD_LABELS = {
    'test': 'Тестовый период 2025',
    'control': 'Контрольный период 2026',
    'oos_all': 'Весь внешний период',
    'full_history': 'Весь сниппет 2023-2026',
}
REGIME_LABELS = {
    'low': '🔵 Низкая',
    'normal': '🟡 Нормальная',
    'high': '🔴 Повышенная',
}
LOOKBACK_OPTIONS = {
    'Последние 90': 90,
    'Последние 180': 180,
    'Последние 252': 252,
    'Последние 504': 504,
    'Последние 1000': 1000,
    'Весь период': None,
}
MODEL_LABELS = {
    'naive': 'Эталонный прогноз',
    'base': 'Инерционная модель',
    'full': 'Расширенная модель',
}
def fmt_num(x: Any, digits: int = 4) -> str:
    try:
        x = float(x)
        if np.isnan(x):
            return '—'
        return f'{x:.{digits}f}'
    except Exception:
        return '—'

def fmt_int(x: Any) -> str:
    try:
        x = float(x)
        if np.isnan(x):
            return '—'
        return f'{x:,.0f}'
    except Exception:
        return '—'

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan
@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df
@st.cache_data(show_spinner=False)
def load_master_context(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=['date', 'log_RV', 'actual_volume', 'log_volume'])

    df = pd.read_csv(p)
    if 'date' not in df.columns:
        return pd.DataFrame(columns=['date', 'log_RV', 'actual_volume', 'log_volume'])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'volume_day' in df.columns:
        df['volume_day'] = pd.to_numeric(df['volume_day'], errors='coerce')
        df['actual_volume'] = df['volume_day']
        df['log_volume'] = np.where(df['volume_day'] > 0, np.log(df['volume_day']), np.nan)
    else:
        df['actual_volume'] = np.nan
        df['log_volume'] = np.nan

    if 'log_RV' in df.columns:
        df['log_RV'] = pd.to_numeric(df['log_RV'], errors='coerce')
    else:
        df['log_RV'] = np.nan

    return df[['date', 'log_RV', 'actual_volume', 'log_volume']].sort_values('date').reset_index(drop=True)


def available_periods(pred: pd.DataFrame, master_ctx: pd.DataFrame) -> list[str]:
    opts: list[str] = []
    if not master_ctx.empty:
        opts.append('full_history')
    opts.append('oos_all')
    if 'period' in pred.columns:
        if (pred['period'] == 'test').any():
            opts.append('test')
        if (pred['period'] == 'control').any():
            opts.append('control')
    return opts

def get_metric_block(metrics: dict[str, Any], section: str, period: str) -> dict[str, Any]:
    sec = metrics.get(section, {})
    if not isinstance(sec, dict):
        return {}

    metric_period = 'oos_all' if period == 'full_history' else period
    if metric_period in sec and isinstance(sec[metric_period], dict):
        return sec[metric_period]

    if section == 'activity_regime':
        acc = sec.get('accuracy', {})
        return {
            'accuracy': acc.get(metric_period),
            'lookback': sec.get('lookback'),
            'min_periods': sec.get('min_periods'),
            'low_q': sec.get('low_q'),
            'high_q': sec.get('high_q'),
        }
    return sec

def build_context_history(master_ctx: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    ctx = master_ctx.copy()
    pred_cols = [c for c in ['date', 'log_RV', 'actual_volume', 'log_volume'] if c in pred.columns]
    pred_actual = pred[pred_cols].copy() if pred_cols else pd.DataFrame(columns=['date'])
    combined = pd.concat([ctx, pred_actual], ignore_index=True, sort=False)
    combined = combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    return combined.reset_index(drop=True)


def build_view_df(pred: pd.DataFrame, context_df: pd.DataFrame, selected_period: str, only_oos: bool, lookback_rows: int | None) -> pd.DataFrame:
    pred_view = pred.copy().sort_values('date')

    if selected_period in {'test', 'control'} and 'period' in pred_view.columns:
        pred_view = pred_view[pred_view['period'] == selected_period].copy()
    elif selected_period == 'oos_all' and 'period' in pred_view.columns:
        pred_view = pred_view[pred_view['period'].isin(['test', 'control'])].copy()

    if only_oos and selected_period != 'full_history' and 'period' in pred_view.columns:
        pred_view = pred_view[pred_view['period'].isin(['test', 'control'])].copy()

    if selected_period == 'full_history':
        base = context_df.copy().sort_values('date')
        merged = base.merge(pred_view, on='date', how='left', suffixes=('', '_pred'))
        if 'period' not in merged.columns:
            merged['period'] = np.where(
                merged['date'] <= pd.Timestamp('2024-12-31'), 'train',
                np.where(merged['date'] <= pd.Timestamp('2025-12-31'), 'test', 'control')
            )
        view = merged
    else:
        view = pred_view.copy()

    view = view.sort_values('date')
    if lookback_rows is not None and len(view) > lookback_rows:
        view = view.tail(lookback_rows)
    return view.reset_index(drop=True)

def get_last_valid_row(pred: pd.DataFrame, selected_period: str) -> pd.Series | None:
    use = pred.copy()
    if selected_period in {'test', 'control'} and 'period' in use.columns:
        use = use[use['period'] == selected_period].copy()
    if selected_period == 'oos_all' and 'period' in use.columns:
        use = use[use['period'].isin(['test', 'control'])].copy()
    needed = [c for c in ['pred_log_RV_oos', 'pred_log_volume_full'] if c in use.columns]
    if needed:
        use = use.dropna(subset=needed)
    if use.empty:
        return None
    return use.sort_values('date').iloc[-1]

def make_volume_chart(df: pd.DataFrame, use_log_scale: bool) -> go.Figure:
    fig = go.Figure()
    actual_col = 'log_volume' if use_log_scale else 'actual_volume'
    base_col = 'pred_log_volume_base' if use_log_scale else 'pred_volume_base'
    full_col = 'pred_log_volume_full' if use_log_scale else 'pred_volume_full'
    y_title = 'log(объём)' if use_log_scale else 'Объём торгов'

    if actual_col in df.columns:
        actual_df = df.dropna(subset=['date', actual_col])
        fig.add_trace(go.Scatter(x=actual_df['date'], y=actual_df[actual_col], mode='lines', name='Фактический ряд', line=dict(width=2.2)))
    if base_col in df.columns:
        base_df = df.dropna(subset=['date', base_col])
        fig.add_trace(go.Scatter(x=base_df['date'], y=base_df[base_col], mode='lines', name='Инерционная модель', line=dict(width=2, dash='dot')))
    if full_col in df.columns:
        full_df = df.dropna(subset=['date', full_col])
        fig.add_trace(go.Scatter(x=full_df['date'], y=full_df[full_col], mode='lines', name='Расширенная модель', line=dict(width=3, dash='dash')))

    fig.update_layout(
        template='plotly_dark', height=430, title='Объём торгов: фактический и прогнозный ряды',
        xaxis_title='Дата', yaxis_title=y_title, hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0), margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig

def make_rv_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if 'log_RV' in df.columns:
        actual_df = df.dropna(subset=['date', 'log_RV'])
        fig.add_trace(go.Scatter(x=actual_df['date'], y=actual_df['log_RV'], mode='lines', name='Фактический ряд', line=dict(width=2.2)))
    if 'pred_log_RV_oos' in df.columns:
        pred_df = df.dropna(subset=['date', 'pred_log_RV_oos'])
        fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['pred_log_RV_oos'], mode='lines', name='Прогноз', line=dict(width=3, dash='dash')))
    fig.update_layout(
        template='plotly_dark', height=430, title='Реализованная волатильность: фактический и прогнозный ряды',
        xaxis_title='Дата', yaxis_title='log_RV', hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0), margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig

def make_error_gain_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    need = {'date', 'log_volume', 'pred_log_volume_base', 'pred_log_volume_full'}
    if not need.issubset(df.columns):
        fig.update_layout(template='plotly_dark', height=360, title='Нет данных для сравнения ошибок')
        return fig
    use = df.dropna(subset=list(need)).copy()
    if use.empty:
        fig.update_layout(template='plotly_dark', height=360, title='Нет данных для сравнения ошибок')
        return fig
    use['gain_full_vs_base'] = (use['log_volume'] - use['pred_log_volume_base']).abs() - (use['log_volume'] - use['pred_log_volume_full']).abs()
    fig.add_trace(go.Bar(x=use['date'], y=use['gain_full_vs_base'], name='Преимущество расширенной модели'))
    fig.add_hline(y=0, line_width=1, line_dash='dot')
    fig.update_layout(
        template='plotly_dark', height=360, title='Преимущество расширенной модели по абсолютной ошибке',
        xaxis_title='Дата', yaxis_title='Δ абсолютной ошибки', margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig

def make_regime_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if 'pred_activity_regime' not in df.columns:
        fig.update_layout(template='plotly_dark', height=320, title='Нет режима активности')
        return fig
    order = ['low', 'normal', 'high']
    counts = df['pred_activity_regime'].dropna().value_counts()
    counts = pd.Series({k: int(counts.get(k, 0)) for k in order})
    fig.add_trace(go.Bar(x=[REGIME_LABELS.get(k, k) for k in counts.index], y=counts.values))
    fig.update_layout(template='plotly_dark', height=320, title='Распределение прогнозных режимов активности', margin=dict(l=20, r=20, t=55, b=20))
    return fig



def make_threshold_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    need = {'date', 'log_volume', 'regime_q_low', 'regime_q_high'}
    if not need.issubset(df.columns):
        fig.update_layout(template='plotly_dark', height=360, title='Нет данных для порогов')
        return fig
    use = df.dropna(subset=list(need)).copy()
    if use.empty:
        fig.update_layout(template='plotly_dark', height=360, title='Нет данных для порогов')
        return fig
    fig.add_trace(go.Scatter(x=use['date'], y=use['log_volume'], mode='lines', name='Фактический ряд', line=dict(width=2.2)))
    fig.add_trace(go.Scatter(x=use['date'], y=use['regime_q_low'], mode='lines', name='Нижний порог', line=dict(width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=use['date'], y=use['regime_q_high'], mode='lines', name='Верхний порог', line=dict(width=2, dash='dot')))
    fig.update_layout(
        template='plotly_dark', height=360, title='Динамические пороги режима активности',
        xaxis_title='Дата', yaxis_title='log_volume', hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0), margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig



def summarize_scope(view_df: pd.DataFrame) -> tuple[str, str]:
    if view_df.empty or 'date' not in view_df.columns:
        return '—', '—'
    start = view_df['date'].min()
    end = view_df['date'].max()
    return (start.strftime('%Y-%m-%d') if pd.notna(start) else '—', end.strftime('%Y-%m-%d') if pd.notna(end) else '—')



def build_auto_summary(selected_period: str, full_block: dict[str, Any], base_block: dict[str, Any], last_row: pd.Series | None) -> str:
    period_label = PERIOD_LABELS.get(selected_period, selected_period)
    full_rmse = safe_float(full_block.get('rmse'))
    base_rmse = safe_float(base_block.get('rmse'))
    delta_rmse = base_rmse - full_rmse if not np.isnan(full_rmse) and not np.isnan(base_rmse) else np.nan
    oos_r2 = safe_float(full_block.get('oos_r2_vs_benchmark'))

    if np.isnan(delta_rmse):
        model_phrase = 'Сравнение расширенной и инерционной модели недоступно.'
    elif delta_rmse > 0.02:
        model_phrase = f'Расширенная модель заметно лучше инерционной: выигрыш по RMSE {delta_rmse:.4f}.'
    elif delta_rmse > 0:
        model_phrase = f'Расширенная модель умеренно лучше инерционной: выигрыш по RMSE {delta_rmse:.4f}.'
    elif delta_rmse == 0:
        model_phrase = 'Расширенная и инерционная модели дают одинаковый RMSE.'
    else:
        model_phrase = f'Расширенная модель уступает инерционной: проигрыш по RMSE {abs(delta_rmse):.4f}.'

    if np.isnan(oos_r2):
        benchmark_phrase = 'Сравнение с эталонным прогнозом недоступно.'
    elif oos_r2 > 0:
        benchmark_phrase = f'OOS R² относительно эталонного прогноза положителен: {oos_r2:.4f}.'
    else:
        benchmark_phrase = f'OOS R² относительно эталонного прогноза неположителен: {oos_r2:.4f}.'

    regime_phrase = ''
    if last_row is not None:
        regime = REGIME_LABELS.get(str(last_row.get('pred_activity_regime', '—')), '—')
        vol = fmt_int(last_row.get('pred_volume_full'))
        regime_phrase = f' Последний прогноз: ожидаемый объём {vol}, режим {regime}.'

    return f'{period_label}: {model_phrase} {benchmark_phrase}{regime_phrase}'



def build_metrics_table(selected_period: str, metrics: dict[str, Any]) -> pd.DataFrame:
    period = 'oos_all' if selected_period == 'full_history' else selected_period
    rows = []
    mapping = [
        ('Эталонный прогноз', 'volume_naive'),
        ('Инерционная модель', 'volume_base'),
        ('Расширенная модель', 'volume_full'),
    ]
    for label, key in mapping:
        block = get_metric_block(metrics, key, period)
        rows.append({
            'Модель': label,
            'MAE': safe_float(block.get('mae')),
            'RMSE': safe_float(block.get('rmse')),
            'OOS R² относительно эталона': safe_float(block.get('oos_r2_vs_benchmark')),
        })
    return pd.DataFrame(rows)



def rename_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'date': 'Дата',
        'period': 'Период',
        'log_RV': 'Фактический log_RV',
        'pred_log_RV_oos': 'Прогноз log_RV',
        'actual_volume': 'Фактический объём',
        'naive_volume': 'Эталонный прогноз объёма',
        'pred_volume_base': 'Прогноз объёма: инерционная модель',
        'pred_volume_full': 'Прогноз объёма: расширенная модель',
        'actual_activity_regime': 'Фактический режим',
        'pred_activity_regime': 'Прогнозный режим',
        'regime_hit': 'Совпадение режима',
    }
    out = df.copy()
    existing = {k: v for k, v in mapping.items() if k in out.columns}
    return out.rename(columns=existing)



@st.cache_data(show_spinner=False)
def load_segments_df(path: str) -> pd.DataFrame:
    segments = ensure_segments_file(path)
    rows = [
        {
            'Контракт': seg.security,
            'Начало': seg.start_date,
            'Конец': seg.end_date or 'открытый период',
        }
        for seg in segments
    ]
    return pd.DataFrame(rows)


def describe_source(selected_period: str) -> str:
    if selected_period == 'test':
        return 'Источник: публичный master sample dataset'
    if selected_period == 'control':
        return 'Источник: контрольный блок MOEX API'
    if selected_period == 'oos_all':
        return 'Источник: объединение тестового и контрольного периодов'
    return 'Источник: публичный сниппет 2023-2026 + доступные прогнозы вне обучающего периода'


def run_pipeline_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)


def refresh_all_cached_data() -> None:
    st.cache_data.clear()


def refresh_control_data(
    segments_path: Path,
    output_path: Path,
    rerun_pipeline_after_update: bool,
    pipeline_path: Path,
) -> tuple[bool, str]:
    cfg = UpdateConfig(
        output_path=str(output_path),
        segments_path=str(segments_path),
        candle_interval=10,
        engine='futures',
        market='forts',
        verbose=False,
    )
    segments = load_segments(str(segments_path))
    merged = update_control_daily_csv(segments=segments, cfg=cfg)

    message = f'Control-датасет обновлён: {len(merged)} строк.'
    if rerun_pipeline_after_update:
        result = run_pipeline_script(pipeline_path)
        if result.returncode != 0:
            error_text = result.stderr or result.stdout or 'Неизвестная ошибка пайплайна'
            return False, f'{message}\n\nПайплайн завершился с ошибкой:\n{error_text}'
        message += '\nПайплайн также успешно пересчитан.'

    refresh_all_cached_data()
    return True, message


st.set_page_config(page_title='RTS Activity', layout='wide')
st.title('RTS Market Activity')

with st.sidebar:
    st.header('Параметры')
    pipeline_path = st.text_input('Pipeline script', str(PIPELINE_SCRIPT))
    predictions_path = st.text_input('predictions.csv', str(PREDICTIONS_PATH))
    metrics_path = st.text_input('metrics.json', str(METRICS_PATH))
    master_path = st.text_input('master dataset', str(MASTER_DATA_PATH))

    rerun = st.button('Пересчитать прогнозы')
    only_oos = st.checkbox('Показывать только внешний период', value=True)
    use_log_scale = st.checkbox('График объёма в log-шкале', value=False)
    lookback_label = st.selectbox('Сколько последних наблюдений показывать', options=list(LOOKBACK_OPTIONS.keys()), index=5)
    lookback_rows = LOOKBACK_OPTIONS[lookback_label]

    st.divider()
    st.subheader('Новые периоды MOEX')
    st.caption('Добавьте новый контрактный период и подтяните свежие данные в control-блок прямо из дэшборда.')

    segments_path_str = st.text_input('segments.json', str(SEGMENTS_PATH))
    control_path_str = st.text_input('control dataset', str(CONTROL_DATA_PATH))
    rerun_after_update = st.checkbox('Пересчитать пайплайн после обновления периода', value=True)

    try:
        segments_df = load_segments_df(segments_path_str)
        if not segments_df.empty:
            st.dataframe(segments_df, use_container_width=True, hide_index=True)
        else:
            st.info('Контрактные сегменты пока не заданы.')
    except Exception as exc:
        st.warning(f'Не удалось прочитать список сегментов: {exc}')

    with st.form('add_moex_period_form'):
        security_code = st.text_input('Код контракта', value='', placeholder='Например, RIU6').strip().upper()
        start_dt = st.date_input('Дата начала', value=date(2026, 3, 27), format='YYYY-MM-DD')
        open_end = st.checkbox('Открытый период без даты окончания', value=True)
        end_dt = None
        if not open_end:
            end_dt = st.date_input('Дата окончания', value=date(2026, 6, 18), format='YYYY-MM-DD')
        add_period = st.form_submit_button('Добавить период и загрузить данные')

    if add_period:
        segments_path = Path(segments_path_str)
        control_path = Path(control_path_str)
        pipeline_script = Path(pipeline_path)
        if not security_code:
            st.error('Нужно указать код контракта.')
        elif not open_end and end_dt is not None and end_dt < start_dt:
            st.error('Дата окончания не может быть раньше даты начала.')
        elif rerun_after_update and not pipeline_script.exists():
            st.error(f'Pipeline script не найден: {pipeline_script}')
        else:
            try:
                ensure_segments_file(str(segments_path))
                append_segment(
                    str(segments_path),
                    security=security_code,
                    start_date=start_dt.isoformat(),
                    end_date=None if open_end or end_dt is None else end_dt.isoformat(),
                )
                with st.spinner('Загружаю новый период из MOEX ISS...'):
                    ok, message = refresh_control_data(
                        segments_path=segments_path,
                        output_path=control_path,
                        rerun_pipeline_after_update=rerun_after_update,
                        pipeline_path=pipeline_script,
                    )
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            except Exception as exc:
                st.error(f'Не удалось добавить период: {exc}')

    refresh_control_button = st.button('Обновить control-данные по текущим сегментам')
    if refresh_control_button:
        segments_path = Path(segments_path_str)
        control_path = Path(control_path_str)
        pipeline_script = Path(pipeline_path)
        if rerun_after_update and not pipeline_script.exists():
            st.error(f'Pipeline script не найден: {pipeline_script}')
        else:
            try:
                ensure_segments_file(str(segments_path))
                with st.spinner('Обновляю control-период...'):
                    ok, message = refresh_control_data(
                        segments_path=segments_path,
                        output_path=control_path,
                        rerun_pipeline_after_update=rerun_after_update,
                        pipeline_path=pipeline_script,
                    )
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            except Exception as exc:
                st.error(f'Не удалось обновить control-период: {exc}')

if rerun:
    script = Path(pipeline_path)
    if not script.exists():
        st.error(f'Pipeline script не найден: {script}')
    else:
        with st.spinner('Пересчитываю пайплайн...'):
            result = run_pipeline_script(script)
        if result.returncode == 0:
            st.success('Пайплайн успешно пересчитан')
            refresh_all_cached_data()
        else:
            st.error('Пайплайн завершился с ошибкой')
            st.code(result.stderr or result.stdout)

try:
    metrics = load_metrics(metrics_path)
    pred = load_predictions(predictions_path)
    master_ctx = load_master_context(master_path)
except Exception as e:
    st.error(f'Не удалось загрузить данные: {e}')
    st.stop()

if pred.empty and master_ctx.empty:
    st.error('Нет данных ни в predictions.csv, ни в master dataset')
    st.stop()

context_df = build_context_history(master_ctx, pred)
period_options = available_periods(pred, master_ctx)
selected_period = st.radio('Период анализа', options=period_options, format_func=lambda x: PERIOD_LABELS.get(x, x), horizontal=True)
view_df = build_view_df(pred, context_df, selected_period, only_oos=only_oos, lookback_rows=lookback_rows)
if view_df.empty:
    st.warning('Нет данных для выбранного режима')
    st.stop()

last_row = get_last_valid_row(pred, 'oos_all' if selected_period == 'full_history' else selected_period)
metric_period = 'oos_all' if selected_period == 'full_history' else selected_period
full_block = get_metric_block(metrics, 'volume_full', metric_period)
base_block = get_metric_block(metrics, 'volume_base', metric_period)
rv_block = get_metric_block(metrics, 'rv_model', metric_period)
regime_block = get_metric_block(metrics, 'activity_regime', metric_period)

scope_start, scope_end = summarize_scope(view_df)
delta_rmse = safe_float(base_block.get('rmse')) - safe_float(full_block.get('rmse'))
metrics_table = build_metrics_table(selected_period, metrics)

st.caption(f'{describe_source(selected_period)}')
st.caption(f'Диапазон дат: {scope_start} → {scope_end}  •  Показано наблюдений: {len(view_df)}')
st.info(build_auto_summary(selected_period, full_block, base_block, last_row), icon='📌')

cards = st.columns([1.0, 1.0, 1.0, 1.15, 1.0])
last_date_text = '—' if last_row is None or pd.isna(last_row.get('date')) else pd.to_datetime(last_row.get('date')).strftime('%Y-%m-%d')
cards[0].metric('Дата последнего прогноза', last_date_text)
if last_row is not None:
    cards[1].metric('Прогноз log_RV', fmt_num(last_row.get('pred_log_RV_oos'), 3))
    cards[2].metric('Ожидаемый объём', fmt_int(last_row.get('pred_volume_full')))
    regime = str(last_row.get('pred_activity_regime', '—'))
    cards[3].metric('Прогнозный режим активности', REGIME_LABELS.get(regime, regime))
else:
    cards[1].metric('Прогноз log_RV', '—')
    cards[2].metric('Ожидаемый объём', '—')
    cards[3].metric('Прогнозный режим активности', '—')
cards[4].metric('Наблюдений на экране', len(view_df))

model_cols = st.columns(4)
model_cols[0].metric('RMSE расширенной модели', fmt_num(full_block.get('rmse'), 4))
model_cols[1].metric('MAE расширенной модели', fmt_num(full_block.get('mae'), 4))
model_cols[2].metric('OOS R² относительно эталона', fmt_num(full_block.get('oos_r2_vs_benchmark'), 4))
if np.isnan(delta_rmse):
    model_cols[3].metric('Δ RMSE к инерционной модели', '—')
else:
    if delta_rmse > 0:
        model_cols[3].metric('Δ RMSE к инерционной модели', fmt_num(delta_rmse, 4), delta='лучше', delta_color='normal')
    elif delta_rmse < 0:
        model_cols[3].metric('Δ RMSE к инерционной модели', fmt_num(delta_rmse, 4), delta='хуже', delta_color='inverse')
    else:
        model_cols[3].metric('Δ RMSE к инерционной модели', fmt_num(delta_rmse, 4), delta='без изменений', delta_color='off')

comparison_expander = st.expander('Сводка по моделям', expanded=False)
with comparison_expander:
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)


tab_overview, tab_diag, tab_table, tab_model = st.tabs(['Обзор', 'Диагностика', 'Таблица', 'О модели'])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_volume_chart(view_df, use_log_scale=use_log_scale), use_container_width=True)
    with right:
        st.plotly_chart(make_rv_chart(view_df), use_container_width=True)

    sub_left, sub_right = st.columns(2)
    with sub_left:
        st.plotly_chart(make_error_gain_chart(view_df), use_container_width=True)
    with sub_right:
        st.plotly_chart(make_regime_distribution(view_df), use_container_width=True)

with tab_diag:
    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(make_threshold_chart(view_df), use_container_width=True)
    with top_right:
        st.write('**Краткая диагностика по выбранному периоду**')
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        st.write('**Прогноз волатильности**')
        st.json(rv_block)
        st.write('**Параметры классификации режима активности**')
        st.json(regime_block)

    if {'pred_log_volume_base', 'pred_log_volume_full', 'log_volume'}.issubset(view_df.columns):
        diag = view_df.dropna(subset=['pred_log_volume_base', 'pred_log_volume_full', 'log_volume']).copy()
        if not diag.empty:
            diag['abs_err_base_log'] = (diag['log_volume'] - diag['pred_log_volume_base']).abs()
            diag['abs_err_full_log'] = (diag['log_volume'] - diag['pred_log_volume_full']).abs()
            diag['full_better'] = diag['abs_err_full_log'] < diag['abs_err_base_log']
            st.metric('Доля дат, где расширенная модель лучше инерционной', fmt_num(diag['full_better'].mean(), 4))
            cols = [c for c in ['date', 'period', 'log_volume', 'pred_log_volume_base', 'pred_log_volume_full', 'abs_err_base_log', 'abs_err_full_log', 'full_better'] if c in diag.columns]
            diag = diag[cols].sort_values('date', ascending=False).copy()
            diag = diag.rename(columns={
                'date': 'Дата',
                'period': 'Период',
                'log_volume': 'Фактический log_volume',
                'pred_log_volume_base': 'Прогноз: инерционная модель',
                'pred_log_volume_full': 'Прогноз: расширенная модель',
                'abs_err_base_log': 'Абс. ошибка: инерционная модель',
                'abs_err_full_log': 'Абс. ошибка: расширенная модель',
                'full_better': 'Расширенная лучше',
            })
            st.dataframe(diag.head(40), use_container_width=True)

with tab_table:
    show_cols = [
        'date', 'period', 'log_RV', 'pred_log_RV_oos', 'actual_volume', 'naive_volume',
        'pred_volume_base', 'pred_volume_full', 'actual_activity_regime',
        'pred_activity_regime', 'regime_hit'
    ]
    existing_cols = [c for c in show_cols if c in view_df.columns]
    table_df = view_df[existing_cols].copy().sort_values('date', ascending=False)
    table_df = rename_table_columns(table_df)
    st.dataframe(table_df, use_container_width=True)
    st.download_button(
        'Скачать таблицу CSV',
        data=table_df.to_csv(index=False).encode('utf-8'),
        file_name=f'rts_dashboard_{selected_period}.csv',
        mime='text/csv',
    )

with tab_model:
    st.markdown('## О модели')
    st.markdown(
        """
Этот дэшборд показывает **прикладной индикатор ожидаемой рыночной активности** для фьючерса на индекс РТС. Он не пытается предсказывать направление цены и не является торговым сигналом. Его задача скромнее и полезнее: оценить, насколько активным может быть следующий торговый день с точки зрения **реализованной волатильности** и **объёма торгов**.
        """
    )

    with st.expander('1. Что делает модель', expanded=True):
        st.markdown(
            """
**Логика двухэтапная.**

**Этап 1. Прогноз волатильности**
- Модель прогнозирует `log_RV`, то есть логарифм дневной realized volatility.
- Входы: лаги `log_RV` с горизонтами 1, 2 и 5 дней.
- Выход: `pred_log_RV_oos`, то есть одношаговый прогноз волатильности.

**Этап 2. Прогноз объёма торгов**
- **Инерционная модель** прогнозирует `log_volume` только по лагам объёма.
- **Расширенная модель** использует те же лаги объёма и дополнительно прогнозную волатильность `pred_log_RV_oos`.
- **Эталонный прогноз** нужен как минимальная точка сравнения и по сути говорит: “завтра будет примерно как сегодня”.

Идея простая: если ожидаемая волатильность действительно несёт дополнительную информацию о режиме рынка, то **расширенная модель** должна быть хотя бы немного лучше **инерционной**.
            """
        )

    with st.expander('2. Какие данные используются'):
        st.markdown(
            """
**Историческая база**
- Основной master dataset в публичной версии урезан до минимально нужного sample-набора: `date`, `log_RV`, `volume_day`.
- Этот блок покрывает период 2023-2026 и используется как компактный публичный сниппет для обучения и теста.

**Внешний контрольный период**
- Данные 2026 года подтягиваются отдельно через MOEX ISS API.
- Они сохраняются как `moex_control_daily.csv` и затем подмешиваются в pipeline только как `control`.
- Это сделано специально, чтобы контрольный период не маскировался хвостом master dataset.

**Что важно помнить**
- Исторический блок и control-блок могут немного отличаться по способу формирования свечей и агрегации.
- Поэтому метрики на `control` нужно интерпретировать как внешнюю прикладную проверку, а не как идеально сопоставимый лабораторный тест.
            """
        )

    with st.expander('3. Как устроены периоды train / test / control'):
        st.markdown(
            """
Разбиение здесь **жёстко задано по времени**, а не случайным split. Для временных рядов это принципиально важно.

- **Train**: данные до конца 2024 года.
- **Тестовый период 2025**: полный 2025 год. Это основной честный тестовый интервал.
- **Контрольный период 2026**: внешний блок, который поступает через API биржи.
- **Весь внешний период**: объединение тестового и контрольного интервалов.
- **Весь сниппет 2023-2026**: публичный исторический контекст. Прогнозы там отображаются только там, где они реально есть.
            """
        )

    with st.expander('4. Что именно отображается на дэшборде'):
        st.markdown(
            """
**Верхние карточки**
- Дата последнего прогноза
- Прогноз `log_RV`
- Ожидаемый объём
- Прогнозный режим активности
- Размер текущей выборки на экране

**Метрики модели**
- `RMSE расширенной модели`
- `MAE расширенной модели`
- `OOS R² относительно эталона`
- `Δ RMSE к инерционной модели`

**Графики**
- фактический и прогнозные ряды по объёму,
- фактический и прогнозный ряд по волатильности,
- преимущество расширенной модели по ошибке,
- распределение прогнозных режимов активности.
            """
        )

    with st.expander('5. Как читать метрики качества'):
        st.markdown(
            """
**MAE**
- Средняя абсолютная ошибка прогноза.
- Чем ниже, тем лучше.

**RMSE**
- Корень из средней квадратичной ошибки.
- Чем ниже, тем лучше.
- Сильнее штрафует крупные промахи.

**OOS R² относительно эталона**
- Это out-of-sample R² относительно эталонного прогноза.
- Если значение положительное, модель лучше самого простого ориентира.

**Δ RMSE к инерционной модели**
- Показывает, насколько расширенная модель лучше или хуже инерционной.
- Положительное значение означает, что добавление прогнозной волатильности действительно улучшило результат.
            """
        )

    with st.expander('6. Что такое режим активности и зачем он нужен'):
        st.markdown(
            """
Режим активности — это **интерпретационный слой поверх прогноза объёма**, а не отдельная основная модель.

Используются три категории:
- **низкая активность**,
- **нормальная активность**,
- **повышенная активность**.

Границы строятся динамически через rolling quantiles, то есть режим оценивается **относительно недавней истории**, а не относительно всего публичного сниппета сразу.

**Почему точность режима вторична**
- Она полезна для удобства аналитика.
- Но сильно зависит от окна и порогов.
- Поэтому для оценки модели важнее `MAE`, `RMSE` и `OOS R²`.
            """
        )

    with st.expander('7. Ограничения текущего прототипа'):
        st.markdown(
            """
1. Модели здесь намеренно простые и интерпретируемые.
2. В текущей версии не используются макрофакторы.
3. Контрольный период 2026 пока заметно короче тестового интервала 2025.
4. Историческая и API-часть данных могут отличаться по техническому способу формирования.
5. Дэшборд — это прототип аналитического инструмента, а не торговый сигнал.
            """
        )

    with st.expander('8. Практический смысл для аналитики'):
        st.markdown(
            """
Этот прототип полезен как **индикатор ожидаемого режима активности**.

Что он может дать аналитически:
- быструю оценку, будет ли день спокойным или напряжённым,
- дополнительный сигнал к интерпретации ожидаемого объёма,
- компактный инструмент мониторинга для срочного рынка,
- понятную витрину для сравнения простой и расширенной модели.
            """
        )
