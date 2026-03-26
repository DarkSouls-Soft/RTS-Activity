from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from moex_api import MoexISSClient, MoexAPIError

BASE_DIR = Path(__file__).resolve().parent

SUPPORTED_ISS_CANDLE_INTERVALS = {1, 10, 60, 24, 7, 31, 4}
REQUIRED_CANDLE_COLS = {'begin', 'close', 'volume'}


@dataclass(frozen=True)
class ContractSegment:
    security: str
    start_date: str
    end_date: str | None = None


@dataclass
class UpdateConfig:
    output_path: str = str(BASE_DIR / 'moex_control_daily.csv')
    segments_path: str = str(BASE_DIR / 'moex_contract_segments.json')
    candle_interval: int = 10
    engine: str = 'futures'
    market: str = 'forts'
    verbose: bool = True


class UpdateDataError(RuntimeError):
    pass


def log(msg: str, cfg: UpdateConfig) -> None:
    if cfg.verbose:
        print(msg)


DEFAULT_SEGMENTS = [
    ContractSegment(security='RIH6', start_date='2026-01-01', end_date='2026-03-19'),
    ContractSegment(security='RIM6', start_date='2026-03-20', end_date=None),
]



def validate_update_config(cfg: UpdateConfig) -> None:
    if cfg.candle_interval not in SUPPORTED_ISS_CANDLE_INTERVALS:
        raise UpdateDataError(
            'ISS candle interval looks unsupported. '
            f'Got {cfg.candle_interval}. Supported examples for ISS candles are 1, 10, 60, 24, 7, 31, 4.'
        )



def validate_candles(df: pd.DataFrame) -> None:
    missing = REQUIRED_CANDLE_COLS - set(df.columns)
    if missing:
        raise UpdateDataError(f'Missing required candle columns: {sorted(missing)}')



def ensure_segments_file(path: str) -> list[ContractSegment]:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump([asdict(x) for x in DEFAULT_SEGMENTS], f, ensure_ascii=False, indent=2)
        return list(DEFAULT_SEGMENTS)
    return load_segments(path)



def load_segments(path: str) -> list[ContractSegment]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise UpdateDataError('segments JSON must contain a list of contract segments.')
    segments: list[ContractSegment] = []
    for item in raw:
        segments.append(
            ContractSegment(
                security=str(item['security']).strip(),
                start_date=str(item['start_date']).strip(),
                end_date=None if item.get('end_date') in (None, '', 'null') else str(item.get('end_date')).strip(),
            )
        )
    return segments



def save_segments(path: str, segments: Iterable[ContractSegment]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump([asdict(x) for x in segments], f, ensure_ascii=False, indent=2)



def append_segment(path: str, security: str, start_date: str, end_date: str | None = None) -> list[ContractSegment]:
    segments = ensure_segments_file(path)
    if any(seg.security == security for seg in segments):
        return segments
    segments.append(ContractSegment(security=security, start_date=start_date, end_date=end_date))
    segments = sorted(segments, key=lambda x: x.start_date)
    save_segments(path, segments)
    return segments



def compute_daily_features_from_candles(candles_df: pd.DataFrame, security: str) -> pd.DataFrame:
    validate_candles(candles_df)
    out = candles_df.copy().sort_values('begin').reset_index(drop=True)
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out['volume'] = pd.to_numeric(out['volume'], errors='coerce')
    out = out.dropna(subset=['begin', 'close', 'volume'])
    out['date'] = pd.to_datetime(out['begin']).dt.floor('D')
    out['log_close'] = np.log(out['close'])
    out['log_ret'] = out.groupby('date')['log_close'].diff()
    daily = (
        out.groupby('date', as_index=False)
        .agg(
            RV=('log_ret', lambda x: float(np.nansum(np.square(x)))),
            bars=('close', 'size'),
            volume_day=('volume', 'sum'),
        )
        .sort_values('date')
        .reset_index(drop=True)
    )
    daily['log_RV'] = np.log(daily['RV'].replace(0, np.nan))
    daily['security'] = security
    return daily



def fetch_segment_daily_features(client: MoexISSClient, segment: ContractSegment, cfg: UpdateConfig) -> pd.DataFrame:
    end_date = segment.end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    log(f'\nFetching {segment.security}: {segment.start_date} -> {end_date}, interval={cfg.candle_interval}', cfg)

    try:
        exists = client.security_exists(segment.security, engine=cfg.engine, market=cfg.market)
    except Exception as exc:
        exists = None
        log(f'security_exists check failed: {exc}', cfg)
    log(f'security_exists={exists}', cfg)

    candles = client.fetch_all_candles(
        security=segment.security,
        start_date=segment.start_date,
        end_date=end_date,
        interval=cfg.candle_interval,
        engine=cfg.engine,
        market=cfg.market,
    )
    log(f'received candle rows={len(candles)}', cfg)

    if candles.empty:
        log(f'WARNING: empty candle response for {segment.security}', cfg)
        return pd.DataFrame(columns=['date', 'RV', 'bars', 'volume_day', 'log_RV', 'security'])

    daily = compute_daily_features_from_candles(candles, security=segment.security)
    log(f'daily rows after aggregation={len(daily)}', cfg)
    return daily



def build_control_daily_dataset(segments: Iterable[ContractSegment], cfg: UpdateConfig | None = None) -> pd.DataFrame:
    cfg = cfg or UpdateConfig()
    validate_update_config(cfg)
    client = MoexISSClient()

    parts: list[pd.DataFrame] = []
    for segment in segments:
        try:
            part = fetch_segment_daily_features(client=client, segment=segment, cfg=cfg)
        except (MoexAPIError, UpdateDataError, Exception) as exc:
            log(f'ERROR for {segment.security}: {exc}', cfg)
            part = pd.DataFrame(columns=['date', 'RV', 'bars', 'volume_day', 'log_RV', 'security'])
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=['date', 'RV', 'bars', 'volume_day', 'log_RV', 'security'])

    out = pd.concat(parts, ignore_index=True)
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['date', 'security']).drop_duplicates(subset=['date'], keep='last')
    return out.reset_index(drop=True)



def update_control_daily_csv(segments: Iterable[ContractSegment], cfg: UpdateConfig | None = None) -> pd.DataFrame:
    cfg = cfg or UpdateConfig()
    path = Path(cfg.output_path)
    new_df = build_control_daily_dataset(segments=segments, cfg=cfg)
    log(f'new rows from ISS={len(new_df)}', cfg)

    if path.exists():
        old_df = pd.read_csv(path)
        if not old_df.empty and 'date' in old_df.columns:
            old_df['date'] = pd.to_datetime(old_df['date'])
        merged = pd.concat([old_df, new_df], ignore_index=True)
        if not merged.empty:
            merged['date'] = pd.to_datetime(merged['date'])
            merged = merged.sort_values(['date', 'security']).drop_duplicates(subset=['date'], keep='last')
    else:
        merged = new_df.copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    log(f'Saved {len(merged)} rows to control dataset: {path}', cfg)
    return merged


if __name__ == '__main__':
    cfg = UpdateConfig(
        output_path=str(BASE_DIR / 'moex_control_daily.csv'),
        segments_path=str(BASE_DIR / 'moex_contract_segments.json'),
        candle_interval=10,
        engine='futures',
        market='forts',
        verbose=True,
    )

    segments = ensure_segments_file(cfg.segments_path)
    update_control_daily_csv(segments=segments, cfg=cfg)
