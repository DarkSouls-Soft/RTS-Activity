from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


class MoexAPIError(RuntimeError):
    """Raised when ISS API returns an unexpected response."""


@dataclass
class MoexISSConfig:
    base_url: str = "https://iss.moex.com/iss"
    timeout: int = 30
    user_agent: str = "rts-activity-prototype/0.2"
    engine: str = "futures"
    market: str = "forts"
    default_page_size: int = 500


class MoexISSClient:
    """Small ISS client with explicit diagnostics."""

    def __init__(self, cfg: MoexISSConfig | None = None) -> None:
        self.cfg = cfg or MoexISSConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.user_agent})

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.cfg.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.cfg.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise MoexAPIError(f"ISS request failed: {resp.url} -> HTTP {resp.status_code}") from exc

        payload = resp.json()
        if not isinstance(payload, dict):
            raise MoexAPIError(f"Unexpected ISS response format for {resp.url}")
        return payload

    @staticmethod
    def _block_to_df(payload: dict[str, Any], block_name: str) -> pd.DataFrame:
        block = payload.get(block_name)
        if not isinstance(block, dict) or "columns" not in block or "data" not in block:
            raise MoexAPIError(f"ISS block '{block_name}' is missing or malformed.")
        return pd.DataFrame(block["data"], columns=block["columns"])

    def fetch_security_listing(
        self,
        security: str,
        engine: str | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        engine = engine or self.cfg.engine
        market = market or self.cfg.market
        payload = self._get_json(
            f"engines/{engine}/markets/{market}/securities/{security}.json",
            params={"iss.meta": "off"},
        )
        return self._block_to_df(payload, "securities")

    def security_exists(
        self,
        security: str,
        engine: str | None = None,
        market: str | None = None,
    ) -> bool:
        try:
            df = self.fetch_security_listing(security=security, engine=engine, market=market)
        except Exception:
            return False
        return not df.empty

    def fetch_candles_page(
        self,
        security: str,
        start_date: str,
        end_date: str,
        interval: int,
        start: int = 0,
        engine: str | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        engine = engine or self.cfg.engine
        market = market or self.cfg.market
        payload = self._get_json(
            f"engines/{engine}/markets/{market}/securities/{security}/candles.json",
            params={
                "from": start_date,
                "till": end_date,
                "interval": interval,
                "start": start,
                "iss.meta": "off",
                "iss.only": "candles",
            },
        )
        return self._block_to_df(payload, "candles")

    def fetch_all_candles(
        self,
        security: str,
        start_date: str,
        end_date: str,
        interval: int,
        page_size: int | None = None,
        engine: str | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        page_size = page_size or self.cfg.default_page_size
        chunks: list[pd.DataFrame] = []
        start = 0

        while True:
            chunk = self.fetch_candles_page(
                security=security,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                start=start,
                engine=engine,
                market=market,
            )
            if chunk.empty:
                break
            chunks.append(chunk)
            if len(chunk) < page_size:
                break
            start += len(chunk)

        if not chunks:
            return pd.DataFrame()

        out = pd.concat(chunks, ignore_index=True)
        for col in ("begin", "end"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col])
        return out
