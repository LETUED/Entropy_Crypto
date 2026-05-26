"""
라이브 데이터 수집.

Binance REST API로 최근 WARMUP_BARS개 1H 캔들 수집.
온체인 (펀딩비, 공포탐욕) 수집.
"""

import numpy as np
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone

from live.config import API_KEY, API_SECRET, WARMUP_BARS, INTERVAL
from live.logger_setup import get_logger

log = get_logger()


def _client() -> Client:
    return Client(API_KEY, API_SECRET)


def fetch_ohlcv(sym: str, n_bars: int = WARMUP_BARS) -> pd.DataFrame:
    """최근 n_bars개 1H 캔들 반환."""
    client = _client()
    raw = client.get_klines(
        symbol=sym,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=n_bars,
    )
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    df = df.astype(float)
    return df


def fetch_funding_rates(sym: str = "BTCUSDT", limit: int = 500) -> pd.Series:
    """
    BTC 펀딩비 수집 (선물 API).
    8시간마다 발표 → 1H 인덱스로 forward-fill.
    """
    client = _client()
    try:
        raw = client.futures_funding_rate(symbol=sym, limit=limit)
        if not raw:
            return pd.Series(dtype=float)
        df = pd.DataFrame(raw)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        s = df.set_index("fundingTime")["fundingRate"]
        return s.sort_index()
    except Exception as e:
        log.warning(f"펀딩비 수집 실패: {e}")
        return pd.Series(dtype=float)


def fetch_fear_greed() -> pd.Series:
    """
    Alternative.me 공포탐욕 지수 수집.
    일별 데이터 → 1H 인덱스로 forward-fill.
    """
    import urllib.request
    import json
    try:
        url = "https://api.alternative.me/fng/?limit=100&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"] = df["value"].astype(int)
        s = df.set_index("timestamp")["value"].sort_index()
        return s
    except Exception as e:
        log.warning(f"공포탐욕 수집 실패: {e}")
        return pd.Series(dtype=float)
