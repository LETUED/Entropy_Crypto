"""
Binance 공개 API에서 OHLCV 데이터 수집 (API 키 불필요)
"""

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BINANCE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def collect(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2021-01-01",
    end: str = "2025-01-01",
    force: bool = False,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{symbol}_{interval}_{start}_{end}.parquet"

    if cache_path.exists() and not force:
        print(f"캐시 로드: {cache_path.name}")
        return pd.read_parquet(cache_path)

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    interval_ms = _interval_to_ms(interval)

    total_candles = (end_ms - start_ms) // interval_ms
    total_batches = (total_candles // 1000) + 1

    all_rows = []
    cursor = start_ms

    with tqdm(total=total_batches, desc=f"{symbol} {interval} 수집") as pbar:
        while cursor < end_ms:
            batch = fetch_klines(symbol, interval, cursor, end_ms)
            if not batch:
                break
            all_rows.extend(batch)
            cursor = batch[-1][0] + interval_ms
            pbar.update(1)
            time.sleep(0.1)  # rate limit 방지

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    df.to_parquet(cache_path)
    print(f"저장 완료: {cache_path.name} ({len(df):,}개 캔들)")
    return df


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    mapping = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * mapping[unit]
