"""
온체인/시장 심리 데이터 수집 (API 키 불필요)

1. 펀딩비 (Binance Futures) — 8시간마다 갱신
   양수 = 롱 과열, 음수 = 숏 과열, 0 근처 = 중립
2. 공포탐욕지수 (alternative.me) — 일별
   0 = 극단적 공포, 100 = 극단적 탐욕
"""

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1/fundingRate"
FEAR_GREED_URL  = "https://api.alternative.me/fng/"


# ── 펀딩비 ───────────────────────────────────────────────────────────────────

def collect_funding_rate(
    symbol: str = "BTCUSDT",
    start: str = "2021-01-01",
    end: str = "2025-01-01",
    force: bool = False,
) -> pd.Series:
    cache = DATA_DIR / f"funding_{symbol}_{start}_{end}.parquet"
    if cache.exists() and not force:
        print(f"캐시 로드: {cache.name}")
        return pd.read_parquet(cache)["fundingRate"]

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end).timestamp() * 1000)
    interval_ms = 8 * 3600 * 1000  # 8시간

    rows, cursor = [], start_ms
    total = (end_ms - start_ms) // (interval_ms * 1000) + 1

    with tqdm(desc="펀딩비 수집", total=total) as pbar:
        while cursor < end_ms:
            resp = requests.get(BINANCE_FUTURES, params={
                "symbol": symbol, "startTime": cursor,
                "endTime": end_ms, "limit": 1000,
            }, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1]["fundingTime"] + interval_ms
            pbar.update(len(batch))
            time.sleep(0.1)

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df = df.set_index("fundingTime")[["fundingRate"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"펀딩비 저장: {cache.name} ({len(df):,}개)")
    return df["fundingRate"]


# ── 공포탐욕지수 ─────────────────────────────────────────────────────────────

def collect_fear_greed(
    start: str = "2021-01-01",
    end: str = "2025-01-01",
    force: bool = False,
) -> pd.Series:
    cache = DATA_DIR / f"fear_greed_{start}_{end}.parquet"
    if cache.exists() and not force:
        print(f"캐시 로드: {cache.name}")
        return pd.read_parquet(cache)["value"]

    # 최대 2000일치 요청
    resp = requests.get(FEAR_GREED_URL, params={"limit": 2000, "format": "json"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df = df.set_index("timestamp")[["value"]].astype(float)
    df = df.sort_index()

    # 기간 필터
    df = df[start:end]

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"공포탐욕 저장: {cache.name} ({len(df):,}일)")
    return df["value"]
