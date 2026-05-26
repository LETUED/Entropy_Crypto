"""
MPE 사전 계산 + 캐시 저장 (멀티프로세싱)

10코인을 CPU 코어 수만큼 병렬 계산 → 20분 → ~2분
캐시 저장 후 모든 exp 스크립트가 즉시 로드.

실행: py prefetch_mpe.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-", "") not in ("utf8", "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import time
import multiprocessing as mp

from src.data.binance_collector import collect
from src.entropy.calculators import rolling_mpe

START, END = "2021-01-01", "2025-01-01"

COINS = [
    "BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
    "ETHUSDT",  "BNBUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT",
]


def compute_one(sym: str):
    t0 = time.time()
    df  = collect(sym, "1h", START, END)
    cache_key = f"{sym}_1h_{START}_{END}"
    mpe = rolling_mpe(df["close"], window=168, cache_key=cache_key)
    elapsed = time.time() - t0
    print(f"  [{sym}] 완료 ({len(mpe)}포인트, {elapsed:.0f}초)", flush=True)
    return sym


if __name__ == "__main__":
    n_cores = min(mp.cpu_count(), len(COINS))
    print(f"MPE 사전 계산 시작 ({len(COINS)}코인 / {n_cores}코어 병렬)")
    print(f"기간: {START} ~ {END}  |  window=168, m=3, tau=1, scales=[1,2,4,8]")
    print("-" * 50)

    t_start = time.time()
    with mp.Pool(processes=n_cores) as pool:
        pool.map(compute_one, COINS)

    elapsed = time.time() - t_start
    print(f"\n완료 — 총 {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print("data/cache/ 에 MPE 캐시 저장됨. 이후 exp 스크립트는 즉시 로드.")
