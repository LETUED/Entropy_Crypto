"""
전체 코인 데이터 수집 + MPE 사전 계산 (33코인 / 2021~2026)
멀티프로세싱 병렬 처리

실행: py prefetch_all.py
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

START = "2021-01-01"
END   = "2026-05-26"

# 8개 카테고리 전체 코인
CATEGORIES = {
    "가치저장":        ["BTCUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT"],
    "스마트컨트랙트L1": ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
                        "ATOMUSDT", "NEARUSDT", "FTMUSDT", "MATICUSDT"],
    "DeFi":           ["LINKUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "1INCHUSDT"],
    "거래소토큰":      ["BNBUSDT", "CROUSDT"],
    "밈":             ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"],
    "게임메타버스":    ["AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT"],
    "인프라":         ["FILUSDT", "GRTUSDT", "RNDRUSDT"],
    "결제크로스체인":  ["XRPUSDT", "XLMUSDT", "ALGOUSDT"],
}

ALL_COINS = [sym for coins in CATEGORIES.values() for sym in coins]


def fetch_one(sym: str):
    t0 = time.time()
    try:
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168,
                          cache_key=f"{sym}_1h_{START}_{END}")
        elapsed = time.time() - t0
        print(f"  [{sym:12s}] OK  {len(df):>6,}봉  MPE {len(mpe.dropna()):>6,}포인트  ({elapsed:.0f}s)", flush=True)
        return (sym, True, None)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{sym:12s}] FAIL ({elapsed:.0f}s): {e}", flush=True)
        return (sym, False, str(e))


if __name__ == "__main__":
    n_cores = min(mp.cpu_count(), len(ALL_COINS))
    print(f"데이터 수집 + MPE 계산 시작")
    print(f"  코인: {len(ALL_COINS)}개  |  코어: {n_cores}개  |  기간: {START} ~ {END}")
    print("-" * 60)

    t_start = time.time()
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(fetch_one, ALL_COINS)

    ok    = [r for r in results if r[1]]
    fail  = [r for r in results if not r[1]]
    elapsed = time.time() - t_start

    print(f"\n완료: {len(ok)}개 성공 / {len(fail)}개 실패  ({elapsed:.0f}s)")
    if fail:
        print("실패 목록:")
        for sym, _, err in fail:
            print(f"  {sym}: {err}")
