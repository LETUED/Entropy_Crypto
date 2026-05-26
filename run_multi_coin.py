"""
멀티 코인 실행 스크립트

실행:
    py run_multi_coin.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.multi_coin import run_multi_coin, plot_multi_coin

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
START, END = "2021-01-01", "2025-01-01"


def main():
    print(f"코인: {COINS}\n")

    # ── 공통 데이터 ───────────────────────────────────────────────────────
    print("공포탐욕지수 로드...")
    fg = collect_fear_greed(START, END)

    # ── 코인별 데이터 준비 ────────────────────────────────────────────────
    coin_data = {}
    bah_data  = {}

    for sym in COINS:
        print(f"\n[{sym}] 데이터 준비 중...")
        df      = collect(sym, "1h", START, END)
        funding = collect_funding_rate(sym, START, END)

        print(f"[{sym}] MPE 계산...")
        mpe = rolling_mpe(df["close"], window=168)

        print(f"[{sym}] 온체인 엔트로피 계산...")
        h_fund   = funding_entropy(funding, df.index)
        h_fg     = fear_greed_entropy(fg, df.index)
        h_onchain = combined_onchain_entropy(h_fund, h_fg)

        coin_data[sym] = {
            "df":       df,
            "mpe":      mpe,
            "h_onchain": h_onchain,
        }
        bah_data[sym] = df

    # ── 멀티 코인 전략 실행 ───────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("멀티 코인 전략 실행")
    print("━" * 60)
    results = run_multi_coin(coin_data)

    # ── 시각화 ───────────────────────────────────────────────────────────
    plot_multi_coin(results, bah_data)

    print("\n완료: results/multi_coin.png")


if __name__ == "__main__":
    main()
