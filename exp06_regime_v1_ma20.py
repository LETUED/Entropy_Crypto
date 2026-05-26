"""
엔트로피 레짐 복합 전략 실행

저엔트로피 -> 역추세 / 고엔트로피 -> 모멘텀 / 중간 -> 관망

실행: py run_regime_strategy.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.regime_strategy import run_comparison, plot_regime_results

START, END = "2021-01-01", "2025-01-01"


def main():
    print("[1/3] 데이터 수집...")
    df      = collect("BTCUSDT", "1h", START, END)
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("[2/3] MPE 계산... (수분 소요)")
    mpe = rolling_mpe(df["close"], window=168)
    print(f"  완료: {mpe.dropna().shape[0]}개 유효값")

    print("[3/3] 온체인 엔트로피 계산...")
    h_onchain = combined_onchain_entropy(
        funding_entropy(funding, df.index),
        fear_greed_entropy(fg, df.index),
    )

    results = run_comparison(df, mpe, h_onchain)
    plot_regime_results(results, df, mpe)


if __name__ == "__main__":
    main()
