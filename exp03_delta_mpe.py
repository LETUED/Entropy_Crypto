"""
Delta MPE 전략 백테스트 실행

목적:
    MPE 레벨(이미 낮음) 대신 MPE 기울기(빠르게 낮아짐)를 신호로 사용
    → 레짐 전환 순간을 포착 → 신호 빈도 ↑

근거:
    Thermodynamic Analysis (2023) — Delta Entropy가 분산의 41~57% 설명
    기존 MPE level 전략: 4년 16번 진입
    목표: 의미있는 신호를 유지하면서 진입 횟수 증가

실행:
    py run_delta_mpe.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe, rolling_delta_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.delta_mpe_backtest import run_delta_analysis, plot_delta_results

START, END = "2021-01-01", "2025-01-01"


def main():
    print("=" * 65)
    print("Delta MPE 전략 백테스트")
    print(f"  기간: {START} ~ {END}")
    print(f"  신호: MPE 기울기 하위 15% + RSI < 30")
    print(f"  청산: RSI > 50 또는 최대 168H")
    print("=" * 65)

    # ── 데이터 수집 ─────────────────────────────────────────────────────────
    print("\n[1/4] 데이터 수집 중...")
    df      = collect("BTCUSDT", "1h", START, END)
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    # ── MPE + Delta MPE 계산 ────────────────────────────────────────────────
    print("\n[2/4] MPE 및 Delta MPE 계산 중... (수분 소요)")
    mpe       = rolling_mpe(df["close"], window=168)
    delta_mpe = rolling_delta_mpe(mpe, smooth_window=20)

    print(f"  MPE 완료: {mpe.dropna().shape[0]}개 유효값")
    print(f"  Delta MPE 완료: {delta_mpe.dropna().shape[0]}개 유효값")

    # ── 온체인 엔트로피 ──────────────────────────────────────────────────────
    print("\n[3/4] 온체인 엔트로피 계산 중...")
    h_funding = funding_entropy(funding, df.index)
    h_fg      = fear_greed_entropy(fg, df.index)
    h_onchain = combined_onchain_entropy(h_funding, h_fg)

    # ── 전략 실행 및 비교 ────────────────────────────────────────────────────
    print("\n[4/4] 전략 비교 실행 중...")
    results, counts = run_delta_analysis(df, mpe, delta_mpe, h_onchain)

    # ── 시각화 ──────────────────────────────────────────────────────────────
    plot_delta_results(results, counts, delta_mpe)

    print("\n완료: results/delta_mpe.png")
    print("\n해석 기준:")
    print("  신호 빈도  : 기존 16번 대비 Delta 전략이 몇 배 많은가")
    print("  Sharpe > 0 : 신호가 무작위보다 나은가")
    print("  MA200 추가 시 Sharpe 향상 여부 확인")


if __name__ == "__main__":
    main()
