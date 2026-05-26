"""
Phase 1 + Phase 2 전체 실행

실행:
    py run_phase1.py            # 숏 없음 (국내 거래소)
    py run_phase1.py --short    # 숏 포함 (바이낸스 선물)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h2_validation import validate_h2
from src.analysis.h3_validation import validate_h3, plot_h3_results
from src.analysis.h4_backtest import validate_h4, plot_h4_results
from src.analysis.performance_analysis import analyze, plot_performance, print_summary
from src.analysis.visualizer import plot_h2_results, plot_distribution_comparison, plot_summary_table

ALLOW_SHORT = "--short" in sys.argv
START, END  = "2021-01-01", "2025-01-01"


def main():
    print(f"실행 모드: {'숏 포함 (바이낸스 선물)' if ALLOW_SHORT else '롱 전용 (국내 거래소)'}\n")

    # ── 데이터 수집 ───────────────────────────────────────────────────────
    df      = collect("BTCUSDT", "1h", START, END)
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    # ── MPE 한 번만 계산 ──────────────────────────────────────────────────
    print("\nMPE 계산 중... (최초 1회만)")
    mpe = rolling_mpe(df["close"], window=168)

    # ── Phase 2: 온체인 엔트로피 계산 ────────────────────────────────────
    print("온체인 엔트로피 계산 중...")
    h_funding  = funding_entropy(funding, df.index)
    h_fg       = fear_greed_entropy(fg, df.index)
    h_onchain  = combined_onchain_entropy(h_funding, h_fg)

    # ── H2 ───────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("PHASE 1-A: H2 가설 검증")
    print("━" * 60)
    h2_results = validate_h2(df, mpe=mpe)
    plot_summary_table(h2_results)
    plot_h2_results(h2_results)
    plot_distribution_comparison(h2_results)

    # ── H3 ───────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("PHASE 1-B: H3 가설 검증")
    print("━" * 60)
    h3_results = validate_h3(df, mpe=mpe)
    plot_h3_results(h3_results)

    # ── H4: Phase 1 + Phase 2 비교 ───────────────────────────────────────
    print("\n" + "━" * 60)
    print("PHASE 1-C + 2: H4 백테스팅 (온체인 필터 포함)")
    print("━" * 60)
    h4_results = validate_h4(df, mpe, allow_short=ALLOW_SHORT, h_onchain=h_onchain)
    plot_h4_results(h4_results)

    # ── 성과 분석 ─────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("PHASE 1-D: 안정성 중심 성과 분석")
    print("━" * 60)
    perf = analyze(h4_results)
    print_summary(perf)
    plot_performance(perf)

    print("\n모든 결과 저장 완료: results/")


if __name__ == "__main__":
    main()
