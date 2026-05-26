"""
Walk-forward 검증 실행

목적:
    "공식이 과거에도 미래에도 일정하게 통하는가?"
    학습 기간(12개월)에서 임계값을 계산하고
    테스트 기간(6개월)에서 그 임계값으로 전략을 실행 (out-of-sample)
    → 모든 구간에서 Sharpe가 일관되면 공식이 타임리스함을 증명

실행:
    py run_walkforward.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.walkforward import run_walkforward, print_walkforward, plot_walkforward
from src.analysis.signal_quality import analyze_signal_quality, print_signal_quality, plot_signal_quality
from src.analysis.filter_diagnosis import diagnose_filters, print_diagnosis, plot_diagnosis

START, END = "2021-01-01", "2025-01-01"

# 학습 12개월 → 테스트 6개월 → 6개월씩 이동
# 결과: 2022~2024 구간을 6개 윈도우로 커버
TRAIN_MONTHS = 12
TEST_MONTHS  = 6
STEP_MONTHS  = 6


def main():
    print("=" * 65)
    print("Walk-forward 검증")
    print(f"  학습 {TRAIN_MONTHS}개월 → 테스트 {TEST_MONTHS}개월 → {STEP_MONTHS}개월씩 이동")
    print(f"  기간: {START} ~ {END}")
    print("=" * 65)

    # ── 데이터 수집 ───────────────────────────────────────────────────────
    print("\n[1/4] 데이터 수집 중...")
    df      = collect("BTCUSDT", "1h", START, END)
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    # ── MPE 계산 (전체 기간 1회만) ───────────────────────────────────────
    print("\n[2/4] MPE 계산 중... (수분 소요)")
    mpe = rolling_mpe(df["close"], window=168)

    # ── 온체인 엔트로피 ───────────────────────────────────────────────────
    print("\n[3/4] 온체인 엔트로피 계산 중...")
    h_funding = funding_entropy(funding, df.index)
    h_fg      = fear_greed_entropy(fg, df.index)
    h_onchain = combined_onchain_entropy(h_funding, h_fg)

    # ── Walk-forward 실행 ─────────────────────────────────────────────────
    print("\n[4/5] Walk-forward 실행 중...")
    print("  (각 윈도우: 학습 기간 임계값 → 테스트 기간 out-of-sample 적용)")
    print()

    wf_results = run_walkforward(
        df, mpe, h_onchain,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    print_walkforward(wf_results)
    plot_walkforward(wf_results)

    # ── 신호 품질 분석 ────────────────────────────────────────────────────
    print("\n" + "━" * 65)
    print("[5/5] 엔트로피 신호 품질 분석 중...")
    print("  (리스크 관리 제거 — 순수 신호 정확도 측정)")
    print("━" * 65)

    sq_results = analyze_signal_quality(
        df, mpe, h_onchain,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    print_signal_quality(sq_results)
    plot_signal_quality(sq_results)

    # ── 필터 통과율 진단 ──────────────────────────────────────────────────
    print("\n" + "━" * 65)
    print("필터 통과율 진단 중...")
    print("━" * 65)

    periods, threshold, oc_thresh = diagnose_filters(df, mpe, h_onchain)
    print_diagnosis(periods, threshold, oc_thresh)
    plot_diagnosis(periods)

    print("\n완료: results/walkforward.png  |  results/signal_quality.png  |  results/filter_diagnosis.png")
    print("\n판단 기준:")
    print("  [Walk-forward]  모든 구간 Sharpe > 0       → 공식이 타임리스")
    print("  [신호 품질]     승률 > 랜덤, 모든 구간 일관 → 신호 자체가 살아있음")


if __name__ == "__main__":
    main()
