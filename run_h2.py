"""
Phase 1 — H2 가설 검증 실행 스크립트

실행:
    py run_h2.py

결과:
    results/h2_timeseries.png   — 시계열 시각화
    results/h2_distribution.png — 분포 비교
    results/h2_summary.csv      — 요약 테이블
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_collector import collect
from src.analysis.h2_validation import validate_h2
from src.analysis.visualizer import plot_h2_results, plot_distribution_comparison, plot_summary_table


def main():
    # ── 데이터 수집 ───────────────────────────────────────────
    # 훈련+검증: 2021~2024, 테스트: 2025 (지금은 H2 전체 기간으로 탐색)
    df = collect(
        symbol="BTCUSDT",
        interval="1h",
        start="2021-01-01",
        end="2025-01-01",
    )

    # ── H2 검증 ──────────────────────────────────────────────
    results = validate_h2(df, mpe_window=168)  # 168H = 7일 롤링 윈도우

    # ── 시각화 ───────────────────────────────────────────────
    plot_summary_table(results)
    plot_h2_results(results)
    plot_distribution_comparison(results)


if __name__ == "__main__":
    main()
