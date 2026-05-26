"""
5코인 포트폴리오 Walk-forward 검증
목적: 원래 전략(저엔트로피 롱+Kelly+온체인)을 양성 코인 5개에 집중
     통계적 유의성 확보 (코인당 18번 × 5코인 = ~90번)

코인: BTC, SOL, AVAX, ADA, DOT (원래 전략에서 양수 Sharpe 확인된 코인)
방법: 학습 12개월 → 테스트 6개월, 슬라이딩 6개월 (2022~2024, 6구간)
지표: 구간별 평균 Sharpe, 총 거래수, 포트폴리오 누적 수익

실행: py run_fivecoin_walkforward.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.walkforward import run_walkforward, print_walkforward
from src.analysis.h4_backtest import compute_metrics

RESULTS_DIR = Path("results")
START, END   = "2021-01-01", "2025-01-01"

# 원래 전략에서 양수 Sharpe 확인된 5코인
COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
COIN_LABELS = {
    "BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
    "ADAUSDT": "ADA", "DOTUSDT": "DOT",
}
COIN_COLORS = ["#f0c040", "#56d364", "#d2a8ff", "#ffab70", "#58a6ff"]

TRAIN_MONTHS = 12
TEST_MONTHS  = 6


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def run_coin_walkforward(sym, df, mpe, h_onchain):
    """단일 코인 walk-forward 실행 후 구간별 결과 반환"""
    results = run_walkforward(df, mpe, h_onchain,
                              train_months=TRAIN_MONTHS,
                              test_months=TEST_MONTHS,
                              step_months=TEST_MONTHS)
    return results


def aggregate_windows(coin_results: dict) -> list:
    """
    코인별 구간 결과를 통합 — 같은 테스트 구간의 Sharpe 평균 + 거래수 합산
    포트폴리오 equity = 5코인 균등 배분
    """
    # 공통 테스트 구간 식별 (BTC 기준)
    btc_results = coin_results["BTCUSDT"]
    windows = []

    for win_idx, btc_win in enumerate(btc_results):
        label      = btc_win["label"]
        short_lbl  = btc_win["short_label"]
        test_start = btc_win["test_start"]
        test_end   = btc_win["test_end"]

        sharpe_list = []
        ret_list    = []
        mdd_list    = []
        trade_total = 0
        equity_list = []

        for sym in COINS:
            coin_wins = coin_results[sym]
            # 같은 구간 찾기
            match = None
            for w in coin_wins:
                if w["test_start"] == test_start and w["test_end"] == test_end:
                    match = w
                    break
            if match is None:
                continue

            sharpe_list.append(float(match["metrics"]["Sharpe"]))
            ret_list.append(float(str(match["metrics"]["총 수익률"]).replace("%", "")))
            mdd_list.append(float(str(match["metrics"]["최대 낙폭"]).replace("%", "")))
            equity_list.append(match["equity"])

        if not sharpe_list:
            continue

        # 포트폴리오 equity: 각 코인 equity를 균등 가중 평균
        if equity_list:
            port_df = pd.concat(equity_list, axis=1).dropna()
            # 각 코인의 수익률 평균 → 포트폴리오 수익률
            port_equity = port_df.mean(axis=1)
        else:
            port_equity = None

        port_metrics = compute_metrics(port_equity) if port_equity is not None else {}

        windows.append({
            "label":        label,
            "short_label":  short_lbl,
            "test_start":   test_start,
            "test_end":     test_end,
            "avg_sharpe":   np.mean(sharpe_list),
            "sharpes":      dict(zip([COIN_LABELS[s] for s in COINS if s in
                                      [k for k in COINS]], sharpe_list)),
            "avg_ret":      np.mean(ret_list),
            "avg_mdd":      np.mean(mdd_list),
            "port_equity":  port_equity,
            "port_metrics": port_metrics,
            "n_coins":      len(sharpe_list),
        })

    return windows


def print_summary(windows, coin_results):
    print("\n" + "=" * 100)
    print(f"{'기간':<20} {'평균Sharpe':>10} {'BTC':>8} {'SOL':>8} {'AVAX':>8} "
          f"{'ADA':>8} {'DOT':>8} {'포트Sharpe':>10}")
    print("-" * 100)

    avg_sharpes   = []
    port_sharpes  = []

    # 코인별 Sharpe 테이블
    coin_sharpe_by_window = {sym: {} for sym in COINS}
    for sym in COINS:
        for w in coin_results[sym]:
            coin_sharpe_by_window[sym][w["short_label"]] = float(w["metrics"]["Sharpe"])

    for win in windows:
        sl = win["short_label"]
        avg = win["avg_sharpe"]
        ps  = float(str(win["port_metrics"].get("Sharpe", "0")).strip()) if win["port_metrics"] else 0

        coin_cols = ""
        for sym in COINS:
            s = coin_sharpe_by_window[sym].get(sl, float("nan"))
            sign = "+" if s > 0 else ""
            coin_cols += f"{sign}{s:>6.2f} " if not np.isnan(s) else f"{'N/A':>8}"

        marker = "★" if avg > 0 else "▼"
        print(f"{marker} {win['label']:<18} {avg:>+10.3f} {coin_cols} {ps:>+10.3f}")
        avg_sharpes.append(avg)
        port_sharpes.append(ps)

    print("=" * 100)
    print(f"  평균 Sharpe (5코인 평균) : {np.mean(avg_sharpes):+.3f}")
    print(f"  평균 Sharpe (포트폴리오) : {np.mean(port_sharpes):+.3f}")
    print(f"  양수 구간 비율          : "
          f"{sum(s > 0 for s in avg_sharpes)}/{len(avg_sharpes)} "
          f"({sum(s>0 for s in avg_sharpes)/len(avg_sharpes)*100:.0f}%)")

    # 전체 거래 통계 요약
    print(f"\n[참고: 원래 전략 전체 기간 성과]")
    print(f"  BTC +0.436 / SOL +0.630 / AVAX +0.437 / ADA +0.379 / DOT +0.395")
    print(f"  → 단일 코인 ~18번/4년 → 5코인 × ~18번 = ~90번 목표")


def plot_summary(windows, coin_results):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    n = len(windows)
    if n == 0:
        print("그릴 데이터 없음")
        return

    fig = plt.figure(figsize=(22, 22), facecolor="#0d1117")
    fig.suptitle(f"5코인 Walk-forward 검증 (학습 {TRAIN_MONTHS}M → 테스트 {TEST_MONTHS}M)\n"
                 f"BTC · SOL · AVAX · ADA · DOT | 원래 전략(저엔트로피 롱+Kelly+온체인)",
                 color="#e6edf3", fontsize=13, y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2.5, 1.3, 1.3, 1.2])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--")

    wlabels = [w["short_label"] for w in windows]
    avgs    = [w["avg_sharpe"] for w in windows]
    ports   = [float(str(w["port_metrics"].get("Sharpe","0"))) if w["port_metrics"] else 0
               for w in windows]

    # ── 1. 포트폴리오 누적 수익 (구간별 연결) ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    win_colors = plt.cm.plasma(np.linspace(0.15, 0.88, n))
    for i, win in enumerate(windows):
        eq = win["port_equity"]
        if eq is not None:
            ax1.plot(eq.index, eq.values,
                     color=win_colors[i], linewidth=2.0, alpha=0.9,
                     label=f"{win['label']}  Sharpe {win['avg_sharpe']:+.3f}")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 (5코인 균등 배분, 구간별)", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. 구간별 평균 Sharpe ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    colors_bar = ["#56d364" if s > 0 else "#ff7b72" for s in avgs]
    bars = ax2.bar(range(n), avgs, color=colors_bar, alpha=0.85)
    for bar, v in zip(bars, avgs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.01 if v >= 0 else -0.08),
                 f"{v:+.3f}", ha="center", color="#e6edf3", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.axhline(np.mean(avgs), color="#f7931a", linewidth=1.3, linestyle="--",
                label=f"평균 {np.mean(avgs):+.3f}")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(wlabels, rotation=30)
    ax2.set_title("구간별 5코인 평균 Sharpe (핵심 지표)", color="#e6edf3")
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 3. 포트폴리오 Sharpe ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    colors_p = ["#56d364" if s > 0 else "#ff7b72" for s in ports]
    bars3 = ax3.bar(range(n), ports, color=colors_p, alpha=0.85)
    for bar, v in zip(bars3, ports):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.01 if v >= 0 else -0.08),
                 f"{v:+.3f}", ha="center", color="#e6edf3", fontsize=9, fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.axhline(np.mean(ports), color="#f7931a", linewidth=1.3, linestyle="--",
                label=f"평균 {np.mean(ports):+.3f}")
    ax3.set_xticks(range(n)); ax3.set_xticklabels(wlabels, rotation=30)
    ax3.set_title("구간별 포트폴리오 Sharpe (균등 배분)", color="#e6edf3")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. 코인별 구간 Sharpe 히트맵 ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    _style(ax4)
    coin_sharpe_by_window = {sym: {} for sym in COINS}
    for sym in COINS:
        for w in coin_results[sym]:
            coin_sharpe_by_window[sym][w["short_label"]] = float(w["metrics"]["Sharpe"])

    matrix = []
    for sym in COINS:
        row = [coin_sharpe_by_window[sym].get(sl, np.nan) for sl in wlabels]
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    # 히트맵
    im = ax4.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax4.set_xticks(range(n)); ax4.set_xticklabels(wlabels, color="#8b949e")
    ax4.set_yticks(range(len(COINS)))
    ax4.set_yticklabels([COIN_LABELS[s] for s in COINS], color="#e6edf3")
    for row in range(len(COINS)):
        for col in range(n):
            val = matrix[row, col]
            if not np.isnan(val):
                ax4.text(col, row, f"{val:+.2f}", ha="center", va="center",
                         color="white" if abs(val) > 1 else "#0d1117", fontsize=9)
    plt.colorbar(im, ax=ax4, label="Sharpe")
    ax4.set_title("코인 × 구간 Sharpe 히트맵 (녹색=양수 / 적색=음수)", color="#e6edf3")

    # ── 5. 요약 통계 테이블 ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    _style(ax5)
    ax5.axis("off")

    pos_avg = sum(s > 0 for s in avgs)
    pos_port = sum(s > 0 for s in ports)

    # 코인별 전체 기간 양수 구간 수
    coin_pos = {}
    for sym in COINS:
        lbl = COIN_LABELS[sym]
        wins_this = [float(w["metrics"]["Sharpe"]) for w in coin_results[sym]]
        coin_pos[lbl] = f"{sum(s>0 for s in wins_this)}/{len(wins_this)}"

    rows = [
        ["테스트 구간 수",      f"{n}개 (각 {TEST_MONTHS}개월)"],
        ["5코인 평균 Sharpe",   f"{np.mean(avgs):+.3f}"],
        ["포트폴리오 Sharpe",   f"{np.mean(ports):+.3f}"],
        ["양수 구간 (평균)",    f"{pos_avg}/{n} ({pos_avg/n*100:.0f}%)"],
        ["양수 구간 (포트폴)",  f"{pos_port}/{n} ({pos_port/n*100:.0f}%)"],
        ["Sharpe 표준편차",     f"{np.std(avgs):.3f}"],
        ["BTC 양수 구간",       coin_pos.get("BTC","?")],
        ["SOL 양수 구간",       coin_pos.get("SOL","?")],
        ["AVAX 양수 구간",      coin_pos.get("AVAX","?")],
        ["ADA 양수 구간",       coin_pos.get("ADA","?")],
        ["DOT 양수 구간",       coin_pos.get("DOT","?")],
    ]

    tbl = ax5.table(cellText=rows, colLabels=["지표", "값"],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax5.set_title("Walk-forward 일관성 요약", color="#e6edf3", fontsize=10, pad=8)

    plt.savefig(RESULTS_DIR / "fivecoin_walkforward.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/fivecoin_walkforward.png")


def main():
    print("=" * 70)
    print("5코인 포트폴리오 Walk-forward 검증")
    print(f"  코인: BTC · SOL · AVAX · ADA · DOT")
    print(f"  기간: {START} ~ {END}")
    print(f"  방법: 학습 {TRAIN_MONTHS}M → 테스트 {TEST_MONTHS}M (슬라이딩)")
    print(f"  전략: 저엔트로피 롱 + Kelly 사이징 + MA200 + 온체인")
    print("=" * 70)

    print("\n[공통 온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인별 Walk-forward 실행...]")
    coin_results = {}
    for sym in COINS:
        label = COIN_LABELS[sym]
        print(f"\n  [{label}] 데이터 수집 + WF 실행...")
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168)
        h_onchain = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        results = run_coin_walkforward(sym, df, mpe, h_onchain)
        coin_results[sym] = results

        # 코인별 WF 요약
        print(f"  [{label}] Walk-forward 결과:")
        print_walkforward(results)

    print("\n" + "=" * 70)
    print("5코인 통합 집계")
    print("=" * 70)
    windows = aggregate_windows(coin_results)
    print_summary(windows, coin_results)
    plot_summary(windows, coin_results)


if __name__ == "__main__":
    main()
