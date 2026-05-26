"""
저엔트로피 롱만 테스트 - 숏 제거 후 성과 변화 확인
가설: 숏 거래가 저엔트로피 역추세 PnL을 끌어내리는지 확인

기존: allow_short=True -> 저엔트로피 369건, 승률 65.6%, PnL -0.099%
변경: allow_short=False -> 롱 거래만 허용

실행: py run_longonly_multicoin.py
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
from src.analysis.regime_strategy import run_regime_strategy, LOW_PCT, HIGH_PCT, BREAKOUT_BARS
from src.analysis.h4_backtest import compute_metrics

RESULTS_DIR = Path("results")
START, END  = "2021-01-01", "2025-01-01"

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
]

COIN_LABELS = {
    "BTCUSDT":  "BTC", "ETHUSDT": "ETH",  "SOLUSDT":  "SOL",
    "BNBUSDT":  "BNB", "AVAXUSDT": "AVAX", "ADAUSDT":  "ADA",
    "DOGEUSDT": "DOGE","LINKUSDT": "LINK", "DOTUSDT":  "DOT",
    "MATICUSDT":"MATIC",
}

COIN_COLORS = [
    "#f0c040","#79c0ff","#56d364","#f78166","#d2a8ff",
    "#ffab70","#3fb950","#58a6ff","#ff7b72","#bc8cff",
]


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def run_all_coins(funding, fg):
    all_results_long = {}
    all_results_both = {}

    for sym in COINS:
        label = COIN_LABELS[sym]
        print(f"  [{label}] 데이터 수집 + MPE 계산 중...")
        try:
            df  = collect(sym, "1h", START, END)
            mpe = rolling_mpe(df["close"], window=168)
            h_onchain = combined_onchain_entropy(
                funding_entropy(funding, df.index),
                fear_greed_entropy(fg, df.index),
            )

            # 롱 전용
            out_long = run_regime_strategy(df, mpe, h_onchain, allow_short=False)
            m_long   = compute_metrics(out_long["equity"])
            trades_l = out_long["trades"]
            n_low_l  = len(trades_l[trades_l["regime"] == "low"])  if len(trades_l) else 0
            n_high_l = len(trades_l[trades_l["regime"] == "high"]) if len(trades_l) else 0

            # 롱+숏 (비교용)
            out_both = run_regime_strategy(df, mpe, h_onchain, allow_short=True)
            m_both   = compute_metrics(out_both["equity"])
            trades_b = out_both["trades"]
            n_low_b  = len(trades_b[trades_b["regime"] == "low"])  if len(trades_b) else 0
            n_high_b = len(trades_b[trades_b["regime"] == "high"]) if len(trades_b) else 0

            all_results_long[sym] = {
                "label":   label,
                "equity":  out_long["equity"],
                "metrics": m_long,
                "trades":  trades_l,
                "n_low":   n_low_l,
                "n_high":  n_high_l,
            }
            all_results_both[sym] = {
                "label":   label,
                "equity":  out_both["equity"],
                "metrics": m_both,
                "trades":  trades_b,
                "n_low":   n_low_b,
                "n_high":  n_high_b,
            }

            print(f"       롱전용: Sharpe {m_long['Sharpe']:>7} | "
                  f"저엔트 {n_low_l}번 | 고엔트 {n_high_l}번  |  "
                  f"롱+숏: Sharpe {m_both['Sharpe']:>7} | "
                  f"저엔트 {n_low_b}번 | 고엔트 {n_high_b}번")
        except Exception as e:
            print(f"       오류: {e}")

    return all_results_long, all_results_both


def print_summary(all_results_long, all_results_both):
    print("\n" + "=" * 100)
    print(f"{'코인':<6} | {'롱전용 Sharpe':>12} {'저엔트':>6} {'고엔트':>6} | "
          f"{'롱+숏 Sharpe':>12} {'저엔트':>6} {'고엔트':>6} | {'Sharpe 차이':>10}")
    print("-" * 100)

    sharpe_long_list = []
    sharpe_both_list = []
    for sym in COINS:
        if sym not in all_results_long:
            continue
        rl = all_results_long[sym]
        rb = all_results_both[sym]
        sl = float(rl["metrics"]["Sharpe"])
        sb = float(rb["metrics"]["Sharpe"])
        diff = sl - sb
        diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
        print(f"{rl['label']:<6} | {sl:>12} {rl['n_low']:>6}번 {rl['n_high']:>6}번 | "
              f"{sb:>12} {rb['n_low']:>6}번 {rb['n_high']:>6}번 | {diff_str:>10}")
        sharpe_long_list.append(sl)
        sharpe_both_list.append(sb)

    print("=" * 100)
    avg_l = np.mean(sharpe_long_list)
    avg_b = np.mean(sharpe_both_list)
    diff_avg = avg_l - avg_b
    diff_str = f"+{diff_avg:.3f}" if diff_avg >= 0 else f"{diff_avg:.3f}"
    print(f"{'평균':<6} | {avg_l:>12.3f}{'':>13} | {avg_b:>12.3f}{'':>13} | {diff_str:>10}")

    # 레짐별 롱 전용 PnL 통계
    print("\n[롱 전용 - 레짐별 trade 통계]")
    all_trades_long = pd.concat(
        [r["trades"] for r in all_results_long.values() if len(r["trades"]) > 0],
        ignore_index=True
    )
    if len(all_trades_long):
        closed = all_trades_long.dropna(subset=["pnl"])
        for reg, label in [("low", "저엔트로피 역추세"), ("high", "고엔트로피 모멘텀")]:
            sub = closed[closed["regime"] == reg]
            if len(sub):
                wr  = (sub["pnl"] > 0).mean() * 100
                avg = sub["pnl"].mean() * 100
                print(f"  {label:<20}: {len(sub):>4}건 | 승률 {wr:.1f}% | 평균 PnL {avg:.3f}%")

    # 레짐별 롱+숏 PnL 통계 (비교)
    print("\n[롱+숏 - 레짐별 trade 통계 (비교)]")
    all_trades_both = pd.concat(
        [r["trades"] for r in all_results_both.values() if len(r["trades"]) > 0],
        ignore_index=True
    )
    if len(all_trades_both):
        closed = all_trades_both.dropna(subset=["pnl"])
        for reg, label in [("low", "저엔트로피 역추세"), ("high", "고엔트로피 모멘텀")]:
            sub = closed[closed["regime"] == reg]
            if len(sub):
                wr  = (sub["pnl"] > 0).mean() * 100
                avg = sub["pnl"].mean() * 100
                print(f"  {label:<20}: {len(sub):>4}건 | 승률 {wr:.1f}% | 평균 PnL {avg:.3f}%")

        # 숏만 분리
        print("\n[숏 거래만 분리 - 얼마나 끌어내리는지]")
        for reg, label in [("low", "저엔트로피 역추세"), ("high", "고엔트로피 모멘텀")]:
            sub = closed[(closed["regime"] == reg) & (closed["direction"] == "short")]
            if len(sub):
                wr  = (sub["pnl"] > 0).mean() * 100
                avg = sub["pnl"].mean() * 100
                print(f"  {label} (숏만) : {len(sub):>4}건 | 승률 {wr:.1f}% | 평균 PnL {avg:.3f}%")

    return all_trades_long, all_trades_both


def plot_comparison(all_results_long, all_results_both, all_trades_long, all_trades_both):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    fig.suptitle("롱 전용 vs 롱+숏 비교 (저엔트로피 역추세 / 고엔트로피 모멘텀)",
                 color="#e6edf3", fontsize=14, y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                           height_ratios=[2.5, 1.2, 1.2])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── 1. 누적 수익 비교 (BTC만) ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    for (sym, r_l), (_, r_b), color in zip(
        all_results_long.items(), all_results_both.items(), COIN_COLORS
    ):
        label = r_l["label"]
        s_l = r_l["metrics"]["Sharpe"]
        s_b = r_b["metrics"]["Sharpe"]
        ax1.plot(r_l["equity"].index, r_l["equity"].values,
                 color=color, linewidth=1.5, alpha=0.9,
                 label=f"{label} 롱전용(S={s_l})")
        ax1.plot(r_b["equity"].index, r_b["equity"].values,
                 color=color, linewidth=0.8, alpha=0.4, linestyle="--")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("코인별 누적 수익 (실선=롱전용, 점선=롱+숏)", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. Sharpe 비교 바 ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    labels  = [r["label"] for r in all_results_long.values()]
    s_long  = [float(r["metrics"]["Sharpe"]) for r in all_results_long.values()]
    s_both  = [float(r["metrics"]["Sharpe"]) for r in all_results_both.values()]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w/2, s_long, w, color="#56d364", alpha=0.85, label="롱 전용")
    ax2.bar(x + w/2, s_both, w, color="#f78166", alpha=0.85, label="롱+숏")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30)
    ax2.set_title("코인별 Sharpe 비교", color="#e6edf3")
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 3. 거래수 비교 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    n_long = [r["n_low"] + r["n_high"] for r in all_results_long.values()]
    n_both = [r["n_low"] + r["n_high"] for r in all_results_both.values()]
    ax3.bar(x - w/2, n_long, w, color="#56d364", alpha=0.85, label="롱 전용")
    ax3.bar(x + w/2, n_both, w, color="#f78166", alpha=0.85, label="롱+숏")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30)
    ax3.set_title("코인별 거래 수 비교", color="#e6edf3")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. 레짐별 승률/PnL 비교 ───────────────────────────────────────────
    for col_idx, (all_trades, title_suffix) in enumerate([
        (all_trades_long, "롱 전용"),
        (all_trades_both, "롱+숏"),
    ]):
        ax = fig.add_subplot(gs[2, col_idx])
        _style(ax)
        if len(all_trades):
            closed = all_trades.dropna(subset=["pnl"])
            regs   = ["low", "high"]
            reg_labels = ["저엔트로피\n(역추세)", "고엔트로피\n(모멘텀)"]
            win_rates, avg_pnls, counts = [], [], []
            for reg in regs:
                sub = closed[closed["regime"] == reg]
                win_rates.append((sub["pnl"] > 0).mean() * 100 if len(sub) else 0)
                avg_pnls.append(sub["pnl"].mean() * 100         if len(sub) else 0)
                counts.append(len(sub))

            x2 = np.arange(len(regs))
            w2 = 0.3
            b1 = ax.bar(x2 - w2/2, win_rates, w2, color=["#56d364", "#f78166"], alpha=0.85, label="승률 (%)")
            b2 = ax.bar(x2 + w2/2, avg_pnls,  w2, color=["#1f6feb", "#b08800"], alpha=0.85, label="평균 PnL (%)")
            for bar, val in zip(list(b1) + list(b2), win_rates + avg_pnls):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", color="#e6edf3", fontsize=9)
            ax.set_xticks(x2)
            ax.set_xticklabels([f"{l}\n({c}건)" for l, c in zip(reg_labels, counts)],
                               color="#e6edf3")
            ax.axhline(50, color="#8b949e", linewidth=0.8, linestyle=":", label="50% 기준")
            ax.axhline(0,  color="#8b949e", linewidth=0.8)
            ax.set_title(f"레짐별 승률 vs 평균 PnL ({title_suffix})", color="#e6edf3")
            ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    plt.savefig(RESULTS_DIR / "longonly_vs_both.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/longonly_vs_both.png")


def main():
    print("=" * 65)
    print("저엔트로피 롱만 테스트 - 숏 제거 효과 분석")
    print(f"  코인: {len(COINS)}개  |  기간: {START} ~ {END}")
    print(f"  저엔트로피 MPE<{LOW_PCT}% + 고엔트로피 MPE>{HIGH_PCT}%")
    print(f"  롱 전용(allow_short=False) vs 롱+숏(allow_short=True) 비교")
    print("=" * 65)

    print("\n[공통 온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인별 실행 ({len(COINS)}개)]")
    all_results_long, all_results_both = run_all_coins(funding, fg)

    all_trades_long, all_trades_both = print_summary(all_results_long, all_results_both)
    plot_comparison(all_results_long, all_results_both, all_trades_long, all_trades_both)


if __name__ == "__main__":
    main()
