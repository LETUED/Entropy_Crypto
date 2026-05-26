"""
엔트로피 레짐 복합 전략 - 멀티코인 확장판
목적: 통계적으로 유의미한 거래 데이터 축적

코인: BTC ETH SOL BNB AVAX ADA DOGE LINK DOT MATIC
저엔트로피 역추세 + 고엔트로피 Donchian 돌파 (48H 신고가/신저가)

실행: py run_regime_multicoin.py
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
    all_results = {}

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
            out = run_regime_strategy(df, mpe, h_onchain, allow_short=True)
            metrics = compute_metrics(out["equity"])

            trades  = out["trades"]
            n_low   = len(trades[trades["regime"] == "low"])  if len(trades) else 0
            n_high  = len(trades[trades["regime"] == "high"]) if len(trades) else 0

            all_results[sym] = {
                "label":   label,
                "equity":  out["equity"],
                "metrics": metrics,
                "trades":  trades,
                "n_low":   n_low,
                "n_high":  n_high,
                "n_total": n_low + n_high,
            }
            print(f"       Sharpe {metrics['Sharpe']:>7} | "
                  f"저엔트로피 {n_low}번 | 고엔트로피 {n_high}번 | 총 {n_low+n_high}번")
        except Exception as e:
            print(f"       오류: {e}")

    return all_results


def print_summary(all_results):
    print("\n" + "=" * 80)
    print(f"{'코인':<8} {'수익률':>9} {'Sharpe':>8} {'최대낙폭':>10} "
          f"{'저엔트(역추세)':>12} {'고엔트(모멘텀)':>12} {'총거래':>6}")
    print("-" * 80)

    total_low = total_high = 0
    sharpe_list = []
    for sym, r in all_results.items():
        m = r["metrics"]
        print(f"{r['label']:<8} {m['총 수익률']:>9} {m['Sharpe']:>8} "
              f"{m['최대 낙폭']:>10} {r['n_low']:>12}번 {r['n_high']:>12}번 {r['n_total']:>5}번")
        total_low  += r["n_low"]
        total_high += r["n_high"]
        sharpe_list.append(float(m["Sharpe"]))

    print("=" * 80)
    print(f"{'합계':<8} {'':>9} {'평균 ' + f'{np.mean(sharpe_list):.3f}':>8} "
          f"{'':>10} {total_low:>12}번 {total_high:>12}번 {total_low+total_high:>5}번")

    # 레짐별 전체 통계
    all_trades = pd.concat([r["trades"] for r in all_results.values()
                             if len(r["trades"]) > 0], ignore_index=True)
    if len(all_trades):
        closed = all_trades.dropna(subset=["pnl"])
        print(f"\n[전체 {len(closed)}건 closed trade 통계]")
        for reg, label in [("low", "저엔트로피 역추세"), ("high", "고엔트로피 모멘텀")]:
            sub = closed[closed["regime"] == reg]
            if len(sub):
                wr  = (sub["pnl"] > 0).mean() * 100
                avg = sub["pnl"].mean() * 100
                print(f"  {label:<20}: {len(sub):>4}건 | 승률 {wr:.1f}% | 평균 PnL {avg:.3f}%")

    return all_trades


def plot_summary(all_results, all_trades):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    fig.suptitle("엔트로피 레짐 복합 전략 - 멀티코인 데이터 축적",
                 color="#e6edf3", fontsize=14, y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35,
                           height_ratios=[2.5, 1.2, 1.2, 1])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── 1. 코인별 누적 수익 ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    for (sym, r), color in zip(all_results.items(), COIN_COLORS):
        sharpe = r["metrics"]["Sharpe"]
        ax1.plot(r["equity"].index, r["equity"].values,
                 color=color, linewidth=1.4, alpha=0.85,
                 label=f"{r['label']}  (S={sharpe})")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("코인별 누적 수익", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. 코인별 Sharpe ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    labels  = [r["label"] for r in all_results.values()]
    sharpes = [float(r["metrics"]["Sharpe"]) for r in all_results.values()]
    colors  = ["#56d364" if s > 0 else "#ff7b72" for s in sharpes]
    bars = ax2.bar(labels, sharpes, color=colors, width=0.6)
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.01 if val >= 0 else -0.08),
                 f"{val:.2f}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=8)
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_title("코인별 Sharpe", color="#e6edf3")
    ax2.tick_params(axis="x", rotation=30)

    # ── 3. 코인별 거래 수 (저/고 엔트로피 스택) ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    n_low_list  = [r["n_low"]  for r in all_results.values()]
    n_high_list = [r["n_high"] for r in all_results.values()]
    x = np.arange(len(labels))
    ax3.bar(x, n_low_list,  color="#56d364", label="저엔트로피 (역추세)", width=0.6)
    ax3.bar(x, n_high_list, bottom=n_low_list, color="#f78166",
            label="고엔트로피 (모멘텀)", width=0.6)
    for i, (nl, nh) in enumerate(zip(n_low_list, n_high_list)):
        ax3.text(i, nl + nh + 0.5, str(nl+nh), ha="center",
                 color="#e6edf3", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30)
    ax3.set_title("코인별 거래 수", color="#e6edf3")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")

    # ── 4. 레짐별 전체 승률 & 평균 PnL ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    _style(ax4)
    if len(all_trades):
        closed = all_trades.dropna(subset=["pnl"])
        regs   = ["low", "high"]
        reg_labels = ["저엔트로피 (역추세)", "고엔트로피 (모멘텀, Donchian48H)"]
        win_rates, avg_pnls, counts = [], [], []
        for reg in regs:
            sub = closed[closed["regime"] == reg]
            win_rates.append((sub["pnl"] > 0).mean() * 100 if len(sub) else 0)
            avg_pnls.append(sub["pnl"].mean() * 100         if len(sub) else 0)
            counts.append(len(sub))

        x2 = np.arange(len(regs))
        w  = 0.3
        b1 = ax4.bar(x2 - w/2, win_rates, w, color=["#56d364","#f78166"], alpha=0.85, label="승률 (%)")
        b2 = ax4.bar(x2 + w/2, avg_pnls,  w, color=["#1f6feb","#b08800"], alpha=0.85, label="평균 PnL (%)")
        for bar, val in zip(list(b1)+list(b2), win_rates+avg_pnls):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", color="#e6edf3", fontsize=9)
        ax4.set_xticks(x2)
        ax4.set_xticklabels([f"{l}\n({c}건)" for l, c in zip(reg_labels, counts)],
                            color="#e6edf3")
        ax4.axhline(50, color="#8b949e", linewidth=0.8, linestyle=":", label="50% 기준")
        ax4.axhline(0,  color="#8b949e", linewidth=0.8)
        ax4.set_title("전체 코인 통합 - 레짐별 승률 vs 평균 PnL", color="#e6edf3")
        ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3")

    # ── 5. PnL 분포 히스토그램 ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    _style(ax5)
    if len(all_trades):
        closed = all_trades.dropna(subset=["pnl"])
        for reg, color, label in [
            ("low",  "#56d364", "저엔트로피"),
            ("high", "#f78166", "고엔트로피"),
        ]:
            sub = closed[closed["regime"] == reg]["pnl"] * 100
            if len(sub):
                ax5.hist(sub, bins=40, color=color, alpha=0.6,
                         label=f"{label} (n={len(sub)}, 평균={sub.mean():.2f}%)")
        ax5.axvline(0, color="#8b949e", linewidth=1.0, linestyle="--")
        ax5.set_title("PnL 분포 (전체 코인 통합)", color="#e6edf3")
        ax5.set_xlabel("거래당 PnL (%)", color="#8b949e")
        ax5.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3")

    plt.savefig(RESULTS_DIR / "regime_multicoin.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/regime_multicoin.png")


def main():
    print("=" * 65)
    print("엔트로피 레짐 복합 전략 - 멀티코인 데이터 축적")
    print(f"  코인: {len(COINS)}개  |  기간: {START} ~ {END}")
    print(f"  저엔트로피 MPE<{LOW_PCT}% : 역추세 (RSI<30 롱/RSI>70 숏)")
    print(f"  고엔트로피 MPE>{HIGH_PCT}% : Donchian {BREAKOUT_BARS}H 돌파 모멘텀")
    print("=" * 65)

    print("\n[공통 온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인별 실행 ({len(COINS)}개)]")
    all_results = run_all_coins(funding, fg)

    all_trades = print_summary(all_results)
    plot_summary(all_results, all_trades)


if __name__ == "__main__":
    main()
