"""
Kelly + 롱 전용 레짐 전략 - 원래 전략 강점 + 레짐 복합 결합
목적: 저엔트로피 Kelly 사이징 + 고엔트로피 Donchian 롱만

설계:
  저엔트로피 (MPE<10%): RSI<30 롱 + MA200 + 온체인 + Kelly 사이징 (50/30/15%)
  고엔트로피 (MPE>70%): Donchian 168H 신고가 롱 + MA200 + 고정 10%
  숏: 제거

비교 대상:
  A. Kelly+롱전용 레짐 (이번 실험)
  B. 롱전용 고정0.15 레짐 (이전 실험)
  C. 원래 전략 (저엔트로피 롱+Kelly, 레짐 없음)

실행: py run_kelly_regime_multicoin.py
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
from src.analysis.h4_backtest import compute_metrics, run_strategy

RESULTS_DIR = Path("results")
START, END  = "2021-01-01", "2025-01-01"

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
]
COIN_LABELS = {
    "BTCUSDT": "BTC", "ETHUSDT": "ETH",  "SOLUSDT": "SOL",
    "BNBUSDT": "BNB", "AVAXUSDT":"AVAX", "ADAUSDT": "ADA",
    "DOGEUSDT":"DOGE","LINKUSDT":"LINK", "DOTUSDT": "DOT",
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


def run_all(funding, fg):
    results = {}

    for sym in COINS:
        label = COIN_LABELS[sym]
        print(f"  [{label}] 실행 중...")
        try:
            df  = collect(sym, "1h", START, END)
            mpe = rolling_mpe(df["close"], window=168)
            h_onchain = combined_onchain_entropy(
                funding_entropy(funding, df.index),
                fear_greed_entropy(fg, df.index),
            )

            # A. Kelly + 롱 전용 레짐 (새 실험)
            out_kelly = run_regime_strategy(
                df, mpe, h_onchain, allow_short=False, use_kelly=True
            )
            m_kelly = compute_metrics(out_kelly["equity"])

            # B. 고정0.15 + 롱 전용 레짐 (이전 실험)
            out_fixed = run_regime_strategy(
                df, mpe, h_onchain, allow_short=False, use_kelly=False
            )
            m_fixed = compute_metrics(out_fixed["equity"])

            # C. 원래 전략 (저엔트로피 롱+Kelly, 레짐 없음)
            eq_orig = run_strategy(
                df, mpe,
                use_entropy_filter=True,
                use_trend_filter=True,
                use_kelly=True,
                allow_short=False,
                h_onchain=h_onchain,
            )
            m_orig = compute_metrics(eq_orig)

            trades_k = out_kelly["trades"]
            n_low  = len(trades_k[trades_k["regime"] == "low"])  if len(trades_k) else 0
            n_high = len(trades_k[trades_k["regime"] == "high"]) if len(trades_k) else 0

            results[sym] = {
                "label":    label,
                "kelly":    {"equity": out_kelly["equity"], "metrics": m_kelly, "trades": trades_k},
                "fixed":    {"equity": out_fixed["equity"], "metrics": m_fixed},
                "original": {"equity": eq_orig,             "metrics": m_orig},
                "n_low":    n_low,
                "n_high":   n_high,
            }

            print(
                f"       A(Kelly레짐): Sharpe {m_kelly['Sharpe']:>7} | "
                f"저엔트 {n_low}번 | 고엔트 {n_high}번  ||  "
                f"B(고정레짐): {m_fixed['Sharpe']:>7}  ||  "
                f"C(원래): {m_orig['Sharpe']:>7}"
            )

        except Exception as e:
            print(f"       오류: {e}")
            import traceback; traceback.print_exc()

    return results


def print_summary(results):
    print("\n" + "=" * 90)
    print(f"{'코인':<6} | {'A:Kelly레짐':>11} | {'B:고정레짐':>10} | {'C:원래전략':>10} | "
          f"{'A-B':>7} | {'A-C':>7}")
    print("-" * 90)

    s_kelly, s_fixed, s_orig = [], [], []
    for sym, r in results.items():
        sk = float(r["kelly"]["metrics"]["Sharpe"])
        sf = float(r["fixed"]["metrics"]["Sharpe"])
        so = float(r["original"]["metrics"]["Sharpe"])
        ab = f"+{sk-sf:.3f}" if sk >= sf else f"{sk-sf:.3f}"
        ac = f"+{sk-so:.3f}" if sk >= so else f"{sk-so:.3f}"
        print(f"{r['label']:<6} | {sk:>11.3f} | {sf:>10.3f} | {so:>10.3f} | {ab:>7} | {ac:>7}")
        s_kelly.append(sk); s_fixed.append(sf); s_orig.append(so)

    print("=" * 90)
    ak = np.mean(s_kelly); af = np.mean(s_fixed); ao = np.mean(s_orig)
    ab = f"+{ak-af:.3f}" if ak >= af else f"{ak-af:.3f}"
    ac = f"+{ak-ao:.3f}" if ak >= ao else f"{ak-ao:.3f}"
    print(f"{'평균':<6} | {ak:>11.3f} | {af:>10.3f} | {ao:>10.3f} | {ab:>7} | {ac:>7}")

    # Kelly 레짐 레짐별 상세
    all_trades = pd.concat(
        [r["kelly"]["trades"] for r in results.values() if len(r["kelly"]["trades"]) > 0],
        ignore_index=True
    )
    if len(all_trades):
        print("\n[Kelly 레짐 전략 - 레짐별 trade 통계]")
        closed = all_trades.dropna(subset=["pnl"])
        for reg, label in [("low", "저엔트로피 (Kelly)"), ("high", "고엔트로피 (고정0.10)")]:
            sub = closed[closed["regime"] == reg]
            if len(sub):
                wr  = (sub["pnl"] > 0).mean() * 100
                avg = sub["pnl"].mean() * 100
                # 평균 포지션 크기
                avg_sz = sub["pos_size"].mean() * 100 if "pos_size" in sub.columns else 0
                print(f"  {label:<22}: {len(sub):>4}건 | 승률 {wr:.1f}% | "
                      f"평균 PnL {avg:.3f}% | 평균 포지션 {avg_sz:.1f}%")

    return all_trades


def plot_results(results, all_trades):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(22, 20), facecolor="#0d1117")
    fig.suptitle("Kelly+롱전용 레짐 전략 vs 비교 (A=Kelly레짐 / B=고정레짐 / C=원래전략)",
                 color="#e6edf3", fontsize=13, y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                           height_ratios=[2.5, 1.3, 1.3])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── 1. 코인별 누적 수익 (A 실선, C 점선) ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    for (sym, r), color in zip(results.items(), COIN_COLORS):
        sk = r["kelly"]["metrics"]["Sharpe"]
        so = r["original"]["metrics"]["Sharpe"]
        ax1.plot(r["kelly"]["equity"].index, r["kelly"]["equity"].values,
                 color=color, linewidth=1.6, alpha=0.9,
                 label=f"{r['label']} A(S={sk})")
        ax1.plot(r["original"]["equity"].index, r["original"]["equity"].values,
                 color=color, linewidth=0.8, alpha=0.35, linestyle="--")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("코인별 누적 수익 (실선=Kelly레짐 / 점선=원래전략)", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. 3-way Sharpe 비교 바 ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    labels   = [r["label"] for r in results.values()]
    s_kelly  = [float(r["kelly"]["metrics"]["Sharpe"])    for r in results.values()]
    s_fixed  = [float(r["fixed"]["metrics"]["Sharpe"])    for r in results.values()]
    s_orig   = [float(r["original"]["metrics"]["Sharpe"]) for r in results.values()]
    x = np.arange(len(labels))
    w = 0.25
    ax2.bar(x - w, s_kelly, w, color="#56d364", alpha=0.85, label="A: Kelly 레짐")
    ax2.bar(x,     s_fixed, w, color="#79c0ff", alpha=0.85, label="B: 고정0.15 레짐")
    ax2.bar(x + w, s_orig,  w, color="#f0c040", alpha=0.85, label="C: 원래 전략")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30)
    ax2.set_title("코인별 3-way Sharpe 비교", color="#e6edf3")
    ax2.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 3. 거래 수 (저/고 엔트로피) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    n_low  = [r["n_low"]  for r in results.values()]
    n_high = [r["n_high"] for r in results.values()]
    ax3.bar(x, n_low,  color="#56d364", label="저엔트로피 (Kelly)", width=0.6)
    ax3.bar(x, n_high, bottom=n_low, color="#f78166",
            label="고엔트로피 (Donchian)", width=0.6)
    for i, (nl, nh) in enumerate(zip(n_low, n_high)):
        ax3.text(i, nl+nh+0.5, str(nl+nh), ha="center", color="#e6edf3", fontsize=8)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=30)
    ax3.set_title("코인별 거래 수 (Kelly 레짐)", color="#e6edf3")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. 레짐별 승률/PnL (Kelly) ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style(ax4)
    if len(all_trades):
        closed = all_trades.dropna(subset=["pnl"])
        regs   = ["low", "high"]
        reg_labels = ["저엔트로피\n(Kelly)", "고엔트로피\n(고정10%)"]
        win_rates, avg_pnls, counts = [], [], []
        for reg in regs:
            sub = closed[closed["regime"] == reg]
            win_rates.append((sub["pnl"] > 0).mean() * 100 if len(sub) else 0)
            avg_pnls.append(sub["pnl"].mean() * 100         if len(sub) else 0)
            counts.append(len(sub))
        x2 = np.arange(len(regs)); w2 = 0.3
        b1 = ax4.bar(x2 - w2/2, win_rates, w2, color=["#56d364","#f78166"], alpha=0.85, label="승률 (%)")
        b2 = ax4.bar(x2 + w2/2, avg_pnls,  w2, color=["#1f6feb","#b08800"], alpha=0.85, label="평균 PnL (%)")
        for bar, val in zip(list(b1)+list(b2), win_rates+avg_pnls):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}", ha="center", color="#e6edf3", fontsize=9)
        ax4.set_xticks(x2)
        ax4.set_xticklabels([f"{l}\n({c}건)" for l, c in zip(reg_labels, counts)],
                            color="#e6edf3")
        ax4.axhline(50, color="#8b949e", linewidth=0.8, linestyle=":", label="50% 기준")
        ax4.axhline(0,  color="#8b949e", linewidth=0.8)
        ax4.set_title("Kelly 레짐 - 레짐별 승률 vs 평균 PnL", color="#e6edf3")
        ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 5. Kelly 저엔트로피 포지션 사이즈 분포 ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style(ax5)
    if len(all_trades) and "pos_size" in all_trades.columns:
        closed = all_trades.dropna(subset=["pnl"])
        low_trades = closed[closed["regime"] == "low"]
        if len(low_trades):
            sizes = low_trades["pos_size"].value_counts().sort_index()
            colors_k = {0.50: "#f0c040", 0.30: "#56d364", 0.15: "#79c0ff", 0.10: "#f78166"}
            ax5.bar([f"{int(s*100)}%" for s in sizes.index],
                    sizes.values,
                    color=[colors_k.get(s, "#8b949e") for s in sizes.index],
                    width=0.5)
            for i, (sz, cnt) in enumerate(zip(sizes.index, sizes.values)):
                ax5.text(i, cnt+0.3, str(cnt), ha="center", color="#e6edf3", fontsize=10)
        ax5.set_title("저엔트로피 Kelly 포지션 크기 분포", color="#e6edf3")
        ax5.set_ylabel("거래 수", color="#8b949e")

    plt.savefig(RESULTS_DIR / "kelly_regime_multicoin.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/kelly_regime_multicoin.png")


def main():
    print("=" * 65)
    print("Kelly + 롱 전용 레짐 전략 - 원래 전략 강점 결합")
    print(f"  코인: {len(COINS)}개  |  기간: {START} ~ {END}")
    print(f"  저엔트로피 MPE<{LOW_PCT}%: RSI<30 롱 + Kelly 사이징 (50/30/15%)")
    print(f"  고엔트로피 MPE>{HIGH_PCT}%: Donchian {BREAKOUT_BARS}H 롱 + 고정 10%")
    print("=" * 65)

    print("\n[공통 온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인별 실행 ({len(COINS)}개) - A/B/C 3가지 전략 동시 비교]")
    results = run_all(funding, fg)

    all_trades = print_summary(results)
    plot_results(results, all_trades)


if __name__ == "__main__":
    main()
