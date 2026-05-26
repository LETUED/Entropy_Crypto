"""
MPE Ablation Test - 엔트로피가 진짜 기여하는가?

비교 전략 (포지션 사이징 0.15 고정, 청산 RSI>50 OR 168H 동일):
  A. RSI<30 + MA200             (엔트로피 없음)
  B. RSI<30 + MA200 + MPE<10%   (온체인 없음)
  C. RSI<30 + MA200 + MPE<10% + 온체인  (현재 풀 전략)

만약 A ≈ B ≈ C → MPE는 장식
만약 B > A      → MPE가 실제 필터 역할
만약 C > B      → 온체인이 추가 기여

실행: py run_mpe_ablation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import compute_metrics, MA_PERIOD, MAX_HOLD_H, FEE_RATE, ENTROPY_PCT

RESULTS_DIR = Path("results")
START, END  = "2021-01-01", "2025-01-01"
POS_SIZE    = 0.15   # 전략 간 공정 비교: 고정 포지션


def run_ablation(df, mpe, h_onchain=None):
    """
    A / B / C 전략을 동일 조건으로 실행.
    포지션 사이징 고정(0.15), 청산 동일(RSI>50 OR 168H).
    """
    rsi      = compute_rsi(df["close"])
    signal   = generate_signals(rsi)
    ma200    = df["close"].rolling(MA_PERIOD).mean()
    threshold = np.percentile(mpe.dropna(), ENTROPY_PCT)

    onchain_ok = pd.Series(True, index=df.index)
    if h_onchain is not None:
        oc_thresh  = np.percentile(h_onchain.dropna(), 40)
        onchain_ok = h_onchain <= oc_thresh

    strategies = {
        "A: RSI+MA200 (엔트로피 없음)":        dict(use_mpe=False, use_oc=False),
        "B: RSI+MA200+MPE (온체인 없음)":       dict(use_mpe=True,  use_oc=False),
        "C: RSI+MA200+MPE+온체인 (풀 전략)":    dict(use_mpe=True,  use_oc=True),
    }

    results = {}
    for name, cfg in strategies.items():
        equity     = 1.0
        position   = 0
        entry_price = 0.0
        entry_hour  = -999
        curve      = []
        n_trades   = 0

        for i, idx in enumerate(df.index):
            price   = df["close"].loc[idx]
            sig     = signal.loc[idx]    if idx in signal.index    else 0
            mpe_val = mpe.loc[idx]       if idx in mpe.index       else np.nan
            ma_val  = ma200.loc[idx]     if idx in ma200.index     else np.nan
            rsi_val = rsi.loc[idx]       if idx in rsi.index       else 50.0
            oc_ok   = onchain_ok.loc[idx] if idx in onchain_ok.index else True

            # 청산
            if position == 1:
                if rsi_val > 50 or (i - entry_hour) >= MAX_HOLD_H:
                    pnl = (price - entry_price) / entry_price
                    equity *= (1 + pnl * POS_SIZE) * (1 - FEE_RATE)
                    position = 0

            # 진입 조건 조합
            if position == 0 and sig == 1:
                ma_ok  = (not np.isnan(ma_val)) and (price > ma_val)
                mpe_ok = (not np.isnan(mpe_val)) and (mpe_val <= threshold)

                long_ok = ma_ok
                if cfg["use_mpe"]: long_ok = long_ok and mpe_ok
                if cfg["use_oc"]:  long_ok = long_ok and oc_ok

                if long_ok:
                    position    = 1
                    entry_price = price * (1 + FEE_RATE)
                    entry_hour  = i
                    n_trades   += 1

            curve.append(equity)

        eq = pd.Series(curve, index=df.index)
        results[name] = {"equity": eq, "metrics": compute_metrics(eq), "n_trades": n_trades}

    return results


def print_results(results):
    print("\n" + "=" * 80)
    print(f"{'전략':<35} {'진입':>6} {'수익률':>9} {'Sharpe':>8} {'최대낙폭':>10}")
    print("-" * 80)
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<35} {r['n_trades']:>6}번 {m['총 수익률']:>9} {m['Sharpe']:>8} {m['최대 낙폭']:>10}")
    print("=" * 80)

    # 핵심 해석
    sharpes = {k: float(v["metrics"]["Sharpe"]) for k, v in results.items()}
    names   = list(sharpes.keys())
    sa, sb, sc = sharpes[names[0]], sharpes[names[1]], sharpes[names[2]]

    print("\n[해석]")
    diff_ab = sb - sa
    diff_bc = sc - sb
    print(f"  MPE 기여 (B-A): {diff_ab:+.3f}  →  ", end="")
    if diff_ab > 0.05:
        print("MPE가 실제 필터 역할 - 엔트로피 신호력 존재")
    elif diff_ab > -0.05:
        print("MPE 기여 미미 - 사실상 RSI+MA200과 동일")
    else:
        print("MPE가 오히려 성과를 해침")

    print(f"  온체인 기여 (C-B): {diff_bc:+.3f}  →  ", end="")
    if diff_bc > 0.05:
        print("온체인 필터 실제 기여")
    elif diff_bc > -0.05:
        print("온체인 기여 미미")
    else:
        print("온체인이 오히려 성과를 해침")


def plot_results(results, df):
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        import matplotlib.font_manager as fm
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    COLORS = {
        "A: RSI+MA200 (엔트로피 없음)":      "#ff7b72",
        "B: RSI+MA200+MPE (온체인 없음)":     "#79c0ff",
        "C: RSI+MA200+MPE+온체인 (풀 전략)":  "#56d364",
    }

    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    fig.suptitle("MPE Ablation - 엔트로피가 진짜 기여하는가?",
                 color="#e6edf3", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── 1. 누적 수익 곡선 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    bah = df["close"] / df["close"].iloc[0]
    ax1.plot(bah.index, bah.values, color="#8b949e",
             linewidth=0.8, alpha=0.3, linestyle="--", label="BTC 바이앤홀드")
    for name, r in results.items():
        ax1.plot(r["equity"].index, r["equity"].values,
                 color=COLORS[name], linewidth=1.8,
                 label=f"{name}  (Sharpe {r['metrics']['Sharpe']}, {r['n_trades']}번)")
    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.7)
    ax1.set_title("누적 수익 비교", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")

    # ── 2. Sharpe 바 차트 ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    names  = list(results.keys())
    labels = ["A\nRSI+MA200", "B\n+MPE", "C\n+온체인"]
    sharpes = [float(results[n]["metrics"]["Sharpe"]) for n in names]
    colors  = [COLORS[n] for n in names]
    bars = ax2.bar(labels, sharpes, color=colors, width=0.5)
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", color="#e6edf3", fontsize=10)
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_title("Sharpe 비교", color="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # ── 3. 진입 횟수 바 차트 ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    n_trades = [results[n]["n_trades"] for n in names]
    bars2 = ax3.bar(labels, n_trades, color=colors, width=0.5)
    for bar, val in zip(bars2, n_trades):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", color="#e6edf3", fontsize=11)
    ax3.set_title("진입 횟수 비교", color="#e6edf3")
    ax3.set_ylabel("진입 횟수", color="#8b949e")

    # ── 4. Sharpe 차이 (기여도) ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    _style(ax4)
    contrib_labels = ["MPE 기여\n(B - A)", "온체인 기여\n(C - B)"]
    contrib_vals   = [sharpes[1] - sharpes[0], sharpes[2] - sharpes[1]]
    contrib_colors = ["#56d364" if v > 0 else "#ff7b72" for v in contrib_vals]
    bars3 = ax4.bar(contrib_labels, contrib_vals, color=contrib_colors, width=0.4)
    for bar, val in zip(bars3, contrib_vals):
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02
        ax4.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:+.3f}", ha="center", va="bottom", color="#e6edf3", fontsize=12, fontweight="bold")
    ax4.axhline(0, color="#8b949e", linewidth=0.8)
    ax4.set_title("각 필터의 Sharpe 기여도  (양수 = 성과 향상, 음수 = 성과 저하)",
                  color="#e6edf3")
    ax4.set_ylabel("Sharpe 변화량", color="#8b949e")

    plt.savefig(RESULTS_DIR / "mpe_ablation.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("저장: results/mpe_ablation.png")


def main():
    print("=" * 65)
    print("MPE Ablation Test - 엔트로피가 진짜 기여하는가?")
    print(f"  기간: {START} ~ {END}  |  포지션 고정: {POS_SIZE*100:.0f}%")
    print("=" * 65)

    print("\n[1/3] 데이터 수집...")
    df      = collect("BTCUSDT", "1h", START, END)
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("[2/3] MPE 계산... (수분 소요)")
    mpe = rolling_mpe(df["close"], window=168)
    print(f"  완료: {mpe.dropna().shape[0]}개 유효값")

    print("[3/3] 온체인 엔트로피 계산...")
    h_onchain = combined_onchain_entropy(
        funding_entropy(funding, df.index),
        fear_greed_entropy(fg, df.index)
    )

    results = run_ablation(df, mpe, h_onchain)
    print_results(results)
    plot_results(results, df)


if __name__ == "__main__":
    main()
