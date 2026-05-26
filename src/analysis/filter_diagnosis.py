"""
필터 통과율 진단
- 각 진입 조건이 독립적으로 몇 % 통과시키는지 측정
- 어떤 조건의 조합이 병목인지 파악
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import MA_PERIOD, ENTROPY_PCT

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _setup_font():
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()


def diagnose_filters(df, mpe, h_onchain=None, period_months=6):
    """
    전체 기간을 period_months 단위로 나눠
    각 필터 조건의 통과율 및 조합 통과율 계산
    """
    rsi    = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    ma200  = df["close"].rolling(MA_PERIOD).mean()

    threshold = np.percentile(mpe.dropna(), ENTROPY_PCT)
    oc_thresh = None
    if h_onchain is not None:
        oc_thresh = np.percentile(h_onchain.dropna(), 40)

    # 전체 마스크 계산
    valid = mpe.dropna().index.intersection(df.index).intersection(rsi.index)

    f_rsi     = pd.Series(False, index=df.index)
    f_mpe     = pd.Series(False, index=df.index)
    f_ma      = pd.Series(False, index=df.index)
    f_onchain = pd.Series(True,  index=df.index)

    for idx in valid:
        sig     = signal.loc[idx] if idx in signal.index else 0
        mpe_val = mpe.loc[idx]
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        price   = df["close"].loc[idx]

        f_rsi[idx] = (sig == 1)
        f_mpe[idx] = (mpe_val <= threshold)
        f_ma[idx]  = (not np.isnan(ma_val)) and (price > ma_val)

        if h_onchain is not None and oc_thresh is not None:
            oc_val = h_onchain.loc[idx] if idx in h_onchain.index else np.nan
            f_onchain[idx] = (not np.isnan(oc_val)) and (oc_val <= oc_thresh)

    # 기간별 집계
    periods = []
    start   = df.index[0]
    end     = df.index[-1]
    delta   = pd.DateOffset(months=period_months)
    cursor  = start

    while cursor < end:
        p_end = min(cursor + delta, end)
        mask  = (df.index >= cursor) & (df.index < p_end)
        n     = mask.sum()
        if n < 100:
            cursor += delta
            continue

        r  = f_rsi[mask].sum()
        m  = f_mpe[mask].sum()
        a  = f_ma[mask].sum()
        o  = f_onchain[mask].sum()
        rm = (f_rsi & f_mpe)[mask].sum()
        rma= (f_rsi & f_mpe & f_ma)[mask].sum()
        all_= (f_rsi & f_mpe & f_ma & f_onchain)[mask].sum()

        periods.append({
            "label":    cursor.strftime("%Y-%m"),
            "n":        n,
            "rsi":      r / n * 100,
            "mpe":      m / n * 100,
            "ma":       a / n * 100,
            "onchain":  o / n * 100,
            "rsi+mpe":  rm / n * 100,
            "rsi+mpe+ma": rma / n * 100,
            "all":      all_ / n * 100,
            "all_count": int(all_),
        })
        cursor += delta

    return periods, threshold, oc_thresh


def print_diagnosis(periods, threshold, oc_thresh):
    oc_str = f"{oc_thresh:.4f}" if oc_thresh is not None else "N/A"
    print("\n" + "=" * 110)
    print(f"필터 진단  |  MPE 임계값: {threshold:.4f}  |  온체인 임계값: {oc_str}")
    print(f"{'기간':<10} {'총봉수':>7}  "
          f"{'RSI<30':>8} {'MPE<임계':>9} {'>MA200':>8} {'온체인':>8}  "
          f"{'RSI+MPE':>9} {'R+M+MA':>8} {'전체':>7} {'진입수':>6}")
    print("-" * 110)
    for p in periods:
        print(f"{p['label']:<10} {p['n']:>7}  "
              f"{p['rsi']:>7.1f}%  {p['mpe']:>8.1f}%  {p['ma']:>7.1f}%  "
              f"{p['onchain']:>7.1f}%  "
              f"{p['rsi+mpe']:>8.1f}%  {p['rsi+mpe+ma']:>7.1f}%  "
              f"{p['all']:>6.2f}%  {p['all_count']:>5}")
    print("=" * 110)

    # 전체 평균
    avg = lambda k: np.mean([p[k] for p in periods])
    print(f"\n전체 평균:  RSI<30 {avg('rsi'):.1f}%  |  MPE<임계 {avg('mpe'):.1f}%  |  "
          f">MA200 {avg('ma'):.1f}%  |  온체인 {avg('onchain'):.1f}%")
    print(f"           RSI+MPE {avg('rsi+mpe'):.2f}%  |  "
          f"RSI+MPE+MA {avg('rsi+mpe+ma'):.2f}%  |  전체 {avg('all'):.2f}%")


def plot_diagnosis(periods, save=True):
    RESULTS_DIR.mkdir(exist_ok=True)
    n      = len(periods)
    labels = [p["label"] for p in periods]
    x      = np.arange(n)

    fig = plt.figure(figsize=(18, 13), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32,
                            height_ratios=[1.4, 1.2, 1.2])

    # ── 1. 조건별 독립 통과율 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    w = 0.18
    bars = [
        ("RSI<30",   "rsi",     "#58a6ff", -1.5),
        ("MPE<임계", "mpe",     "#f7931a",  -0.5),
        (">MA200",   "ma",      "#3fb950",   0.5),
        ("온체인",   "onchain", "#9945ff",   1.5),
    ]
    for label, key, color, offset in bars:
        vals = [p[key] for p in periods]
        ax1.bar(x + offset * w, vals, w, color=color, alpha=0.85, label=label)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="#8b949e", fontsize=8.5, rotation=30, ha="right")
    ax1.set_ylabel("통과율 (%)", color="#8b949e")
    ax1.set_title("조건별 독립 통과율  —  각 조건이 혼자일 때 몇 %를 통과시키는가",
                  color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=4)
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 2. 조합별 통과율 (누적 필터링) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)

    combos = [
        ("RSI<30",          "rsi",        "#58a6ff"),
        ("RSI+MPE",         "rsi+mpe",    "#f7931a"),
        ("RSI+MPE+MA200",   "rsi+mpe+ma", "#3fb950"),
        ("전체 (진입조건)", "all",        "#ff7b72"),
    ]
    for label, key, color in combos:
        vals = [p[key] for p in periods]
        lw   = 2.2 if key == "all" else 1.4
        ax2.plot(x, vals, color=color, linewidth=lw, marker="o",
                 markersize=5, label=label)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=8.5, rotation=30, ha="right")
    ax2.set_ylabel("통과율 (%)", color="#8b949e")
    ax2.set_yscale("log")
    ax2.set_title("조합 누적 통과율  (로그 스케일)  —  조건을 쌓을수록 얼마나 좁아지는가",
                  color="#e6edf3", fontsize=11)
    ax2.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=4)
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 3. MA200 통과율 vs BTC 가격 ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)

    ma_vals  = [p["ma"]  for p in periods]
    rsi_vals = [p["rsi"] for p in periods]

    ax3.bar(x - 0.2, ma_vals,  0.35, color="#3fb950", alpha=0.85, label=">MA200 통과율")
    ax3.bar(x + 0.2, rsi_vals, 0.35, color="#58a6ff", alpha=0.85, label="RSI<30 통과율")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color="#8b949e", fontsize=8.5, rotation=30, ha="right")
    ax3.set_ylabel("통과율 (%)", color="#8b949e")
    ax3.set_title("핵심 필터 비교  —  MA200 vs RSI<30", color="#e6edf3", fontsize=10)
    ax3.legend(fontsize=9, facecolor="#21262d", labelcolor="#e6edf3")
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. 실제 진입 횟수 ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)

    counts = [p["all_count"] for p in periods]
    colors = ["#3fb950" if c > 0 else "#30363d" for c in counts]
    ax4.bar(x, counts, color=colors, alpha=0.88)
    for xi, c in zip(x, counts):
        if c > 0:
            ax4.text(xi, c + 0.05, str(c), ha="center", va="bottom",
                     color="#e6edf3", fontsize=10, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, color="#8b949e", fontsize=8.5, rotation=30, ha="right")
    ax4.set_ylabel("진입 횟수", color="#8b949e")
    ax4.set_title("기간별 실제 진입 횟수", color="#e6edf3", fontsize=10)
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    fig.suptitle(
        "필터 통과율 진단  —  어떤 조건이 진입을 막고 있는가?",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        path = RESULTS_DIR / "filter_diagnosis.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def _style(ax):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.yaxis.grid(True, color="#21262d", linestyle="--")
    ax.xaxis.grid(True, color="#21262d", linestyle=":", linewidth=0.4)
