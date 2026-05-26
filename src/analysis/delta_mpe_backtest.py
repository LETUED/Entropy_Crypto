"""
Delta MPE 전략 백테스트
- 신호: MPE가 빠르게 하락하는 순간 (고→저 레짐 전환)
- 기존 "MPE 이미 낮음" 대신 "MPE가 낮아지는 속도"를 사용
- 논문 근거: Thermodynamic Analysis (2023) — Delta Entropy가 분산 41~57% 설명

전략 비교:
  A. delta_mpe only      (RSI만, MA200 없음)
  B. delta_mpe + MA200   (추세 필터 추가)
  C. 기존 전략           (MPE level + RSI + MA200, 비교용)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import kelly_size, compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
DELTA_PCT   = 15   # Delta MPE 하위 15% (기울기 급락 구간)


def _setup_font():
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()


# ── 단일 전략 실행 ─────────────────────────────────────────────────────────────
def run_delta_strategy(
    df: pd.DataFrame,
    mpe: pd.Series,
    delta_mpe: pd.Series,
    use_ma200: bool = True,
    h_onchain: pd.Series = None,
) -> pd.Series:
    """
    Delta MPE 기반 전략
    - 진입: delta_mpe < 하위 15% (MPE 빠르게 하락) + RSI < 30
    - 청산: RSI > 50 OR 최대 168H
    """
    rsi    = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    ma200  = df["close"].rolling(MA_PERIOD).mean()

    # 학습 전체 기간 기반 임계값 (단일 백테스트용)
    delta_threshold = np.percentile(delta_mpe.dropna(), DELTA_PCT)
    oc_threshold    = None
    if h_onchain is not None:
        oc_threshold = np.percentile(h_onchain.dropna(), 40)

    equity     = 1.0
    position   = 0
    entry_price = 0.0
    entry_hour  = -999
    pos_size    = 0.3
    curve       = []

    for i, idx in enumerate(df.index):
        price     = df["close"].loc[idx]
        sig       = signal.loc[idx]    if idx in signal.index    else 0
        rsi_val   = rsi.loc[idx]       if idx in rsi.index       else 50.0
        delta_val = delta_mpe.loc[idx] if idx in delta_mpe.index else np.nan
        ma_val    = ma200.loc[idx]     if idx in ma200.index     else np.nan
        mpe_val   = mpe.loc[idx]       if idx in mpe.index       else np.nan

        is_falling = (not np.isnan(delta_val)) and (delta_val <= delta_threshold)

        oc_ok = True
        if h_onchain is not None and oc_threshold is not None:
            if idx in h_onchain.index:
                oc_ok = h_onchain.loc[idx] <= oc_threshold

        # 청산: RSI 회복(>50) 또는 최대 168H
        if position == 1:
            if rsi_val > 50 or (i - entry_hour) >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * pos_size) * (1 - FEE_RATE)
                position = 0

        # 진입
        if position == 0:
            mpe_safe = mpe_val if not np.isnan(mpe_val) else 1.0
            pos_size = kelly_size(mpe_safe, mpe)

            trend_ok = True
            if use_ma200:
                trend_ok = not np.isnan(ma_val) and price > ma_val

            if sig == 1 and is_falling and trend_ok and pos_size > 0 and oc_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i

        curve.append(equity)

    return pd.Series(curve, index=df.index)


# ── 원래 전략 (비교용) ────────────────────────────────────────────────────────
def run_original_strategy(df, mpe, h_onchain=None):
    """기존 MPE level 전략 (비교 기준선)"""
    from src.analysis.h4_backtest import run_strategy
    return run_strategy(df, mpe,
                        use_entropy_filter=True, use_trend_filter=True,
                        use_kelly=True, allow_short=False,
                        h_onchain=h_onchain)


# ── 신호 빈도 진단 ────────────────────────────────────────────────────────────
def count_signals(df, mpe, delta_mpe, h_onchain=None):
    """각 필터 조합별 진입 횟수 비교"""
    rsi    = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    ma200  = df["close"].rolling(MA_PERIOD).mean()

    mpe_threshold   = np.percentile(mpe.dropna(), ENTROPY_PCT)
    delta_threshold = np.percentile(delta_mpe.dropna(), DELTA_PCT)
    oc_threshold    = np.percentile(h_onchain.dropna(), 40) if h_onchain is not None else None

    counts = {
        "RSI<30만":              0,
        "Delta+RSI (MA없음)":    0,
        "Delta+RSI+MA200":       0,
        "기존 (MPE+RSI+MA200)":  0,
    }

    for idx in df.index:
        sig       = signal.loc[idx] if idx in signal.index else 0
        mpe_val   = mpe.loc[idx]    if idx in mpe.index    else np.nan
        delta_val = delta_mpe.loc[idx] if idx in delta_mpe.index else np.nan
        ma_val    = ma200.loc[idx]  if idx in ma200.index  else np.nan
        price     = df["close"].loc[idx]

        if np.isnan(mpe_val) or np.isnan(delta_val):
            continue

        rsi_ok    = (sig == 1)
        delta_ok  = (delta_val <= delta_threshold)
        mpe_ok    = (mpe_val   <= mpe_threshold)
        ma_ok     = not np.isnan(ma_val) and price > ma_val

        oc_ok = True
        if oc_threshold is not None and idx in h_onchain.index:
            oc_ok = h_onchain.loc[idx] <= oc_threshold

        if rsi_ok:
            counts["RSI<30만"] += 1
        if rsi_ok and delta_ok:
            counts["Delta+RSI (MA없음)"] += 1
        if rsi_ok and delta_ok and ma_ok:
            counts["Delta+RSI+MA200"] += 1
        if rsi_ok and mpe_ok and ma_ok and oc_ok:
            counts["기존 (MPE+RSI+MA200)"] += 1

    return counts


# ── 메인 분석 ─────────────────────────────────────────────────────────────────
def run_delta_analysis(df, mpe, delta_mpe, h_onchain=None):
    print("=" * 65)
    print("Delta MPE 전략 분석")
    print(f"  Delta 임계값: 하위 {DELTA_PCT}%  |  최대보유: {MAX_HOLD_H}H")
    print(f"  청산: RSI>50 또는 {MAX_HOLD_H}H")
    print("=" * 65)

    # 신호 빈도 먼저 확인
    print("\n[신호 빈도 비교]")
    counts = count_signals(df, mpe, delta_mpe, h_onchain)
    for name, cnt in counts.items():
        bar = "#" * min(cnt // 2, 50)
        print(f"  {name:<25} {cnt:>5}번  {bar}")

    # 전략 실행
    strategies = {
        "delta_only":   ("Delta+RSI (MA없음)",   run_delta_strategy(df, mpe, delta_mpe, use_ma200=False, h_onchain=None)),
        "delta_ma":     ("Delta+RSI+MA200",       run_delta_strategy(df, mpe, delta_mpe, use_ma200=True,  h_onchain=None)),
        "delta_full":   ("Delta+RSI+MA+온체인",   run_delta_strategy(df, mpe, delta_mpe, use_ma200=True,  h_onchain=h_onchain)),
        "original":     ("기존 MPE level 전략",   run_original_strategy(df, mpe, h_onchain)),
    }

    bah = df["close"] / df["close"].iloc[0]

    print("\n" + "=" * 75)
    print(f"{'전략':<27} {'수익률':>9} {'Sharpe':>8} {'최대낙폭':>10} {'최종자산':>10}")
    print("-" * 75)
    results = {}
    for key, (label, eq) in strategies.items():
        m = compute_metrics(eq)
        results[key] = {"label": label, "equity": eq, "metrics": m}
        print(f"  {label:<25} {m['총 수익률']:>9} {m['Sharpe']:>8} "
              f"{m['최대 낙폭']:>10} {m['최종 자산']:>10}")

    bah_m = compute_metrics(bah)
    print(f"  {'BTC 바이앤홀드':<25} {bah_m['총 수익률']:>9} {bah_m['Sharpe']:>8} "
          f"{bah_m['최대 낙폭']:>10} {bah_m['최종 자산']:>10}")
    print("=" * 75)

    results["bah"] = {"label": "BTC 바이앤홀드", "equity": bah, "metrics": bah_m}
    return results, counts


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_delta_results(results, counts, delta_mpe, save=True):
    RESULTS_DIR.mkdir(exist_ok=True)

    COLORS = {
        "delta_only": "#58a6ff",
        "delta_ma":   "#3fb950",
        "delta_full": "#f7931a",
        "original":   "#9945ff",
        "bah":        "#8b949e",
    }

    fig = plt.figure(figsize=(18, 16), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                            height_ratios=[2.2, 1.2, 1.1])

    # ── 1. 누적 수익 곡선 ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    for key, r in results.items():
        lw = 2.2 if key not in ("bah", "original") else 1.2
        alpha = 0.95 if key not in ("bah",) else 0.4
        ls = "--" if key == "bah" else "-"
        m  = r["metrics"]
        ax1.plot(r["equity"].index, r["equity"].values,
                 color=COLORS.get(key, "#8b949e"), linewidth=lw,
                 alpha=alpha, linestyle=ls,
                 label=f"{r['label']}  ({m['총 수익률']}  Sharpe {m['Sharpe']})")

    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("누적 수익 배율", color="#8b949e")
    ax1.set_title(
        "Delta MPE 전략 비교  —  MPE 기울기 신호 vs 기존 MPE level 신호",
        color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")
    _xfmt(ax1)

    # ── 2. Delta MPE 시계열 ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)

    ax2.plot(delta_mpe.index, delta_mpe.values,
             color="#58a6ff", linewidth=0.8, alpha=0.7)

    threshold = np.percentile(delta_mpe.dropna(), DELTA_PCT)
    ax2.axhline(threshold, color="#ff7b72", linewidth=1.2,
                linestyle="--", label=f"하위 {DELTA_PCT}% 임계값 ({threshold:.5f})")
    ax2.axhline(0, color="#8b949e", linewidth=0.6, linestyle=":")

    # 신호 발생 시점 표시
    signal_times = delta_mpe[delta_mpe <= threshold].index
    ax2.scatter(signal_times,
                delta_mpe.loc[signal_times].values,
                color="#ff7b72", s=8, zorder=5, alpha=0.7)

    ax2.set_ylabel("Delta MPE (기울기)", color="#8b949e")
    ax2.set_title(
        f"Delta MPE 시계열  —  빨간 점 = 신호 발생 ({len(signal_times)}개)",
        color="#e6edf3", fontsize=11)
    ax2.legend(fontsize=9, facecolor="#21262d", labelcolor="#e6edf3")
    _xfmt(ax2)

    # ── 3. 신호 빈도 막대 ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)

    names  = list(counts.keys())
    values = list(counts.values())
    colors = ["#8b949e", "#58a6ff", "#3fb950", "#9945ff"]
    bars   = ax3.barh(range(len(names)), values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax3.text(v + 1, bar.get_y() + bar.get_height() / 2,
                 f"{v}번", va="center", color="#e6edf3", fontsize=9)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, color="#8b949e", fontsize=8.5)
    ax3.set_xlabel("진입 횟수 (4년)", color="#8b949e")
    ax3.set_title("필터 조합별 신호 빈도", color="#e6edf3", fontsize=10)
    ax3.xaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. Sharpe 비교 막대 ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)

    keys   = [k for k in results if k != "bah"]
    sharpe = [float(results[k]["metrics"]["Sharpe"]) for k in keys]
    labels = [results[k]["label"] for k in keys]
    s_colors = ["#3fb950" if s > 0 else "#ff7b72" for s in sharpe]

    bars = ax4.bar(range(len(keys)), sharpe, color=s_colors, alpha=0.85)
    for bar, v in zip(bars, sharpe):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002 if v >= 0 else bar.get_height() - 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=9, fontweight="bold")
    ax4.axhline(0, color="#8b949e", linewidth=0.8)
    ax4.set_xticks(range(len(keys)))
    ax4.set_xticklabels(labels, color="#8b949e", fontsize=7.5, rotation=20, ha="right")
    ax4.set_ylabel("Sharpe Ratio", color="#8b949e")
    ax4.set_title("전략별 Sharpe 비교", color="#e6edf3", fontsize=10)
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    fig.suptitle(
        "Delta MPE 전략  —  엔트로피 기울기로 레짐 전환 감지",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        path = RESULTS_DIR / "delta_mpe.png"
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


def _xfmt(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")
