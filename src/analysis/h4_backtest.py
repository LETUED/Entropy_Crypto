"""
H4 개선 백테스팅
개선 1: 숏 포지션 추가 (선택적)
개선 2: 추세 필터 (200H MA)
개선 3: 켈리 기준 포지션 사이징
개선 4: 온체인 엔트로피 필터 (펀딩비 + 공포탐욕지수)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.analysis.h3_validation import compute_rsi, generate_signals

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FEE_RATE     = 0.001   # 편도 0.1%
MAX_HOLD_H   = 168     # 최대 보유 168H (1주일)
ENTROPY_PCT  = 10      # 엔트로피 하위 10% 진입
MA_PERIOD    = 200     # 추세 필터용 이동평균


# ── 포지션 사이징: 켈리 기준 ────────────────────────────────────────────────
def kelly_size(mpe_value: float, mpe_series: pd.Series) -> float:
    """
    엔트로피가 낮을수록 더 큰 포지션
    - 하위 1%  → 0.5 (자본 50%)
    - 하위 5%  → 0.3
    - 하위 10% → 0.15
    - 그 외    → 0 (진입 안 함)
    """
    pct1  = np.percentile(mpe_series.dropna(), 1)
    pct5  = np.percentile(mpe_series.dropna(), 5)
    pct10 = np.percentile(mpe_series.dropna(), 10)

    if mpe_value <= pct1:
        return 0.5
    elif mpe_value <= pct5:
        return 0.3
    elif mpe_value <= pct10:
        return 0.15
    return 0.0


# ── 전략 실행 ────────────────────────────────────────────────────────────────
def run_strategy(
    df: pd.DataFrame,
    mpe: pd.Series,
    use_entropy_filter: bool = False,
    use_trend_filter: bool = False,
    use_kelly: bool = False,
    allow_short: bool = False,
    h_onchain: pd.Series = None,   # Phase 2: 온체인 엔트로피
    label: str = "",
) -> pd.Series:

    rsi    = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    ma200  = df["close"].rolling(MA_PERIOD).mean()

    # 온체인 필터 임계값 (하위 40% = 온체인 신호 우호적)
    onchain_ok_mask = pd.Series(True, index=df.index)
    if h_onchain is not None:
        threshold_oc = np.percentile(h_onchain.dropna(), 40)
        onchain_ok_mask = h_onchain <= threshold_oc

    threshold = np.percentile(mpe.dropna(), ENTROPY_PCT)

    equity   = 1.0
    position = 0        # 1=롱, -1=숏, 0=현금
    entry_price = 0.0
    entry_hour  = -999
    equity_curve = []

    for i, idx in enumerate(df.index):
        price   = df["close"].loc[idx]
        sig     = signal.loc[idx]   if idx in signal.index else 0
        mpe_val = mpe.loc[idx]      if idx in mpe.index    else np.nan
        ma_val  = ma200.loc[idx]    if idx in ma200.index  else np.nan
        is_low  = (not np.isnan(mpe_val)) and (mpe_val <= threshold)

        # ── 청산: RSI 회복(롱>50 / 숏<50) 또는 최대 168H ─────────────────
        if position != 0:
            rsi_val    = rsi.loc[idx] if idx in rsi.index else 50.0
            hours_held = i - entry_hour
            should_exit = hours_held >= MAX_HOLD_H
            if position ==  1 and rsi_val > 50: should_exit = True
            if position == -1 and rsi_val < 50: should_exit = True

            if should_exit:
                if position == 1:
                    pnl = (price - entry_price) / entry_price
                else:
                    pnl = (entry_price - price) / entry_price
                equity *= (1 + pnl * pos_size) * (1 - FEE_RATE)
                position = 0

        # ── 진입 판단 ─────────────────────────────────────────────────────
        if position == 0:
            # 켈리 사이징: 엔트로피로 진입 크기 결정
            if use_kelly and not np.isnan(mpe_val):
                pos_size = kelly_size(mpe_val, mpe)
            else:
                pos_size = 0.3 if use_entropy_filter else 1.0

            # 온체인 필터
            oc_ok = onchain_ok_mask.loc[idx] if idx in onchain_ok_mask.index else True

            # 롱 진입 조건
            long_ok = sig == 1
            if use_entropy_filter and not is_low:
                long_ok = False
            if use_trend_filter and (np.isnan(ma_val) or price < ma_val):
                long_ok = False
            if use_kelly and pos_size == 0:
                long_ok = False
            if h_onchain is not None and not oc_ok:
                long_ok = False

            # 숏 진입 조건
            short_ok = allow_short and sig == -1
            if use_entropy_filter and not is_low:
                short_ok = False
            if use_trend_filter and (np.isnan(ma_val) or price > ma_val):
                short_ok = False
            if use_kelly and pos_size == 0:
                short_ok = False
            if h_onchain is not None and not oc_ok:
                short_ok = False

            if long_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
            elif short_ok:
                position    = -1
                entry_price = price * (1 - FEE_RATE)
                entry_hour  = i

        equity_curve.append({"datetime": idx, "equity": equity})

    return pd.DataFrame(equity_curve).set_index("datetime")["equity"]


# ── 성과 지표 ────────────────────────────────────────────────────────────────
def compute_metrics(equity: pd.Series) -> dict:
    ret     = equity.pct_change().dropna()
    total   = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    sharpe  = ret.mean() / (ret.std() + 1e-12) * np.sqrt(24 * 365)
    max_dd  = ((equity - equity.cummax()) / equity.cummax() * 100).min()
    return {
        "총 수익률": f"{total:.2f}%",
        "Sharpe":   f"{sharpe:.3f}",
        "최대 낙폭": f"{max_dd:.2f}%",
        "최종 자산": f"{equity.iloc[-1]:.4f}",
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────
def validate_h4(
    df: pd.DataFrame,
    mpe: pd.Series,
    allow_short: bool = False,
    h_onchain: pd.Series = None,
) -> dict:
    print("=" * 60)
    print("H4 개선 백테스팅")
    print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"수수료: {FEE_RATE*100}%  |  추세필터: 200H MA  |  숏허용: {allow_short}")
    print(f"온체인 필터: {'ON' if h_onchain is not None else 'OFF'}")
    print("=" * 60)

    strategies = {
        "rsi_only": {
            "label": "RSI 단독 (베이스라인)",
            "kwargs": dict(use_entropy_filter=False, use_trend_filter=False,
                           use_kelly=False, allow_short=False),
        },
        "kelly_trend": {
            "label": "켈리+추세+엔트로피 (Phase1)",
            "kwargs": dict(use_entropy_filter=True, use_trend_filter=True,
                           use_kelly=True, allow_short=False),
        },
    }

    # Phase 2: 온체인 필터 추가 전략
    if h_onchain is not None:
        strategies["onchain"] = {
            "label": "Phase2: +온체인 필터",
            "kwargs": dict(use_entropy_filter=True, use_trend_filter=True,
                           use_kelly=True, allow_short=False,
                           h_onchain=h_onchain),
        }

    if allow_short:
        strategies["full"] = {
            "label": "풀 시스템 (켈리+추세+숏)",
            "kwargs": dict(use_entropy_filter=True, use_trend_filter=True,
                           use_kelly=True, allow_short=True,
                           h_onchain=h_onchain),
        }

    results = {}
    for key, cfg in strategies.items():
        print(f"  실행 중: {cfg['label']}...")
        eq = run_strategy(df, mpe, label=cfg["label"], **cfg["kwargs"])
        results[key] = {
            "label":   cfg["label"],
            "equity":  eq,
            "metrics": compute_metrics(eq),
        }

    # 바이앤홀드
    bah = df["close"] / df["close"].iloc[0]
    results["bah"] = {
        "label":   "BTC 바이앤홀드",
        "equity":  bah,
        "metrics": compute_metrics(bah),
    }

    _print_results(results)
    return results


def _print_results(results: dict):
    keys = [k for k in results if k != "bah"] + ["bah"]
    header = f"{'지표':<16}" + "".join(f"{results[k]['label']:>22}" for k in keys)
    print("\n" + "=" * (16 + 22 * len(keys)))
    print(header)
    print("-" * (16 + 22 * len(keys)))
    for metric in ["총 수익률", "Sharpe", "최대 낙폭", "최종 자산"]:
        row = f"{metric:<16}"
        for k in keys:
            row += f"{results[k]['metrics'].get(metric, '-'):>22}"
        print(row)
    print("=" * (16 + 22 * len(keys)))


# ── 시각화 ───────────────────────────────────────────────────────────────────
COLORS = {
    "rsi_only":      "#58a6ff",
    "entropy_rsi":   "#e3b341",
    "trend_entropy": "#bc8cff",
    "kelly_trend":   "#3fb950",
    "onchain":       "#ff7b72",
    "full":          "#f0a500",
    "bah":           "#f7931a",
}


def plot_h4_results(results: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                            height_ratios=[2.5, 1.2, 1])

    # ── 1. 누적 수익 곡선 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    highlight = {"kelly_trend", "onchain", "full", "bah"}
    for key, r in results.items():
        lw    = 2.0 if key in highlight else 1.0
        alpha = 1.0 if key in highlight else 0.6
        ax1.plot(r["equity"].index, r["equity"].values,
                 color=COLORS.get(key, "#8b949e"), linewidth=lw, alpha=alpha,
                 label=f"{r['label']}  ({r['metrics']['총 수익률']})")

    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("누적 수익 배율", color="#8b949e")
    ax1.set_title("전략별 누적 수익 비교  (수수료 0.1% 포함)", color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")
    _xfmt(ax1)

    # ── 2. 낙폭 비교 ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)

    show_dd = [k for k in ["rsi_only", "kelly_trend", "onchain", "full"] if k in results]

    for key in show_dd:
        r  = results[key]
        dd = (r["equity"] - r["equity"].cummax()) / r["equity"].cummax() * 100
        ax2.fill_between(dd.index, dd, 0, color=COLORS[key], alpha=0.35,
                         label=f"{r['label']} (최대 {r['metrics']['최대 낙폭']})")

    ax2.set_ylabel("낙폭 (%)", color="#8b949e")
    ax2.set_title("최대 낙폭 비교", color="#e6edf3", fontsize=11)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")
    _xfmt(ax2)

    # ── 3. 성과 테이블 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    ax3.axis("off")

    metric_keys = ["총 수익률", "Sharpe", "최대 낙폭", "최종 자산"]
    col_labels  = ["지표"] + [r["label"] for r in results.values()]
    table_data  = []
    for mk in metric_keys:
        row = [mk] + [r["metrics"].get(mk, "-") for r in results.values()]
        table_data.append(row)

    tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax3.set_title("성과 지표 요약", color="#e6edf3", fontsize=10, pad=8)

    # ── 4. 월별 수익률 ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)

    # 최종 전략 vs RSI 단독 비교
    best_key = "full" if "full" in results else "kelly_trend"
    best_eq  = results[best_key]["equity"]
    rsi_eq   = results["rsi_only"]["equity"]

    monthly_best = best_eq.resample("ME").last().pct_change().dropna() * 100
    monthly_rsi  = rsi_eq.resample("ME").last().pct_change().dropna() * 100

    x = np.arange(len(monthly_best))
    ax4.bar(x, monthly_best.values,
            color=np.where(monthly_best.values >= 0, "#3fb950", "#ff4444"),
            alpha=0.85, width=0.5, label=results[best_key]["label"])
    ax4.plot(x, monthly_rsi.reindex(monthly_best.index).values,
             color="#58a6ff", linewidth=1.2, marker="o", markersize=3,
             alpha=0.7, label="RSI 단독")
    ax4.axhline(0, color="#8b949e", linewidth=0.8)

    step = max(1, len(monthly_best) // 8)
    ax4.set_xticks(x[::step])
    ax4.set_xticklabels(monthly_best.index[::step].strftime("%y-%m"),
                        rotation=45, ha="right", color="#8b949e", fontsize=7)
    ax4.set_ylabel("월 수익률 (%)", color="#8b949e", fontsize=9)
    ax4.set_title("월별 수익률 비교", color="#e6edf3", fontsize=10)
    ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")
    ax4.tick_params(colors="#8b949e")
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    fig.suptitle("H4 개선 백테스팅: 단계별 전략 비교",
                 color="#e6edf3", fontsize=14, fontweight="bold", y=1.01)

    if save:
        path = RESULTS_DIR / "h4_improved_backtest.png"
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
             color="#8b949e")
