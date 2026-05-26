"""
성과 분석: 연도별 / 월별 / 낙폭 회복 분석
핵심 지표: 원금 보존, Sharpe, 승률, 낙폭 회복 속도
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def analyze(results: dict) -> dict:
    """연도별 / 월별 성과 계산"""
    analysis = {}

    for key, r in results.items():
        eq = r["equity"]

        # 월별 수익률
        monthly = eq.resample("ME").last().pct_change().dropna() * 100

        # 연도별 수익률
        yearly = eq.resample("YE").last().pct_change().dropna() * 100

        # 승률 (월 기준)
        win_rate = (monthly > 0).mean() * 100

        # 낙폭 회복 시간 (시간 단위)
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        in_dd = dd < -1.0
        recovery_hours = _avg_recovery_hours(in_dd)

        analysis[key] = {
            "label":          r["label"],
            "monthly":        monthly,
            "yearly":         yearly,
            "win_rate":       round(win_rate, 1),
            "best_month":     round(monthly.max(), 2),
            "worst_month":    round(monthly.min(), 2),
            "avg_monthly":    round(monthly.mean(), 2),
            "recovery_hours": recovery_hours,
            "drawdown":       dd,
        }

    return analysis


def _avg_recovery_hours(in_drawdown: pd.Series) -> float:
    """평균 낙폭 지속 시간 계산"""
    durations = []
    count = 0
    for val in in_drawdown:
        if val:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    return round(np.mean(durations), 1) if durations else 0.0


def plot_performance(analysis: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    show_keys = [k for k in analysis if k != "bah"]
    bah       = analysis.get("bah")

    fig = plt.figure(figsize=(18, 16), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    COLORS = {
        "rsi_only":      "#58a6ff",
        "entropy_rsi":   "#e3b341",
        "trend_entropy": "#bc8cff",
        "kelly_trend":   "#3fb950",
        "full":          "#ff7b72",
        "bah":           "#f7931a",
    }

    # ── 1. 연도별 수익률 비교 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    years      = [2021, 2022, 2023, 2024]
    n_strats   = len(show_keys)
    group_w    = 0.7
    bar_w      = group_w / (n_strats + 1)
    x          = np.arange(len(years))

    for i, key in enumerate(show_keys):
        a    = analysis[key]
        vals = [_year_return(a["yearly"], y) for y in years]
        offset = (i - n_strats / 2) * bar_w + bar_w / 2
        bars = ax1.bar(x + offset, vals, bar_w,
                       color=COLORS.get(key, "#8b949e"), alpha=0.85,
                       label=a["label"])
        for bar, v in zip(bars, vals):
            if abs(v) > 1:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + (1.5 if v >= 0 else -3.5),
                         f"{v:.1f}%", ha="center", va="bottom",
                         color="#e6edf3", fontsize=7.5, fontweight="bold")

    # 바이앤홀드 꺾은선
    if bah:
        bah_vals = [_year_return(bah["yearly"], y) for y in years]
        ax1.plot(x, bah_vals, color="#f7931a", linewidth=2,
                 marker="D", markersize=7, label="BTC 바이앤홀드", zorder=5)
        for xi, v in zip(x, bah_vals):
            ax1.text(xi, v + 3, f"{v:.1f}%", ha="center",
                     color="#f7931a", fontsize=8, fontweight="bold")

    ax1.axhline(0, color="#8b949e", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in years], color="#e6edf3", fontsize=11)
    ax1.set_ylabel("연간 수익률 (%)", color="#8b949e")
    ax1.set_title("연도별 수익률  —  2022년 하락장에서 얼마나 버텼나",
                  color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=3)
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 2. 월별 승률 비교 ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)

    labels    = [analysis[k]["label"] for k in show_keys]
    winrates  = [analysis[k]["win_rate"] for k in show_keys]
    colors_wr = [COLORS.get(k, "#8b949e") for k in show_keys]

    bars = ax2.barh(range(len(labels)), winrates, color=colors_wr, alpha=0.85)
    for bar, v in zip(bars, winrates):
        ax2.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", color="#e6edf3", fontsize=9)

    ax2.axvline(50, color="#8b949e", linestyle="--", linewidth=1, label="50% 기준선")
    if bah:
        ax2.axvline(bah["win_rate"], color="#f7931a", linestyle=":",
                    linewidth=1.5, label=f"바이앤홀드 {bah['win_rate']}%")

    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, color="#e6edf3", fontsize=8)
    ax2.set_xlabel("월별 승률 (%)", color="#8b949e")
    ax2.set_title("월별 승률  (50% 이상 = 수익 달이 더 많음)",
                  color="#e6edf3", fontsize=10)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")
    ax2.set_xlim(0, 100)
    ax2.xaxis.grid(True, color="#21262d", linestyle="--")

    # ── 3. 월별 수익률 분포 (최고/평균/최저) ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)

    x3 = np.arange(len(show_keys))
    for i, key in enumerate(show_keys):
        a = analysis[key]
        ax3.plot([i, i], [a["worst_month"], a["best_month"]],
                 color=COLORS.get(key, "#8b949e"), linewidth=3, alpha=0.5)
        ax3.scatter(i, a["avg_monthly"], color=COLORS.get(key, "#8b949e"),
                    s=80, zorder=5)
        ax3.scatter(i, a["best_month"], color="#3fb950", s=40, marker="^", zorder=5)
        ax3.scatter(i, a["worst_month"], color="#ff4444", s=40, marker="v", zorder=5)
        ax3.text(i, a["avg_monthly"] + 0.5, f"{a['avg_monthly']:+.1f}%",
                 ha="center", color="#e6edf3", fontsize=8)

    ax3.axhline(0, color="#8b949e", linestyle="--", linewidth=1)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([analysis[k]["label"] for k in show_keys],
                        color="#e6edf3", fontsize=7.5, rotation=15, ha="right")
    ax3.set_ylabel("월 수익률 (%)", color="#8b949e")
    ax3.set_title("월별 수익률 범위  (▲최고 / ● 평균 / ▼최악)",
                  color="#e6edf3", fontsize=10)
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. 2022년 집중 분석: 하락장 방어력 ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    _style(ax4)

    for key in show_keys:
        r  = analysis[key]
        dd = r["drawdown"]
        dd_2022 = dd["2022-01":"2022-12"]
        ax4.fill_between(dd_2022.index, dd_2022, 0,
                         color=COLORS.get(key, "#8b949e"), alpha=0.3,
                         label=f"{r['label']} (최대 {dd_2022.min():.1f}%)")

    if bah:
        dd_bah_2022 = bah["drawdown"]["2022-01":"2022-12"]
        ax4.plot(dd_bah_2022.index, dd_bah_2022,
                 color="#f7931a", linewidth=1.5, linestyle="--",
                 label=f"BTC 바이앤홀드 (최대 {dd_bah_2022.min():.1f}%)")

    ax4.set_ylabel("낙폭 (%)", color="#8b949e")
    ax4.set_title("2022년 하락장 낙폭 상세  —  바이앤홀드 대비 방어력",
                  color="#e6edf3", fontsize=11)
    ax4.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="lower left")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right",
             color="#8b949e")
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    fig.suptitle("성과 분석: 수익률보다 안정성 — 잃지 않는 것이 버는 것",
                 color="#e6edf3", fontsize=14, fontweight="bold", y=1.01)

    if save:
        path = RESULTS_DIR / "performance_analysis.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def print_summary(analysis: dict):
    print("\n" + "=" * 70)
    print("성과 요약: 안정성 중심 지표")
    print("=" * 70)
    print(f"{'전략':<28} {'승률':>6} {'평균월수익':>10} {'최악의달':>10} {'낙폭회복':>10}")
    print("-" * 70)
    for key, a in analysis.items():
        print(f"{a['label']:<28} {a['win_rate']:>5.1f}%"
              f" {a['avg_monthly']:>+9.2f}%"
              f" {a['worst_month']:>+9.2f}%"
              f" {a['recovery_hours']:>8.0f}H")
    print("=" * 70)


def _year_return(yearly: pd.Series, year: int) -> float:
    matches = yearly[yearly.index.year == year]
    return round(matches.iloc[0], 2) if len(matches) > 0 else 0.0


def _style(ax):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
