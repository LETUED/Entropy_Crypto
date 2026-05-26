"""
H2 검증 결과 시각화 (개선판)
- Malgun Gothic 폰트로 한글 렌더링
- 다크 테마
- MPE y축 실제 범위로 줌인
- BBW/ATR 패널 분리
- 주요 이벤트 어노테이션
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# ── 한글 폰트 설정 (Windows: Malgun Gothic) ──────────────────────────────────
def _setup_font():
    font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()

# 주요 시장 이벤트
EVENTS = {
    "2021-04-14": ("BTC ATH\n$65k", "top"),
    "2021-11-10": ("BTC ATH\n$69k", "top"),
    "2022-05-12": ("LUNA\n붕괴", "bottom"),
    "2022-11-11": ("FTX\n파산", "bottom"),
    "2024-01-11": ("BTC ETF\n승인", "top"),
    "2024-11-22": ("BTC ATH\n$99k", "top"),
}


def plot_h2_results(results: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    data: pd.DataFrame = results["_data"].copy()

    mpe_valid = data["mpe"].dropna()
    threshold = np.percentile(mpe_valid, 5)
    low_mask = data["mpe"] <= threshold

    # ── 레이아웃: 5 패널 ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16), facecolor="#0d1117")
    gs = fig.add_gridspec(5, 1, hspace=0.08,
                          height_ratios=[2.5, 1.5, 1.5, 1, 1])

    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines[:].set_color("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.6, linestyle="--")
        ax.grid(axis="x", color="#21262d", linewidth=0.4, linestyle=":")

    fig.suptitle("H2 가설 검증: 저엔트로피 구간 vs 이후 24H 변동폭\n"
                 "BTC/USDT 1H  |  2021.01 ~ 2025.01  |  MPE 하위 5% 구간 (빨간 음영)",
                 fontsize=13, color="#e6edf3", fontweight="bold", y=0.98)

    # ── 패널 1: BTC 가격 ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(data.index, data["close"], color="#f7931a", linewidth=0.9, zorder=3)
    ax1.fill_between(data.index, data["close"].min(), data["close"],
                     alpha=0.08, color="#f7931a")
    _shade_low_entropy(ax1, data.index, low_mask)
    ax1.set_ylabel("BTC (USDT)", color="#8b949e", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(axis="y", colors="#8b949e")
    ax1.tick_params(axis="x", labelbottom=False)
    _add_events(ax1, data)

    # ── 패널 2: MPE (y축 실제 범위로 줌인) ──────────────────────────────────
    ax2 = axes[1]
    ax2.plot(data.index, data["mpe"], color="#58a6ff", linewidth=0.8, zorder=3, label="MPE")
    ax2.axhline(threshold, color="#ff4444", linestyle="--", linewidth=1.2,
                label=f"하위 5% 임계값 ({threshold:.3f})", zorder=4)
    _shade_low_entropy(ax2, data.index, low_mask)

    mpe_min = mpe_valid.min()
    mpe_max = mpe_valid.max()
    margin = (mpe_max - mpe_min) * 0.15
    ax2.set_ylim(mpe_min - margin, mpe_max + margin)
    ax2.set_ylabel("MPE", color="#8b949e", fontsize=10)
    ax2.tick_params(axis="x", labelbottom=False)

    r_mpe = results.get("mpe", {})
    legend_text = (f"배율: {r_mpe.get('ratio', 0):.2f}x  |  "
                   f"p={r_mpe.get('p_value', 1):.2e}  |  "
                   f"n저엔트로피={r_mpe.get('n_low_entropy', 0):,}")
    ax2.legend(loc="lower left", fontsize=8, facecolor="#21262d",
               edgecolor="#30363d", labelcolor="#e6edf3", title=legend_text,
               title_fontsize=7)

    # ── 패널 3: 이후 24H 변동폭 ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(data.index, data["fwd_vol"], color="#3fb950", linewidth=0.6,
             alpha=0.7, zorder=3)
    # 저엔트로피 구간의 변동폭을 강조
    fwd_low = data["fwd_vol"].where(low_mask)
    ax3.fill_between(data.index, 0, fwd_low, alpha=0.6, color="#ff4444",
                     label="저엔트로피 구간 변동폭", zorder=4)
    _shade_low_entropy(ax3, data.index, low_mask, alpha=0.1)
    ax3.set_ylabel("이후 24H\n변동폭 (%)", color="#8b949e", fontsize=9)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.legend(loc="upper right", fontsize=8, facecolor="#21262d",
               edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 패널 4: BBW ──────────────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(data.index, data["bbw"], color="#bc8cff", linewidth=0.7, alpha=0.85)
    _shade_low_entropy(ax4, data.index, low_mask)
    r_bbw = results.get("bbw", {})
    ax4.set_ylabel("BBW", color="#8b949e", fontsize=9)
    ax4.tick_params(axis="x", labelbottom=False)
    _add_result_text(ax4, r_bbw)

    # ── 패널 5: ATR% ─────────────────────────────────────────────────────────
    ax5 = axes[4]
    ax5.plot(data.index, data["atr"], color="#e3b341", linewidth=0.7, alpha=0.85)
    _shade_low_entropy(ax5, data.index, low_mask)
    r_atr = results.get("atr", {})
    ax5.set_ylabel("ATR (%)", color="#8b949e", fontsize=9)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")
    _add_result_text(ax5, r_atr)

    # 공통 x축 연결
    for ax in axes[:-1]:
        ax.sharex(axes[-1])

    if save:
        path = RESULTS_DIR / "h2_timeseries.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def plot_distribution_comparison(results: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    data: pd.DataFrame = results["_data"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#0d1117")
    fig.suptitle("저엔트로피 구간 vs 기타 구간  —  이후 24H 변동폭 분포 비교",
                 fontsize=13, color="#e6edf3", fontweight="bold")

    indicators = [
        ("mpe", "MPE (순열 엔트로피)", True, "#58a6ff"),
        ("bbw", "BBW (볼린저 밴드 수축)", True, "#bc8cff"),
        ("atr", "ATR% (평균진폭)", True, "#e3b341"),
    ]

    for ax, (col, label, low_is_low, color) in zip(axes, indicators):
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=9)

        combined = data[[col, "fwd_vol"]].dropna()
        cutoff = np.percentile(combined[col], 5 if low_is_low else 95)
        mask = combined[col] <= cutoff if low_is_low else combined[col] >= cutoff

        low_vol = combined.loc[mask, "fwd_vol"]
        other_vol = combined.loc[~mask, "fwd_vol"]

        # 분포 곡선
        sns.kdeplot(other_vol, ax=ax, color="#8b949e", fill=True, alpha=0.25,
                    label=f"기타 구간 (n={len(other_vol):,})")
        sns.kdeplot(low_vol, ax=ax, color=color, fill=True, alpha=0.45,
                    label=f"저신호 구간 (n={len(low_vol):,})")

        # 중앙값 수직선
        med_low = np.median(low_vol)
        med_other = np.median(other_vol)
        ax.axvline(med_low, color=color, linestyle="--", linewidth=1.8,
                   label=f"중앙값 {med_low:.2f}%")
        ax.axvline(med_other, color="#8b949e", linestyle="--", linewidth=1.4,
                   label=f"중앙값 {med_other:.2f}%")

        r = results.get(col, {})
        ratio = r.get("ratio", med_low / med_other if med_other else 0)
        p = r.get("p_value", 1.0)
        sig = "유의" if p < 0.05 else "비유의"

        ax.set_title(f"{label}\n배율: {ratio:.2f}x  |  p={p:.2e}  ({sig})",
                     fontsize=10, color="#e6edf3", pad=8)
        ax.set_xlabel("이후 24H 변동폭 (%)", color="#8b949e", fontsize=9)
        ax.set_ylabel("밀도", color="#8b949e", fontsize=9)
        ax.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#e6edf3", loc="upper right")

    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "h2_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def plot_summary_table(results: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for key in ["mpe", "bbw", "atr"]:
        r = results.get(key, {})
        if "error" in r:
            continue
        rows.append({
            "지표": r["label"],
            "저엔트로피 n": r["n_low_entropy"],
            "중앙값 배율": f"{r['ratio']:.2f}x",
            "평균 배율": f"{r['mean_ratio']:.2f}x",
            "p-value": f"{r['p_value']:.2e}",
            "유의성": "[O]" if r["significant"] else "[X]",
        })

    df = pd.DataFrame(rows)
    print("\n결과 요약표:")
    print(df.to_string(index=False))

    if save:
        path = RESULTS_DIR / "h2_summary.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path}")


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _shade_low_entropy(ax, index, mask, alpha=0.18):
    ax.fill_between(index, *ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else (0, 1),
                    where=mask, color="#ff4444", alpha=alpha,
                    transform=ax.get_xaxis_transform(), zorder=2)


def _add_events(ax, data):
    ymin, ymax = data["close"].min(), data["close"].max()
    for date_str, (label, pos) in EVENTS.items():
        try:
            dt = pd.Timestamp(date_str)
            if dt not in data.index:
                dt = data.index[data.index.searchsorted(dt)]
            price = data.loc[dt, "close"] if dt in data.index else None
            if price is None:
                continue
            y = ymax * 0.92 if pos == "top" else ymin * 1.15
            ax.annotate(label, xy=(dt, price), xytext=(dt, y),
                        fontsize=6.5, color="#8b949e", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#30363d", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#21262d",
                                  edgecolor="#30363d", alpha=0.8))
        except Exception:
            pass


def _add_result_text(ax, r: dict):
    if not r or "error" in r:
        return
    text = f"배율 {r.get('ratio', 0):.2f}x  p={r.get('p_value', 1):.1e}"
    ax.text(0.01, 0.92, text, transform=ax.transAxes,
            fontsize=8, color="#8b949e",
            bbox=dict(facecolor="#21262d", edgecolor="none", alpha=0.7))
