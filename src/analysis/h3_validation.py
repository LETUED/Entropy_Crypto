"""
H3 가설 검증:
"저엔트로피 구간에서 RSI 방향 신호의 정확도가 다른 구간보다 높다"

검증 방법:
1. RSI 14 계산
2. 신호 생성: RSI < 30 → 상승 예측 / RSI > 70 → 하락 예측
3. 이후 24H 실제 방향 확인 (정답 여부)
4. 저엔트로피 구간 vs 기타 구간 정확도 비교
5. 카이제곱 검정 (비율 차이 유의성)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.entropy.calculators import rolling_mpe

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FORWARD_HOURS = 24
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ENTROPY_PCT = 5


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return (100 - 100 / (1 + rs)).rename("rsi")


def generate_signals(rsi: pd.Series) -> pd.Series:
    """
    1  = 상승 예측 (RSI 과매도)
    -1 = 하락 예측 (RSI 과매수)
     0 = 신호 없음
    """
    signal = pd.Series(0, index=rsi.index, name="signal")
    signal[rsi < RSI_OVERSOLD] = 1
    signal[rsi > RSI_OVERBOUGHT] = -1
    return signal


def compute_actual_direction(df: pd.DataFrame, hours: int = FORWARD_HOURS) -> pd.Series:
    """이후 N시간 실제 방향: 1=상승, -1=하락"""
    fwd_return = df["close"].shift(-hours) / df["close"] - 1
    direction = np.sign(fwd_return).rename("actual")
    return direction


def validate_h3(df: pd.DataFrame, mpe_window: int = 168, mpe=None) -> dict:
    print("=" * 60)
    print("H3 가설 검증 시작")
    print(f"RSI {RSI_PERIOD} | 이후 {FORWARD_HOURS}H | 엔트로피 하위 {ENTROPY_PCT}%")
    print("=" * 60)

    rsi = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    actual = compute_actual_direction(df)
    if mpe is None:
        print("MPE 계산 중...")
        mpe = rolling_mpe(df["close"], window=mpe_window)
    else:
        print("MPE 외부 입력 사용")

    # 엔트로피 임계값
    mpe_valid = mpe.dropna()
    threshold = np.percentile(mpe_valid, ENTROPY_PCT)
    low_entropy_mask = mpe <= threshold

    combined = pd.DataFrame({
        "signal": signal,
        "actual": actual,
        "mpe": mpe,
        "low_entropy": low_entropy_mask,
        "rsi": rsi,
    }).dropna()

    # 신호가 있는 구간만
    has_signal = combined["signal"] != 0
    df_signal = combined[has_signal].copy()
    df_signal["correct"] = (df_signal["signal"] == df_signal["actual"]).astype(int)

    # 저엔트로피 vs 기타
    low = df_signal[df_signal["low_entropy"]]
    other = df_signal[~df_signal["low_entropy"]]

    def accuracy_stats(subset):
        if len(subset) == 0:
            return {"n": 0, "accuracy": 0, "n_correct": 0}
        return {
            "n": len(subset),
            "n_correct": subset["correct"].sum(),
            "accuracy": subset["correct"].mean() * 100,
            "acc_up": subset[subset["signal"] == 1]["correct"].mean() * 100,
            "acc_down": subset[subset["signal"] == -1]["correct"].mean() * 100,
            "n_up": (subset["signal"] == 1).sum(),
            "n_down": (subset["signal"] == -1).sum(),
        }

    stats_low = accuracy_stats(low)
    stats_other = accuracy_stats(other)

    # 카이제곱 검정
    contingency = np.array([
        [stats_low["n_correct"], stats_low["n"] - stats_low["n_correct"]],
        [stats_other["n_correct"], stats_other["n"] - stats_other["n_correct"]],
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    result = {
        "low_entropy": stats_low,
        "other": stats_other,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "accuracy_gap": round(stats_low["accuracy"] - stats_other["accuracy"], 2),
        "_data": combined,
        "_df_signal": df_signal,
    }

    _print_h3_results(result)
    return result


def _print_h3_results(r: dict):
    lo = r["low_entropy"]
    ot = r["other"]
    print(f"\n저엔트로피 구간  : {lo['n']:,}개 신호  |  정확도 {lo['accuracy']:.1f}%  "
          f"(상승 {lo['acc_up']:.1f}%  하락 {lo['acc_down']:.1f}%)")
    print(f"기타 구간        : {ot['n']:,}개 신호  |  정확도 {ot['accuracy']:.1f}%  "
          f"(상승 {ot['acc_up']:.1f}%  하락 {ot['acc_down']:.1f}%)")
    print(f"정확도 차이      : {r['accuracy_gap']:+.2f}%p")
    print(f"카이제곱 p-value : {r['p_value']:.6f}  "
          f"{'[O] 유의' if r['significant'] else '[X] 비유의'}")
    verdict = "[H3 채택]" if r["significant"] and r["accuracy_gap"] > 0 else \
              "[부분 채택]" if r["accuracy_gap"] > 0 else "[H3 기각]"
    print(f"\n{verdict}  저엔트로피 구간 RSI 정확도: {lo['accuracy']:.1f}%  vs  기타: {ot['accuracy']:.1f}%")


def plot_h3_results(result: dict, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    combined = result["_data"]
    df_signal = result["_df_signal"]
    lo = result["low_entropy"]
    ot = result["other"]

    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. 정확도 비교 막대그래프 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    ax1.spines[:].set_color("#30363d")

    categories = ["전체", "상승 신호\n(RSI<30)", "하락 신호\n(RSI>70)"]
    low_accs = [lo["accuracy"], lo["acc_up"], lo["acc_down"]]
    other_accs = [ot["accuracy"], ot["acc_up"], ot["acc_down"]]

    x = np.arange(len(categories))
    w = 0.35
    bars1 = ax1.bar(x - w/2, low_accs, w, label=f"저엔트로피 (n={lo['n']:,})",
                    color="#ff4444", alpha=0.85)
    bars2 = ax1.bar(x + w/2, other_accs, w, label=f"기타 구간 (n={ot['n']:,})",
                    color="#58a6ff", alpha=0.85)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 color="#e6edf3", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 color="#e6edf3", fontsize=9)

    ax1.axhline(50, color="#8b949e", linestyle="--", linewidth=1, label="랜덤(50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, color="#8b949e", fontsize=9)
    ax1.set_ylabel("RSI 방향 정확도 (%)", color="#8b949e")
    ax1.set_ylim(0, 80)
    ax1.set_title(f"H3: RSI 정확도 비교\np={result['p_value']:.4f}  차이={result['accuracy_gap']:+.1f}%p",
                  color="#e6edf3", fontsize=11)
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax1.tick_params(colors="#8b949e")
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 2. MPE 구간별 정확도 분포 (구간 세분화) ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    ax2.spines[:].set_color("#30363d")

    mpe_valid = combined[combined["signal"] != 0]["mpe"]
    percentiles = [0, 5, 10, 20, 30, 50, 70, 100]
    labels, accs, ns = [], [], []

    for i in range(len(percentiles) - 1):
        lo_pct = np.percentile(combined["mpe"].dropna(), percentiles[i])
        hi_pct = np.percentile(combined["mpe"].dropna(), percentiles[i+1])
        mask = (df_signal["mpe"] >= lo_pct) & (df_signal["mpe"] < hi_pct)
        subset = df_signal[mask]
        if len(subset) > 5:
            labels.append(f"{percentiles[i]}~{percentiles[i+1]}%")
            accs.append(subset["correct"].mean() * 100)
            ns.append(len(subset))

    colors = ["#ff4444" if i == 0 else "#58a6ff" for i in range(len(labels))]
    bars = ax2.bar(range(len(labels)), accs, color=colors, alpha=0.85)
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"n={n}", ha="center", va="bottom", color="#8b949e", fontsize=7)

    ax2.axhline(50, color="#8b949e", linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30)
    ax2.set_ylabel("RSI 정확도 (%)", color="#8b949e")
    ax2.set_ylim(0, 80)
    ax2.set_title("MPE 구간별 RSI 정확도\n(빨강 = 저엔트로피 구간)", color="#e6edf3", fontsize=11)
    ax2.tick_params(colors="#8b949e")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 3. 시계열: MPE + RSI + 정답/오답 ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#161b22")
    ax3.spines[:].set_color("#30363d")

    # RSI 플롯
    ax3.plot(combined.index, combined["rsi"], color="#e3b341", linewidth=0.7,
             alpha=0.8, label="RSI", zorder=3)
    ax3.axhline(RSI_OVERSOLD, color="#3fb950", linestyle="--", linewidth=1, alpha=0.7)
    ax3.axhline(RSI_OVERBOUGHT, color="#ff4444", linestyle="--", linewidth=1, alpha=0.7)
    ax3.axhline(50, color="#8b949e", linestyle=":", linewidth=0.8, alpha=0.5)

    # 저엔트로피 음영
    low_mask = combined["low_entropy"]
    ax3.fill_between(combined.index, 0, 100,
                     where=low_mask, color="#ff4444", alpha=0.12,
                     transform=ax3.get_xaxis_transform(), label="저엔트로피 구간")

    # 정답/오답 신호 포인트
    correct_pts = df_signal[df_signal["correct"] == 1]
    wrong_pts = df_signal[df_signal["correct"] == 0]
    ax3.scatter(correct_pts.index, correct_pts["rsi"], c="#3fb950", s=12,
                zorder=5, alpha=0.7, label="정답")
    ax3.scatter(wrong_pts.index, wrong_pts["rsi"], c="#ff4444", s=12,
                marker="x", zorder=5, alpha=0.7, label="오답")

    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color="#8b949e")
    ax3.set_title("RSI 신호 정답/오답  —  저엔트로피 구간(빨간 음영)에서 정답 비율 비교",
                  color="#e6edf3", fontsize=11)
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper right", ncol=4)
    ax3.tick_params(colors="#8b949e")
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")
    import matplotlib.dates as mdates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    fig.suptitle("H3 가설 검증: 저엔트로피 구간에서 RSI 방향 정확도가 더 높은가?",
                 color="#e6edf3", fontsize=13, fontweight="bold", y=1.01)

    if save:
        path = RESULTS_DIR / "h3_rsi_accuracy.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()
