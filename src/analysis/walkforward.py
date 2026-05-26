"""
Walk-forward 검증
- 학습 기간에서 임계값 계산 (in-sample)
- 테스트 기간에서 전략 실행 (out-of-sample)
- 모든 구간에서 일정한 성과 = 공식이 시장에 의존하지 않음 (타임리스)
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


def _setup_font():
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()


def _pct_val(s):
    return float(str(s).replace("%", "").strip())


def _run_on_window(df_test, mpe_test, hoc_test,
                   mpe_threshold, oc_threshold, mpe_train,
                   market_filter=None):
    """
    단일 테스트 윈도우 전략 실행
    임계값(mpe_threshold, oc_threshold)은 학습 기간에서 계산된 값을 그대로 사용
    market_filter: boolean Series — False인 시점에는 신규 진입 금지 (BTC MA200 필터 등)
    """
    rsi    = compute_rsi(df_test["close"])
    signal = generate_signals(rsi)
    ma200  = df_test["close"].rolling(MA_PERIOD).mean()

    equity     = 1.0
    position   = 0
    entry_price = 0.0
    entry_hour  = -999
    pos_size    = 0.3
    curve       = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        sig     = signal.loc[idx]  if idx in signal.index  else 0
        mpe_val = mpe_test.loc[idx] if idx in mpe_test.index else np.nan
        ma_val  = ma200.loc[idx]   if idx in ma200.index   else np.nan

        is_low = (not np.isnan(mpe_val)) and (mpe_val <= mpe_threshold)

        oc_ok = True
        if hoc_test is not None and oc_threshold is not None:
            if idx in hoc_test.index:
                oc_ok = hoc_test.loc[idx] <= oc_threshold

        # 포트폴리오 레벨 시장 필터 (BTC MA200 등)
        mkt_ok = True
        if market_filter is not None and idx in market_filter.index:
            mkt_ok = bool(market_filter.loc[idx])

        # 청산: RSI 회복(>50) 또는 최대 168H (청산은 필터 무관하게 실행)
        if position == 1:
            rsi_val = rsi.loc[idx] if idx in rsi.index else 50.0
            if rsi_val > 50 or (i - entry_hour) >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * pos_size) * (1 - FEE_RATE)
                position = 0

        # 진입 (임계값은 학습 기간 기준, Kelly 분포도 학습 기간 기준)
        if position == 0 and mkt_ok:
            mpe_safe = mpe_val if not np.isnan(mpe_val) else 1.0
            pos_size = kelly_size(mpe_safe, mpe_train)

            if (sig == 1 and is_low
                    and not np.isnan(ma_val) and price > ma_val
                    and pos_size > 0 and oc_ok):
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i

        curve.append(equity)

    return pd.Series(curve, index=df_test.index)


def run_walkforward(df, mpe, h_onchain=None,
                    train_months=12, test_months=6, step_months=6,
                    market_filter=None):
    """
    Walk-forward 검증 실행

    Parameters
    ----------
    train_months : 학습 윈도우 크기 (임계값 계산)
    test_months  : 테스트 윈도우 크기 (실제 전략 실행)
    step_months  : 윈도우 이동 간격
    """
    results = []

    train_delta = pd.DateOffset(months=train_months)
    test_delta  = pd.DateOffset(months=test_months)
    step_delta  = pd.DateOffset(months=step_months)

    cursor = df.index[0]
    end    = df.index[-1]

    while True:
        train_start = cursor
        train_end   = cursor + train_delta
        test_start  = train_end
        test_end    = test_start + test_delta

        if test_end > end:
            break

        # ── 학습 기간 기반 임계값 계산 (핵심) ───────────────────────────
        mpe_train = mpe[train_start:train_end].dropna()
        if len(mpe_train) < 200:
            cursor += step_delta
            continue

        mpe_threshold = np.percentile(mpe_train, ENTROPY_PCT)

        oc_threshold = None
        if h_onchain is not None:
            hoc_train = h_onchain[train_start:train_end].dropna()
            if len(hoc_train) > 0:
                oc_threshold = np.percentile(hoc_train, 40)

        # ── 테스트 기간 실행 (out-of-sample) ────────────────────────────
        df_test  = df[test_start:test_end]
        mpe_test = mpe[test_start:test_end]
        hoc_test = h_onchain[test_start:test_end] if h_onchain is not None else None
        mkt_test = market_filter[test_start:test_end] if market_filter is not None else None

        if len(df_test) < 168:
            cursor += step_delta
            continue

        equity = _run_on_window(df_test, mpe_test, hoc_test,
                                mpe_threshold, oc_threshold, mpe_train,
                                market_filter=mkt_test)

        bah       = df_test["close"] / df_test["close"].iloc[0]
        label     = (f"{test_start.strftime('%Y-%m')} ~ "
                     f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")
        short_lbl = test_start.strftime("%Y-%m")

        results.append({
            "label":         label,
            "short_label":   short_lbl,
            "test_start":    test_start,
            "test_end":      test_end,
            "equity":        equity,
            "bah":           bah,
            "metrics":       compute_metrics(equity),
            "bah_return":    float(bah.iloc[-1] - 1) * 100,
            "mpe_threshold": mpe_threshold,
        })

        cursor += step_delta

    return results


def print_walkforward(results):
    sharpes = [_pct_val(r["metrics"]["Sharpe"]) for r in results]

    print("\n" + "=" * 95)
    print(f"{'테스트 기간':<30} {'수익률':>9} {'BaH':>9} "
          f"{'Sharpe':>8} {'최대낙폭':>10} {'MPE임계':>10}")
    print("-" * 95)
    for r, s in zip(results, sharpes):
        m      = r["metrics"]
        marker = "★" if s > 0 else "▼"
        print(f"{marker} {r['label']:<28} {m['총 수익률']:>9} "
              f"{r['bah_return']:>+8.1f}% {m['Sharpe']:>8} "
              f"{m['최대 낙폭']:>10} {r['mpe_threshold']:>10.4f}")
    print("=" * 95)
    print(f"  평균 Sharpe : {np.mean(sharpes):.3f}  |  "
          f"표준편차 : {np.std(sharpes):.3f}  |  "
          f"양수 비율 : {sum(s > 0 for s in sharpes)}/{len(sharpes)}")


def plot_walkforward(results, save=True):
    RESULTS_DIR.mkdir(exist_ok=True)
    n = len(results)

    sharpes  = [_pct_val(r["metrics"]["Sharpe"])   for r in results]
    mdd_vals = [_pct_val(r["metrics"]["최대 낙폭"]) for r in results]
    ret_vals = [_pct_val(r["metrics"]["총 수익률"]) for r in results]
    bah_vals = [r["bah_return"]                     for r in results]
    labels   = [r["short_label"]                    for r in results]

    colors = plt.cm.plasma(np.linspace(0.15, 0.88, n))

    fig = plt.figure(figsize=(18, 15), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32,
                            height_ratios=[2.2, 1.1, 1.1])

    # ── 1. 각 윈도우 누적 수익 (정규화, 1.0 시작) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    for i, r in enumerate(results):
        eq  = r["equity"]
        bah = r["bah"]
        s   = sharpes[i]
        ax1.plot(eq.index, eq.values,
                 color=colors[i], linewidth=1.9, alpha=0.95,
                 label=f"{r['label']}  Sharpe {s:.3f}")
        ax1.plot(bah.index, bah.values,
                 color=colors[i], linewidth=0.7, alpha=0.28, linestyle="--")

    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("누적 수익 배율", color="#8b949e")
    ax1.set_title(
        "Walk-forward — 각 Out-of-Sample 구간  (실선=전략 / 점선=BaH)",
        color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. 기간별 Sharpe ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)

    bar_c = ["#3fb950" if s > 0 else "#ff7b72" for s in sharpes]
    bars  = ax2.bar(range(n), sharpes, color=bar_c, alpha=0.88)
    for bar, v in zip(bars, sharpes):
        ypos = bar.get_height() + 0.003 if v >= 0 else bar.get_height() - 0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{v:.3f}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=8.5, fontweight="bold")

    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.axhline(np.mean(sharpes), color="#f7931a", linewidth=1.3,
                linestyle="--", label=f"평균 {np.mean(sharpes):.3f}")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")
    ax2.set_title("기간별 Sharpe  ← 핵심 일관성 지표", color="#e6edf3", fontsize=10)
    ax2.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 3. 기간별 최대 낙폭 ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)

    ax3.bar(range(n), mdd_vals, color="#58a6ff", alpha=0.78)
    ax3.axhline(np.mean(mdd_vals), color="#f7931a", linewidth=1.3,
                linestyle="--", label=f"평균 {np.mean(mdd_vals):.2f}%")
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("최대 낙폭 (%)", color="#8b949e")
    ax3.set_title("기간별 최대 낙폭  ← 리스크 일관성", color="#e6edf3", fontsize=10)
    ax3.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. 전략 수익률 vs BaH ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style(ax4)

    x, w = np.arange(n), 0.35
    ax4.bar(x - w/2, ret_vals, w, color="#3fb950", alpha=0.88, label="전략")
    ax4.bar(x + w/2, bah_vals, w, color="#f7931a", alpha=0.45, label="BaH")
    ax4.axhline(0, color="#8b949e", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30, ha="right")
    ax4.set_ylabel("수익률 (%)", color="#8b949e")
    ax4.set_title("기간별 전략 vs BaH 수익률", color="#e6edf3", fontsize=10)
    ax4.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 5. 일관성 요약 테이블 ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style(ax5)
    ax5.axis("off")

    pos_sharpe = sum(s > 0 for s in sharpes)
    beat_bah   = sum(r >= b for r, b in zip(ret_vals, bah_vals))

    rows = [
        ["테스트 윈도우 수",   f"{n}개"],
        ["평균 Sharpe",       f"{np.mean(sharpes):.3f}"],
        ["Sharpe 표준편차",   f"{np.std(sharpes):.3f}  ← 낮을수록 일관"],
        ["양수 Sharpe 비율",  f"{pos_sharpe}/{n}  ({pos_sharpe/n*100:.0f}%)"],
        ["평균 최대낙폭",      f"{np.mean(mdd_vals):.2f}%"],
        ["평균 수익률",        f"{np.mean(ret_vals):.2f}%"],
        ["BaH 초과 구간",     f"{beat_bah}/{n}  ({beat_bah/n*100:.0f}%)"],
    ]

    tbl = ax5.table(
        cellText=rows,
        colLabels=["지표", "값"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax5.set_title("일관성 요약", color="#e6edf3", fontsize=10, pad=8)

    fig.suptitle(
        "Walk-forward 검증  —  공식이 시간에 걸쳐 일정하게 통하는가?",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        path = RESULTS_DIR / "walkforward.png"
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
