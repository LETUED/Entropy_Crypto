"""
엔트로피 레짐 기반 복합 전략

레짐 정의:
    저엔트로피 (MPE < 10th pct) : 시장이 질서 있음 -> 역추세 (mean reversion)
        Long  : RSI < 30 + MA200 위
        Short : RSI > 70 + MA200 아래
        Exit  : RSI crosses 50 OR max 168H

    고엔트로피 (MPE > 70th pct) : 시장이 무질서/방향성 -> 모멘텀 (trend following)
        Long  : price > MA20 + MA20 slope 양수 + MA200 위
        Short : price < MA20 + MA20 slope 음수 + MA200 아래
        Exit  : price MA20 반대편 크로스 OR max 72H

    중간 구간 : 관망

이론적 근거:
    저엔트로피 = 패턴이 반복되는 질서 있는 상태 -> 극값에서 반전 예상
    고엔트로피 = 정보가 빠르게 소화되는 상태 -> 추세에 올라타기
    (Permutation Transition Entropy 2020 / Hidden Order in Trades 2025)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import compute_metrics, kelly_size, MA_PERIOD, MAX_HOLD_H, FEE_RATE, ENTROPY_PCT

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

LOW_PCT       = 10    # 저엔트로피 상한 (하위 10%)
HIGH_PCT      = 70    # 고엔트로피 하한 (상위 30%)
MA_FAST       = 20    # 모멘텀용 단기 이평
MA_SLOW       = MA_PERIOD  # 추세 필터 MA200
MAX_MOM_H     = 168   # 모멘텀 최대 보유 168H (Donchian 스케일과 동일)
POS_SIZE      = 0.15  # 고정 포지션 (비교용), Kelly는 별도
POS_SIZE_MOM  = 0.10  # 고엔트로피 모멘텀 고정 포지션 (승률 낮아 보수적)
BREAKOUT_BARS = 168   # N봉 신고가/신저가 돌파 (= 168H = 1주일)


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def compute_ma_slope(series: pd.Series, window: int, slope_window: int = 5) -> pd.Series:
    """이동평균의 기울기 (양수=상승, 음수=하락)"""
    ma = series.rolling(window).mean()
    return ma.diff(slope_window)


def run_regime_strategy(
    df: pd.DataFrame,
    mpe: pd.Series,
    h_onchain: pd.Series = None,
    allow_short: bool = True,
    use_kelly: bool = False,
) -> dict:
    """
    레짐별 복합 전략 실행.
    use_kelly=True: 저엔트로피는 Kelly(mpe깊이 기반), 고엔트로피는 POS_SIZE_MOM 고정
    반환: {equity, trades, regime_log}
    """
    rsi        = compute_rsi(df["close"])
    ma200      = df["close"].rolling(MA_SLOW).mean()
    ma20       = df["close"].rolling(MA_FAST).mean()
    ma20_slope = compute_ma_slope(df["close"], MA_FAST)
    # Donchian breakout: N봉 최고/최저 (shift(1)로 현재봉 제외)
    don_high   = df["close"].shift(1).rolling(BREAKOUT_BARS).max()
    don_low    = df["close"].shift(1).rolling(BREAKOUT_BARS).min()

    low_thresh  = np.percentile(mpe.dropna(), LOW_PCT)
    high_thresh = np.percentile(mpe.dropna(), HIGH_PCT)

    onchain_ok = pd.Series(True, index=df.index)
    if h_onchain is not None:
        oc_thresh  = np.percentile(h_onchain.dropna(), 40)
        onchain_ok = h_onchain <= oc_thresh

    equity      = 1.0
    position    = 0       # 1=롱, -1=숏
    entry_price = 0.0
    entry_hour  = -999
    cur_max_h   = MAX_HOLD_H
    cur_regime  = "neutral"
    cur_pos_size = POS_SIZE

    curve       = []
    trades      = []
    regime_log  = []

    for i, idx in enumerate(df.index):
        price     = df["close"].loc[idx]
        mpe_val   = mpe.loc[idx]       if idx in mpe.index       else np.nan
        rsi_val   = rsi.loc[idx]       if idx in rsi.index       else 50.0
        ma200_val  = ma200.loc[idx]      if idx in ma200.index      else np.nan
        ma20_val   = ma20.loc[idx]       if idx in ma20.index       else np.nan
        slope_val  = ma20_slope.loc[idx] if idx in ma20_slope.index else 0.0
        don_h_val  = don_high.loc[idx]   if idx in don_high.index   else np.nan
        don_l_val  = don_low.loc[idx]    if idx in don_low.index    else np.nan
        oc_ok      = onchain_ok.loc[idx] if idx in onchain_ok.index else True

        # 레짐 판단
        if np.isnan(mpe_val):
            regime = "neutral"
        elif mpe_val <= low_thresh:
            regime = "low"    # 저엔트로피 -> 역추세
        elif mpe_val >= high_thresh:
            regime = "high"   # 고엔트로피 -> 모멘텀
        else:
            regime = "neutral"

        regime_log.append(regime)

        # ── 청산 ──────────────────────────────────────────────────────────
        if position != 0:
            hours_held  = i - entry_hour
            should_exit = hours_held >= cur_max_h

            if position == 1:
                if cur_regime == "low" and rsi_val > 50:
                    should_exit = True
                elif cur_regime == "high" and (not np.isnan(ma20_val)) and price < ma20_val:
                    should_exit = True
            elif position == -1:
                if cur_regime == "low" and rsi_val < 50:
                    should_exit = True
                elif cur_regime == "high" and (not np.isnan(ma20_val)) and price > ma20_val:
                    should_exit = True

            if should_exit:
                pnl = (price - entry_price) / entry_price if position == 1 \
                      else (entry_price - price) / entry_price
                equity *= (1 + pnl * cur_pos_size) * (1 - FEE_RATE)
                trades[-1]["exit_price"] = price
                trades[-1]["exit_time"]  = idx
                trades[-1]["pnl"]        = pnl
                trades[-1]["pos_size"]   = cur_pos_size
                position = 0

        # ── 진입 ──────────────────────────────────────────────────────────
        if position == 0 and not np.isnan(ma200_val) and not np.isnan(ma20_val):

            if regime == "low" and oc_ok:
                # 역추세: RSI 극값 + MA200 필터
                above_ma200 = price > ma200_val
                below_ma200 = price < ma200_val

                # Kelly: 저엔트로피 깊이에 따라 포지션 결정
                if use_kelly and not np.isnan(mpe_val):
                    k_size = kelly_size(mpe_val, mpe)
                    if k_size == 0.0:
                        k_size = POS_SIZE  # LOW_PCT 이내이므로 최소 0.15
                else:
                    k_size = POS_SIZE

                if rsi_val < 30 and above_ma200:
                    position      = 1
                    cur_max_h     = MAX_HOLD_H
                    cur_regime    = "low"
                    cur_pos_size  = k_size
                elif allow_short and rsi_val > 70 and below_ma200:
                    position      = -1
                    cur_max_h     = MAX_HOLD_H
                    cur_regime    = "low"
                    cur_pos_size  = k_size

            elif regime == "high":
                # 모멘텀: Donchian(N봉) 신고가/신저가 돌파 + MA200 방향 일치
                above_ma200 = price > ma200_val
                below_ma200 = price < ma200_val

                don_break_up   = (not np.isnan(don_h_val)) and (price > don_h_val)
                don_break_down = (not np.isnan(don_l_val)) and (price < don_l_val)

                # 고엔트로피 모멘텀: 고정 사이즈 (승률 낮아 보수적)
                mom_size = POS_SIZE_MOM if use_kelly else POS_SIZE

                if don_break_up and slope_val > 0 and above_ma200:
                    position      = 1
                    cur_max_h     = MAX_MOM_H
                    cur_regime    = "high"
                    cur_pos_size  = mom_size
                elif allow_short and don_break_down and slope_val < 0 and below_ma200:
                    position      = -1
                    cur_max_h     = MAX_MOM_H
                    cur_regime    = "high"
                    cur_pos_size  = mom_size

            if position != 0:
                entry_price = price * (1 + FEE_RATE) if position == 1 \
                              else price * (1 - FEE_RATE)
                entry_hour  = i
                trades.append({
                    "entry_time":  idx,
                    "entry_price": entry_price,
                    "regime":      cur_regime,
                    "direction":   "long" if position == 1 else "short",
                    "pos_size":    cur_pos_size,
                    "exit_price":  None,
                    "exit_time":   None,
                    "pnl":         None,
                })

        curve.append(equity)

    regime_series = pd.Series(regime_log, index=df.index)
    return {
        "equity":  pd.Series(curve, index=df.index),
        "trades":  pd.DataFrame(trades) if trades else pd.DataFrame(),
        "regime":  regime_series,
    }


def run_comparison(df, mpe, h_onchain=None):
    """레짐 전략 vs 기존 전략 비교"""
    from src.analysis.h4_backtest import run_strategy

    print("=" * 65)
    print("엔트로피 레짐 복합 전략 vs 기존 전략")
    print(f"  저엔트로피 MPE < {LOW_PCT}%  : 역추세 (RSI<30 롱 / RSI>70 숏)")
    print(f"  고엔트로피 MPE > {HIGH_PCT}%  : 모멘텀 (MA20 돌파 추세 추종)")
    print(f"  중간 구간        : 관망")
    print("=" * 65)

    # 레짐 전략
    regime_out = run_regime_strategy(df, mpe, h_onchain, allow_short=True)
    eq_regime  = regime_out["equity"]
    trades_df  = regime_out["trades"]
    regime_ser = regime_out["regime"]

    # 기존 전략 (롱 only, 온체인 포함)
    eq_orig = run_strategy(df, mpe,
                           use_entropy_filter=True,
                           use_trend_filter=True,
                           use_kelly=True,
                           h_onchain=h_onchain)

    # 레짐별 거래 통계
    n_low  = len(trades_df[trades_df["regime"] == "low"])  if len(trades_df) else 0
    n_high = len(trades_df[trades_df["regime"] == "high"]) if len(trades_df) else 0
    n_total = len(trades_df)

    regime_counts = regime_ser.value_counts()
    pct_low     = regime_counts.get("low",     0) / len(regime_ser) * 100
    pct_high    = regime_counts.get("high",    0) / len(regime_ser) * 100
    pct_neutral = regime_counts.get("neutral", 0) / len(regime_ser) * 100

    print(f"\n[레짐 분포]")
    print(f"  저엔트로피 (역추세) : {pct_low:.1f}%")
    print(f"  고엔트로피 (모멘텀) : {pct_high:.1f}%")
    print(f"  중간 (관망)         : {pct_neutral:.1f}%")

    print(f"\n[진입 통계]")
    print(f"  저엔트로피 거래 : {n_low}번")
    print(f"  고엔트로피 거래 : {n_high}번")
    print(f"  총 거래         : {n_total}번")

    m_regime = compute_metrics(eq_regime)
    m_orig   = compute_metrics(eq_orig)

    print(f"\n{'전략':<30} {'수익률':>9} {'Sharpe':>8} {'최대낙폭':>10}")
    print("-" * 62)
    print(f"{'레짐 복합 (역추세+모멘텀)':<30} {m_regime['총 수익률']:>9} "
          f"{m_regime['Sharpe']:>8} {m_regime['최대 낙폭']:>10}")
    print(f"{'기존 (저엔트로피 롱만)':<30} {m_orig['총 수익률']:>9} "
          f"{m_orig['Sharpe']:>8} {m_orig['최대 낙폭']:>10}")

    # 롱/숏 분리 성과
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        closed = trades_df.dropna(subset=["pnl"])
        if len(closed):
            for reg in ["low", "high"]:
                sub = closed[closed["regime"] == reg]
                if len(sub):
                    win_r = (sub["pnl"] > 0).mean() * 100
                    avg_p = sub["pnl"].mean() * 100
                    print(f"  {reg} 레짐 - 승률 {win_r:.0f}%, 평균 PnL {avg_p:.2f}%  ({len(sub)}건)")

    return {
        "regime":   {"equity": eq_regime, "metrics": m_regime,
                     "trades": trades_df, "regime_series": regime_ser},
        "original": {"equity": eq_orig,   "metrics": m_orig},
    }


def plot_regime_results(results, df, mpe):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    regime_ser = results["regime"]["regime_series"]
    eq_regime  = results["regime"]["equity"]
    eq_orig    = results["original"]["equity"]
    trades_df  = results["regime"]["trades"]

    bah = df["close"] / df["close"].iloc[0]

    fig = plt.figure(figsize=(18, 20), facecolor="#0d1117")
    fig.suptitle("엔트로피 레짐 복합 전략", color="#e6edf3", fontsize=14, y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35,
                           height_ratios=[2.5, 1, 1, 1])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── 1. 누적 수익 + 레짐 배경 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    # 레짐 배경색
    low_mask  = regime_ser == "low"
    high_mask = regime_ser == "high"
    for mask, color, label in [
        (low_mask,  "#1f4e2e", "저엔트로피 (역추세)"),
        (high_mask, "#1a2e4a", "고엔트로피 (모멘텀)"),
    ]:
        blocks = mask.astype(int).diff().fillna(0)
        starts = regime_ser.index[blocks == 1].tolist()
        ends   = regime_ser.index[blocks == -1].tolist()
        if mask.iloc[0]:
            starts = [regime_ser.index[0]] + starts
        if mask.iloc[-1]:
            ends.append(regime_ser.index[-1])
        for s, e in zip(starts, ends):
            ax1.axvspan(s, e, alpha=0.25, color=color, label=label
                        if s == starts[0] else "")

    ax1.plot(bah.index, bah.values, color="#8b949e",
             linewidth=0.7, alpha=0.25, linestyle="--", label="BTC B&H")
    ax1.plot(eq_regime.index, eq_regime.values, color="#56d364",
             linewidth=2.0,
             label=f"레짐 복합  (Sharpe {results['regime']['metrics']['Sharpe']})")
    ax1.plot(eq_orig.index, eq_orig.values, color="#79c0ff",
             linewidth=1.5, linestyle="--",
             label=f"기존 전략  (Sharpe {results['original']['metrics']['Sharpe']})")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("누적 수익 (초록=저엔트로피 구간 / 파랑=고엔트로피 구간)",
                  color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")

    # ── 2. MPE 시계열 + 레짐 임계값 ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)
    ax2.plot(mpe.index, mpe.values, color="#79c0ff", linewidth=0.8, alpha=0.7)
    low_t  = np.percentile(mpe.dropna(), LOW_PCT)
    high_t = np.percentile(mpe.dropna(), HIGH_PCT)
    ax2.axhline(low_t,  color="#56d364", linewidth=1.2, linestyle="--",
                label=f"저엔트로피 상한 ({LOW_PCT}th pct = {low_t:.3f})")
    ax2.axhline(high_t, color="#f78166", linewidth=1.2, linestyle="--",
                label=f"고엔트로피 하한 ({HIGH_PCT}th pct = {high_t:.3f})")
    ax2.set_title("MPE 시계열 + 레짐 임계값", color="#e6edf3")
    ax2.set_ylabel("MPE", color="#8b949e")
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")

    # ── 3. Sharpe 비교 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    labels = ["레짐 복합\n(역추세+모멘텀)", "기존 전략\n(저엔트로피 롱)"]
    sharpes = [float(results["regime"]["metrics"]["Sharpe"]),
               float(results["original"]["metrics"]["Sharpe"])]
    colors  = ["#56d364" if s > 0 else "#ff7b72" for s in sharpes]
    bars = ax3.bar(labels, sharpes, color=colors, width=0.5)
    for bar, val in zip(bars, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.01 if val >= 0 else -0.05),
                 f"{val:.3f}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=11, fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_title("Sharpe 비교", color="#e6edf3")

    # ── 4. 진입 횟수 by 레짐 ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    if len(trades_df):
        reg_counts = trades_df["regime"].value_counts()
        reg_labels = list(reg_counts.index)
        reg_vals   = list(reg_counts.values)
        reg_colors = {"low": "#56d364", "high": "#f78166"}
        bars2 = ax4.bar([f"{r}\n레짐" for r in reg_labels],
                        reg_vals,
                        color=[reg_colors.get(r, "#8b949e") for r in reg_labels],
                        width=0.5)
        for bar, val in zip(bars2, reg_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom",
                     color="#e6edf3", fontsize=12)
    ax4.set_title("레짐별 진입 횟수", color="#e6edf3")
    ax4.set_ylabel("거래 수", color="#8b949e")

    # ── 5. 레짐별 승률 & 평균 PnL ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    _style(ax5)
    if len(trades_df):
        closed = trades_df.dropna(subset=["pnl"])
        if len(closed):
            x    = np.arange(2)
            regs = ["low", "high"]
            win_rates = []
            avg_pnls  = []
            for reg in regs:
                sub = closed[closed["regime"] == reg]
                if len(sub):
                    win_rates.append((sub["pnl"] > 0).mean() * 100)
                    avg_pnls.append(sub["pnl"].mean() * 100)
                else:
                    win_rates.append(0)
                    avg_pnls.append(0)

            width = 0.35
            bars3 = ax5.bar(x - width/2, win_rates, width,
                            color=["#56d364", "#f78166"], alpha=0.8, label="승률 (%)")
            bars4 = ax5.bar(x + width/2, avg_pnls,  width,
                            color=["#1f6feb", "#b08800"], alpha=0.8, label="평균 PnL (%)")
            for bar, val in zip(list(bars3) + list(bars4),
                                win_rates + avg_pnls):
                ax5.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.2,
                         f"{val:.1f}", ha="center", va="bottom",
                         color="#e6edf3", fontsize=9)
            ax5.set_xticks(x)
            ax5.set_xticklabels(["저엔트로피 (역추세)", "고엔트로피 (모멘텀)"],
                                color="#e6edf3")
            ax5.axhline(50, color="#8b949e", linewidth=0.8, linestyle=":",
                        label="50% 기준선")
            ax5.axhline(0,  color="#8b949e", linewidth=0.8)
            ax5.set_title("레짐별 승률 vs 평균 PnL", color="#e6edf3")
            ax5.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                       labelcolor="#e6edf3")

    plt.savefig(RESULTS_DIR / "regime_strategy.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/regime_strategy.png")
