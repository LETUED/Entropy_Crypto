"""
실험 25: 레짐 적응형 임계값 Walk-forward

기반: Exp23-D (가격MPE<10%ile AND 볼륨MPE>=50%ile, WF Sharpe +1.291)

Exp22 인사이트 (볼륨 필터 없을 때):
  2022-07~12: A(고정10%ile) = -2.252  vs  D(동적5%ile) = -0.180
  → 약세장에서 임계값을 낮추거나 진입을 막으면 손실 방어 가능

레짐 감지: BTC close vs BTC MA200(200봉)
  - BTC > BTC_MA200 = 강세 레짐 (altcoin 진입 허용)
  - BTC < BTC_MA200 = 약세 레짐 (더 엄격한 조건 or 진입 중단)

비교 조건 (O5: BTC, SOL, AVAX, ADA, DOT):
  A (기준):     Exp23-D (고정 10%ile + 볼륨MPE>=50%ile)
  B (고정전환): BTC 강세→고정10%ile, BTC 약세→고정5%ile  + 볼륨>=50%ile
  C (동적전환): BTC 강세→동적168H 10%ile, BTC 약세→동적168H 5%ile + 볼륨>=50%ile
  D (약세스킵): BTC 강세→A와 동일 진입, BTC 약세→진입 완전 중단 (청산은 유지)

핵심 가설: 2022-07~12 BTC < MA200 → D 조건에서 5개 손실 거래 차단 → Sharpe 대폭 개선

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp25_regime_adaptive_wf.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-","") not in ("utf8","utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h4_backtest import compute_metrics, FEE_RATE, MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi, generate_signals

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
         "ADAUSDT": "ADA", "DOTUSDT": "DOT"}

MPE_WINDOW  = 168
MPE_M       = 3
MPE_SCALES  = [1, 2, 4, 8]
ENTROPY_PCT = 10   # 기본 가격 MPE 임계값
VOL_PCT_D   = 50   # Exp23-D: 볼륨 MPE >= 50%ile

BTC_SYM     = "BTCUSDT"


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── BTC 레짐 시리즈 계산 ──────────────────────────────────────────────────────
def compute_btc_regime(btc_df: pd.DataFrame, ma_period: int = MA_PERIOD) -> pd.Series:
    """
    BTC close > MA200 → 1 (강세 레짐)
    BTC close < MA200 → 0 (약세 레짐)
    look-ahead 없음: rolling(200)은 과거 데이터만 사용
    """
    ma200  = btc_df["close"].rolling(ma_period, min_periods=ma_period).mean()
    regime = (btc_df["close"] > ma200).astype(float)
    regime[ma200.isna()] = np.nan
    return regime


# ── 동적 임계값 사전 계산 ─────────────────────────────────────────────────────
def precompute_rolling_thresholds(price_mpe: pd.Series, window: int = MPE_WINDOW):
    """
    rolling percentile 계산 (look-ahead 없음: rolling은 과거 window봉 기준)
    """
    pct10 = price_mpe.rolling(window, min_periods=window // 2).quantile(0.10)
    pct5  = price_mpe.rolling(window, min_periods=window // 2).quantile(0.05)
    return pct10, pct5


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, price_mpe_test, vol_mpe_test,
                btc_regime_test, roll_pct10_test, roll_pct5_test,
                h_test,
                price_mpe_train, vol_mpe_train, oc_threshold,
                k_pct1, k_pct5, k_pct10, mode="A"):
    """
    mode:
      A — Exp23-D 기준 (고정10%ile + 볼륨>=50%ile)
      B — BTC 레짐 + 고정 임계값 전환 (강세→10%, 약세→5%)
      C — BTC 레짐 + 동적168H 임계값 전환 (강세→동적10%, 약세→동적5%)
      D — BTC 레짐 + 약세 진입 완전 중단 (강세=A, 약세=신규진입 없음)
    """
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    # 고정 임계값 (train 기간 기준)
    fixed_thr10 = np.percentile(price_mpe_train.dropna(), 10)
    fixed_thr5  = np.percentile(price_mpe_train.dropna(), 5)

    # 볼륨 임계값
    vol_train_clean = vol_mpe_train.dropna()
    vol_thr_50      = (np.percentile(vol_train_clean, VOL_PCT_D)
                       if len(vol_train_clean) > 10 else np.nan)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity      = 1.0
    position    = 0
    entry_price = entry_hour = 0
    entry_kfrac = 0.0
    trades      = []
    curve       = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe   = price_mpe_test.loc[idx] if idx in price_mpe_test.index else np.nan
        v_mpe   = vol_mpe_test.loc[idx]   if idx in vol_mpe_test.index   else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        # BTC 레짐 (None이면 강세로 간주)
        btc_bull = True
        if btc_regime_test is not None and idx in btc_regime_test.index:
            rv = btc_regime_test.loc[idx]
            if not np.isnan(rv):
                btc_bull = bool(rv == 1)

        # 동적 임계값 (C 모드용)
        dyn_thr10 = roll_pct10_test.loc[idx] if (roll_pct10_test is not None
                                                  and idx in roll_pct10_test.index) else np.nan
        dyn_thr5  = roll_pct5_test.loc[idx]  if (roll_pct5_test is not None
                                                  and idx in roll_pct5_test.index) else np.nan

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        # 볼륨 조건 (공통: Exp23-D 기준)
        vol_ok = (not np.isnan(v_mpe)
                  and not np.isnan(vol_thr_50)
                  and v_mpe >= vol_thr_50)

        # 레짐별 진입 임계값 + 스킵 여부
        skip_entry = False
        if mode == "A":
            price_thr  = fixed_thr10
        elif mode == "B":
            price_thr  = fixed_thr10 if btc_bull else fixed_thr5
        elif mode == "C":
            # 동적 임계값; 유효값 없으면 고정 임계값으로 fallback
            if btc_bull:
                price_thr = dyn_thr10 if not np.isnan(dyn_thr10) else fixed_thr10
            else:
                price_thr = dyn_thr5  if not np.isnan(dyn_thr5)  else fixed_thr5
        else:  # D: 약세 시 진입 완전 스킵
            price_thr  = fixed_thr10
            skip_entry = not btc_bull

        # 청산 (항상 처리 — 레짐에 무관)
        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        # 진입
        if (position == 0
                and not skip_entry
                and not np.isnan(p_mpe)
                and not np.isnan(ma_val)
                and p_mpe <= price_thr
                and price > ma_val
                and oc_ok
                and sig_val == 1
                and vol_ok):
            k_frac = _kelly(p_mpe)
            if k_frac > 0:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
                entry_kfrac = k_frac

        curve.append(equity)

    if position == 1:
        pnl = (df_test["close"].iloc[-1] - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    return pd.Series(curve, index=df_test.index), trades


# ── 단일 코인 Walk-forward ────────────────────────────────────────────────────
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, btc_regime,
                    roll_pct10, roll_pct5, h_onchain):
    results_by_mode = {m: [] for m in ["A", "B", "C", "D"]}

    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta  = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end    = pd.Timestamp(END)

    while True:
        train_start = cursor
        train_end   = cursor + train_delta
        test_start  = train_end
        test_end    = test_start + test_delta
        if test_end > end:
            break

        p_train = price_mpe[train_start:train_end].dropna()
        if len(p_train) < 200:
            cursor += test_delta
            continue

        k_pct1  = np.percentile(p_train, 1)
        k_pct5  = np.percentile(p_train, 5)
        k_pct10 = np.percentile(p_train, 10)

        v_train = vol_mpe[train_start:train_end]

        oc_threshold = None
        if h_onchain is not None:
            hoc_tr = h_onchain[train_start:train_end].dropna()
            if len(hoc_tr) > 0:
                oc_threshold = np.percentile(hoc_tr, 40)

        df_test        = df[test_start:test_end]
        p_test         = price_mpe[test_start:test_end]
        v_test         = vol_mpe[test_start:test_end]
        btc_reg_test   = btc_regime[test_start:test_end] if btc_regime is not None else None
        rp10_test      = roll_pct10[test_start:test_end] if roll_pct10 is not None else None
        rp5_test       = roll_pct5[test_start:test_end]  if roll_pct5  is not None else None
        h_test         = h_onchain[test_start:test_end]  if h_onchain is not None else None

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, p_test, v_test,
                btc_reg_test, rp10_test, rp5_test,
                h_test,
                p_train, v_train, oc_threshold,
                k_pct1, k_pct5, k_pct10, mode=mode,
            )
            results_by_mode[mode].append({
                "label":      label,
                "short_lbl":  test_start.strftime("%Y-%m"),
                "test_start": test_start,
                "test_end":   test_end,
                "equity":     eq,
                "metrics":    compute_metrics(eq),
                "trades":     trades,
                "n_trades":   len(trades),
            })

        cursor += test_delta

    return results_by_mode


# ── 포트폴리오 집계 ───────────────────────────────────────────────────────────
def aggregate_portfolio(coin_results_by_mode, mode):
    ref         = COINS[0]
    port_results = []
    for win in coin_results_by_mode[ref][mode]:
        ts, te = win["test_start"], win["test_end"]
        equities, all_pnls, total_trades = [], [], 0
        for sym in COINS:
            match = next((w for w in coin_results_by_mode[sym][mode]
                          if w["test_start"] == ts and w["test_end"] == te), None)
            if match is None:
                continue
            equities.append(match["equity"])
            total_trades += match["n_trades"]
            all_pnls.extend([t["pnl_pct"] for t in match["trades"]])
        if not equities:
            continue
        n       = len(equities)
        port_eq = sum(eq / n for eq in equities)
        wr      = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0
        port_results.append({
            "label":      win["label"],
            "short_lbl":  win["short_lbl"],
            "test_start": ts,
            "equity":     port_eq,
            "metrics":    compute_metrics(port_eq),
            "n_trades":   total_trades,
            "win_rate":   wr,
        })
    return port_results


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(port_results, name):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]
    pos     = sum(s > 0 for s in sharpes)
    total_n = sum(r["n_trades"] for r in port_results)
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'기간':<20} {'Sharpe':>8} {'수익률':>9} {'거래수':>7} {'승률':>7}")
    print(f"  {'-'*55}")
    for r, s in zip(port_results, sharpes):
        m      = r["metrics"]
        marker = "[+]" if s > 0 else "[-]"
        print(f"  {marker} {r['label']:<20} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}%")
    print(f"  {'-'*55}")
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  양수: {pos}/{len(sharpes)}  |  총 거래: {total_n}")
    return np.mean(sharpes), pos, total_n


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port, btc_df):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": "A: Exp23-D 기준 (고정10%ile)",
        "B": "B: 레짐전환 (강세10%, 약세5%)",
        "C": "C: 동적전환 (강세동적10%, 약세동적5%)",
        "D": "D: 약세 진입 완전 스킵",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#f78166"}

    labels  = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins  = len(labels)

    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. BTC 가격 + MA200 레짐 시각화
    ax0 = fig.add_subplot(gs[0, :])
    _style(ax0)
    btc_close = btc_df["close"]
    btc_ma200 = btc_close.rolling(MA_PERIOD).mean()
    bull_mask = btc_close > btc_ma200

    ax0.plot(btc_close.index, btc_close.values, color="#79c0ff", linewidth=0.5,
             alpha=0.6, label="BTC Close")
    ax0.plot(btc_ma200.index, btc_ma200.values, color="#f0c040", linewidth=1.2,
             alpha=0.8, label="BTC MA200")

    # 약세 레짐 영역 음영
    bear_periods = []
    in_bear = False
    bear_start = None
    for ts, is_bull in bull_mask.items():
        if not is_bull and not in_bear:
            bear_start = ts
            in_bear = True
        elif is_bull and in_bear:
            bear_periods.append((bear_start, ts))
            in_bear = False
    if in_bear:
        bear_periods.append((bear_start, btc_close.index[-1]))

    for bs, be in bear_periods:
        ax0.axvspan(bs, be, alpha=0.12, color="#f78166", label="_")

    ax0.set_yscale("log")
    ax0.set_title("BTC 가격 vs MA200 (빨간 영역 = 약세 레짐 = D 조건 진입 차단)", color="#e6edf3")
    ax0.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax0.set_ylabel("BTC 가격 (log)", color="#8b949e")

    # 2. 누적 수익 비교
    ax1 = fig.add_subplot(gs[1, :])
    _style(ax1)
    for mode, prs in all_port.items():
        lw = 2.2 if mode != "A" else 1.0
        ls = "-"  if mode != "A" else ":"
        al = 0.9  if mode != "A" else 0.5
        for i, r in enumerate(prs):
            ax1.plot(r["equity"].index, r["equity"].values,
                     color=palette[mode], linewidth=lw, linestyle=ls, alpha=al,
                     label=mode_names[mode] if i == 0 else "_")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.6, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 비교 (점선=A 기준)", color="#e6edf3")
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=2, loc="upper left")

    # 3. 구간별 Sharpe
    ax2 = fig.add_subplot(gs[2, :])
    _style(ax2)
    x = np.arange(n_wins)
    w = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w
    for ni, (mode, prs) in enumerate(all_port.items()):
        sharpes = [float(r["metrics"]["Sharpe"]) for r in prs]
        bars = ax2.bar(x + offsets[ni], sharpes, w, color=palette[mode], alpha=0.85,
                       label=f"{mode_names[mode]}  avg {np.mean(sharpes):+.3f}")
        for bar, v in zip(bars, sharpes):
            yp = v + 0.02 if v >= 0 else v - 0.09
            ax2.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.2f}",
                     ha="center", fontsize=7, color="#e6edf3", fontweight="bold")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", rotation=30, ha="right")
    ax2.set_title("구간별 포트폴리오 Sharpe (레짐 적응형 임계값 조건별)", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 4. 평균 Sharpe 요약
    ax3 = fig.add_subplot(gs[3, 0])
    _style(ax3)
    avg_sharpes = [np.mean([float(r["metrics"]["Sharpe"]) for r in prs])
                   for prs in all_port.values()]
    pos_counts  = [sum(float(r["metrics"]["Sharpe"]) > 0 for r in prs)
                   for prs in all_port.values()]
    bars = ax3.bar(range(4), avg_sharpes, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v, pos in zip(bars, avg_sharpes, pos_counts):
        yp = v + 0.01 if v >= 0 else v - 0.07
        ax3.text(bar.get_x() + bar.get_width()/2, yp,
                 f"{v:+.3f}\n({pos}/{n_wins})", ha="center", fontsize=9,
                 color="#e6edf3", fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax3.set_title("조건별 평균 Sharpe", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")

    # 5. 거래수 비교
    ax4 = fig.add_subplot(gs[3, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(4), total_trades, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax4.set_title("조건별 총 거래수", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")

    fig.suptitle(
        "Exp25: 레짐 적응형 임계값 Walk-forward (O5 포트폴리오)\n"
        "A(Exp23-D기준) vs B(고정전환) vs C(동적전환) vs D(약세스킵)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp25_regime_adaptive_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp25: 레짐 적응형 임계값 Walk-forward")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print("  기준 (A): Exp23-D (가격MPE<10%ile AND 볼륨MPE>=50%ile, WF +1.291)")
    print("  조건  B:  BTC MA200 레짐 + 고정전환 (강세→10%ile, 약세→5%ile)")
    print("  조건  C:  BTC MA200 레짐 + 동적전환 (강세→동적10%ile, 약세→동적5%ile)")
    print("  조건  D:  BTC MA200 레짐 + 약세 진입 완전 중단")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE 계산 (캐시 사용)...]")
    coin_data = {}
    btc_df    = None

    for sym in COINS:
        df = collect(sym, "1h", START, END)

        price_mpe = rolling_mpe(df["close"], window=MPE_WINDOW,
                                m=MPE_M, scales=MPE_SCALES,
                                cache_key=f"{sym}_1h_{START}_{END}")

        vol_series = df["volume"].replace(0, np.nan).ffill()
        vol_mpe    = rolling_mpe(vol_series, window=MPE_WINDOW,
                                 m=MPE_M, scales=MPE_SCALES,
                                 cache_key=f"{sym}_1h_volume_{START}_{END}")

        h = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )

        # 동적 임계값 사전 계산 (모드 C용)
        roll_pct10, roll_pct5 = precompute_rolling_thresholds(price_mpe)

        coin_data[sym] = {
            "df": df, "price_mpe": price_mpe, "vol_mpe": vol_mpe, "h": h,
            "roll_pct10": roll_pct10, "roll_pct5": roll_pct5,
        }

        if sym == BTC_SYM:
            btc_df = df

        p_valid = price_mpe.notna().sum()
        print(f"  [{LABEL[sym]:6s}] {len(df):,}봉 | 가격MPE {p_valid:,}봉")

    # BTC 레짐 시리즈 계산
    print("\n[BTC MA200 레짐 시리즈 계산...]")
    btc_regime = compute_btc_regime(btc_df)
    bull_ratio = btc_regime.dropna().mean() * 100
    print(f"  강세 레짐 비율: {bull_ratio:.1f}%  |  약세 레짐 비율: {100-bull_ratio:.1f}%")

    # 구간별 레짐 통계
    for period, label in [("2022-07", "2022-07~12"), ("2023-01", "2023-01~06"),
                           ("2023-07", "2023-07~12"), ("2024-01", "2024-01~06")]:
        end_p = str(int(period[:4]) + (1 if period[5:7] == "07" else 0)) + \
                ("-01" if period[5:7] == "07" else "-07")
        # simpler: just slice 6 months
        try:
            ts = pd.Timestamp(period)
            te = ts + pd.DateOffset(months=6)
            slice_ = btc_regime[ts:te].dropna()
            if len(slice_) > 0:
                bull_pct = slice_.mean() * 100
                print(f"  [{label}] BTC 강세 {bull_pct:.1f}% / 약세 {100-bull_pct:.1f}%")
        except Exception:
            pass

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print("\n[Walk-forward 실행 중 (4개 조건 × 5코인)...]")
    coin_results_by_mode = {}

    for sym in COINS:
        d       = coin_data[sym]
        results = run_coin_wf_all(
            sym, d["df"], d["price_mpe"], d["vol_mpe"],
            btc_regime, d["roll_pct10"], d["roll_pct5"], d["h"],
        )
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            print(f"  [{LABEL[sym]:6s} {mode}] "
                  f"Sharpe={np.mean(sharpes):+.3f}  양수:{pos}/6  거래:{n_total}")

    # ── 포트폴리오 집계 ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  포트폴리오 결과 (O5 균등 배분)")
    print("=" * 70)

    all_port = {}
    summary  = {}
    for mode in ["A", "B", "C", "D"]:
        port                    = aggregate_portfolio(coin_results_by_mode, mode)
        all_port[mode]          = port
        avg_s, pos_cnt, total_n = print_results(port, f"조건 {mode}")
        summary[mode] = {"avg_sharpe": avg_s, "pos_cnt": pos_cnt, "total_n": total_n}

    # ── 요약 비교표 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  요약 비교 (기준: Exp23-D A = +1.291)")
    print(f"  {'조건':<6} | {'평균 Sharpe':>12} | {'개선폭':>8} | {'양수구간':>10} | {'거래수':>6}")
    print("  " + "-" * 52)
    base = summary["A"]["avg_sharpe"]
    for mode in ["A", "B", "C", "D"]:
        s     = summary[mode]
        delta = s["avg_sharpe"] - base
        mark  = "★" if s["avg_sharpe"] > base else " "
        print(f"  {mode}{mark}     | {s['avg_sharpe']:>+12.3f} | {delta:>+8.3f} | "
              f"{s['pos_cnt']}/6{' ':>7} | {s['total_n']:>6}")
    print("=" * 70)

    for mode in ["A", "B", "C", "D"]:
        if summary[mode]["total_n"] < 30:
            print(f"  ⚠ 조건 {mode}: 총 거래수 {summary[mode]['total_n']} < 30 → 통계적 유의성 낮음")

    print("\n[시각화 생성...]")
    plot_results(all_port, btc_df)
    print("\nExp25 완료!")


if __name__ == "__main__":
    main()
