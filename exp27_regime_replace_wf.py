"""
실험 27: MA200 교체 Walk-forward

기반: Exp23-D (가격MPE<10%ile AND 볼륨MPE>=50%ile, WF Sharpe +1.291, 거래 37건)

목표: 거래량↑ AND Sharpe↑ 동시 달성
      Exp26B SHAP 인사이트 적용:
        - MA200 (SHAP 꼴찌 15.9%) 제거 → 막혔던 거래 해방
        - BTC 24H 수익률 (SHAP 2위 67.4%) 레짐 신호로 교체
        - RSI<35 확장(D)으로 거래수 최대화

비교 조건 (O5: BTC, SOL, AVAX, ADA, DOT):
  A (기준):   RSI<30 AND price > MA200       (Exp23-D 그대로)
  B (제거):   RSI<30 AND MA200 없음           (MA200 단독 효과 측정)
  C (교체):   RSI<30 AND btc_24h > -3%       (MA200 → BTC 단기 레짐)
  D (확장):   RSI<35 AND btc_24h > -3%       (C + RSI 확장, 거래 최대화)

고정 조건 (모든 모드):
  - 가격 MPE < 10%ile (train 기간)
  - 볼륨 MPE >= 50%ile (train 기간)
  - 온체인 엔트로피 < 40%ile
  - Kelly 사이징 (1%→50%, 5%→30%, 10%→15%)
  - 청산: RSI>50 OR 168H 보유

BTC 24H 임계값: -3%
  (평균 -0.211%, std 2.326% → -3%는 1.2σ 이하 급락 차단)

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp27_regime_replace_wf.py
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
from src.analysis.h3_validation import compute_rsi

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
ENTROPY_PCT = 10
VOL_PCT_D   = 50

RSI_OVERSOLD_A = 30
RSI_OVERSOLD_D = 35

BTC_REGIME_THRESH = -3.0  # btc_24h > -3%: 급락 차단


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, price_mpe_test, vol_mpe_test,
                h_test, price_mpe_train, vol_mpe_train,
                oc_threshold, k_pct1, k_pct5, k_pct10,
                btc_close_test,
                mode="A"):
    """
    mode:
      A — Exp23-D 기준 (RSI<30 AND MA200)
      B — RSI<30, MA200 없음
      C — RSI<30 AND btc_24h > -3%
      D — RSI<35 AND btc_24h > -3%
    """
    rsi   = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    price_thr   = np.percentile(price_mpe_train.dropna(), ENTROPY_PCT)
    vol_train_c = vol_mpe_train.dropna()
    vol_thr_50  = np.percentile(vol_train_c, VOL_PCT_D) if len(vol_train_c) > 10 else np.nan

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

        # BTC 24H 수익률 (C/D 모드에서 사용)
        btc_r24 = np.nan
        if btc_close_test is not None and idx in btc_close_test.index:
            loc = btc_close_test.index.get_loc(idx)
            if loc >= 24:
                btc_r24 = (btc_close_test.iloc[loc] /
                           btc_close_test.iloc[loc - 24] - 1) * 100

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        vol_ok = (not np.isnan(v_mpe)
                  and not np.isnan(vol_thr_50)
                  and v_mpe >= vol_thr_50)

        # RSI 신호 (모드별)
        if mode in ("A", "B", "C"):
            sig_ok = rsi_val < RSI_OVERSOLD_A
        else:  # D
            sig_ok = rsi_val < RSI_OVERSOLD_D

        # 레짐 조건 (모드별)
        if mode == "A":
            regime_ok = (not np.isnan(ma_val) and price > ma_val)
        elif mode == "B":
            regime_ok = True
        else:  # C, D
            regime_ok = (not np.isnan(btc_r24) and btc_r24 > BTC_REGIME_THRESH)

        # 청산
        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        # 진입
        if (position == 0
                and sig_ok
                and regime_ok
                and not np.isnan(p_mpe)
                and p_mpe <= price_thr
                and oc_ok
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
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, h_onchain, btc_close_full):
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

        df_test       = df[test_start:test_end]
        p_test        = price_mpe[test_start:test_end]
        v_test        = vol_mpe[test_start:test_end]
        h_test        = h_onchain[test_start:test_end] if h_onchain is not None else None
        btc_test      = btc_close_full[test_start:test_end]

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, p_test, v_test,
                h_test, p_train, v_train, oc_threshold,
                k_pct1, k_pct5, k_pct10,
                btc_test, mode=mode,
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
    ref          = COINS[0]
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
    if total_n < 30:
        print(f"  ⚠ 총 거래수 {total_n} < 30 → 통계적 유의성 낮음")
    return np.mean(sharpes), pos, total_n


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": "A: RSI<30 + MA200 (기준)",
        "B": "B: RSI<30, MA200 없음",
        "C": "C: RSI<30 + BTC 24H>-3%",
        "D": "D: RSI<35 + BTC 24H>-3%",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#f78166"}

    labels  = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins  = len(labels)

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. 누적 수익 비교
    ax1 = fig.add_subplot(gs[0, :])
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

    # 2. 구간별 Sharpe
    ax2 = fig.add_subplot(gs[1, :])
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
    ax2.set_title("구간별 포트폴리오 Sharpe (MA200 vs BTC 레짐 신호 비교)", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 3. 평균 Sharpe 요약
    ax3 = fig.add_subplot(gs[2, 0])
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
    ax3.axhline(1.291, color="#56d364", linewidth=1.0, linestyle="--", alpha=0.7,
                label="기준 A +1.291")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax3.set_title("조건별 평균 Sharpe", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # 4. 거래수 비교
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(4), total_trades, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.axhline(50, color="#56d364", linewidth=1.0, linestyle="--", label="목표 50건")
    ax4.axhline(37, color="#8b949e", linewidth=0.8, linestyle=":", label="기준 A 37건")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax4.set_title("조건별 총 거래수", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")
    ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    fig.suptitle(
        "Exp27: MA200 교체 Walk-forward (O5 포트폴리오)\n"
        f"A(MA200 기준) vs B(MA200 제거) vs C(BTC 24H>-3%) vs D(C+RSI<35)\n"
        f"목표: Sharpe ≥ +1.2 AND 거래수 ≥ 50건",
        color="#e6edf3", fontsize=11, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp27_regime_replace_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp27: MA200 교체 Walk-forward (SHAP 인사이트 기반)")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print(f"  BTC 레짐 임계값: btc_24h > {BTC_REGIME_THRESH}%")
    print("  A: RSI<30 + MA200 (Exp23-D 기준)")
    print("  B: RSI<30, MA200 없음")
    print("  C: RSI<30 + BTC 24H > -3%")
    print("  D: RSI<35 + BTC 24H > -3%")
    print("  목표: Sharpe ≥ +1.2 AND 거래수 ≥ 50건")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE 계산 (캐시 사용)...]")
    coin_data = {}
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

        coin_data[sym] = {"df": df, "price_mpe": price_mpe, "vol_mpe": vol_mpe, "h": h}
        print(f"  [{LABEL[sym]:6s}] {len(df):,}봉 로드 완료")

    # BTC 종가 (모든 코인에 레짐 신호로 전달)
    btc_close_full = coin_data["BTCUSDT"]["df"]["close"]
    btc_r24_all    = btc_close_full.pct_change(24) * 100
    below_thresh   = (btc_r24_all < BTC_REGIME_THRESH).mean() * 100
    print(f"\n  BTC 24H 수익률 분포: 평균={btc_r24_all.mean():.3f}%  "
          f"std={btc_r24_all.std():.3f}%  "
          f"({BTC_REGIME_THRESH}% 이하 비율: {below_thresh:.1f}%)")

    print("\n[Walk-forward 실행 중 (4개 조건 × 5코인)...]")
    coin_results_by_mode = {}
    for sym in COINS:
        d       = coin_data[sym]
        results = run_coin_wf_all(
            sym, d["df"], d["price_mpe"], d["vol_mpe"], d["h"],
            btc_close_full,
        )
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            print(f"  [{LABEL[sym]:6s} {mode}] "
                  f"Sharpe={np.mean(sharpes):+.3f}  양수:{pos}/6  거래:{n_total}")

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

    print("\n" + "=" * 70)
    print("  요약 비교 (기준: A = Exp23-D +1.291, 37건)")
    print(f"  {'조건':<6} | {'평균 Sharpe':>12} | {'개선폭':>8} | {'양수구간':>8} | {'거래수':>6} | {'목표':>6}")
    print("  " + "-" * 62)
    base_s = summary["A"]["avg_sharpe"]
    for mode in ["A", "B", "C", "D"]:
        s     = summary[mode]
        delta = s["avg_sharpe"] - base_s
        ok    = s["avg_sharpe"] >= 1.2 and s["total_n"] >= 50
        mark  = " ★" if ok else "  "
        print(f"  {mode}{mark}   | {s['avg_sharpe']:>+12.3f} | {delta:>+8.3f} | "
              f"{s['pos_cnt']}/6{'':<4} | {s['total_n']:>6} | {'✅' if ok else '  '}")
    print("=" * 70)
    print("  목표 달성 기준: Sharpe ≥ +1.2 AND 거래수 ≥ 50")

    # 2022-07~12 구간 방어 확인 (핵심 체크포인트)
    print("\n  [2022-07~12 구간 Sharpe — 레짐 필터 효과]")
    bad_period = "2022-07"
    for mode in ["A", "B", "C", "D"]:
        for r in all_port[mode]:
            if r["short_lbl"] == bad_period:
                s = float(r["metrics"]["Sharpe"])
                print(f"  {mode}: {s:+.3f}  ({r['n_trades']}건)")

    print("\n[시각화 생성...]")
    plot_results(all_port)
    print("\nExp27 완료!")


if __name__ == "__main__":
    main()
