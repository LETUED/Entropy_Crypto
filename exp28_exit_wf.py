"""
실험 28: 청산 전략 최적화 Walk-forward

기반: Exp23-D (가격MPE<10%ile AND 볼륨MPE>=50%ile, WF Sharpe +1.291, 거래 37건)

목표: 청산 전략 최적화로 Sharpe 개선
  - 진입 조건 Exp23-D 완전 고정 (변경 없음)
  - 청산 조건만 4개 비교

이론 근거:
  Leung & Li (2015) https://arxiv.org/abs/1411.5062
    → 청산 타이밍이 초과수익 대부분 결정 (Double Stopping Problem)
  Ning et al. (2024) https://arxiv.org/abs/2309.16008
    → 신호 소멸 기반 동적 청산이 고정 RSI 청산보다 Sharpe 우수
  PE Bitcoin (2024) https://www.sciencedirect.com/science/article/pii/S0378437124001171
    → 저엔트로피 상태 지속 → 엔트로피 상승(신호 소멸) 시 청산이 논리적 일관성 보장

비교 조건 (진입: Exp23-D 고정 / 청산만 다름):
  A (기준):  RSI>50 OR 168H            (Exp23-D 현재 기준)
  B (소멸):  MPE>50%ile OR 168H        (신호 소멸 기반 청산)
  C (복합):  MPE>50%ile OR RSI>50 OR 168H  (B + 기존 RSI 조합)
  D (목표가): TP+2% OR RSI>50 OR 168H  (목표 수익률 달성 시 청산)

진입 조건 (모든 모드 동일):
  - 가격 MPE < 10%ile (train 기간)
  - 볼륨 MPE >= 50%ile (train 기간)
  - RSI < 30
  - 가격 > MA200
  - 온체인 엔트로피 < 40%ile
  - Kelly 사이징 그대로 (1%→50%, 5%→30%, 10%→15%)

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp28_exit_wf.py
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
ENTROPY_PCT = 10       # 진입: MPE 하위 10%ile
MPE_EXIT_PCT = 50      # 청산(B/C): MPE 상위 50%ile 돌파
VOL_PCT_D   = 50       # 볼륨 MPE >= 50%ile
RSI_OVERSOLD = 30      # 진입 RSI 임계값
TP_PCT      = 2.0      # 목표 수익률 (D 모드, %)


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
                mode="A"):
    """
    진입: Exp23-D 고정 (RSI<30 AND price>MA200 AND MPE<10% AND vol>=50% AND 온체인<40%)
    mode (청산만 다름):
      A — RSI>50 OR 168H                    (기준)
      B — MPE>50%ile OR 168H                (신호 소멸)
      C — MPE>50%ile OR RSI>50 OR 168H      (복합)
      D — TP+2% OR RSI>50 OR 168H           (목표가)
    """
    rsi   = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    p_train_clean = price_mpe_train.dropna()
    price_thr     = np.percentile(p_train_clean, ENTROPY_PCT)   # 진입: 10%ile
    price_thr_50  = np.percentile(p_train_clean, MPE_EXIT_PCT)  # 청산(B/C): 50%ile

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

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        vol_ok = (not np.isnan(v_mpe)
                  and not np.isnan(vol_thr_50)
                  and v_mpe >= vol_thr_50)

        # 청산 (mode별 청산 조건)
        if position == 1:
            held = i - entry_hour
            current_pnl_pct = (price - entry_price) / entry_price * 100

            if mode == "A":
                exit_ok = (rsi_val > 50 or held >= MAX_HOLD_H)
            elif mode == "B":
                mpe_exit = (not np.isnan(p_mpe) and p_mpe > price_thr_50)
                exit_ok  = (mpe_exit or held >= MAX_HOLD_H)
            elif mode == "C":
                mpe_exit = (not np.isnan(p_mpe) and p_mpe > price_thr_50)
                exit_ok  = (mpe_exit or rsi_val > 50 or held >= MAX_HOLD_H)
            else:  # D
                exit_ok = (current_pnl_pct >= TP_PCT or rsi_val > 50 or held >= MAX_HOLD_H)

            if exit_ok:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        # 진입 (Exp23-D 고정, 모든 모드 동일)
        if (position == 0
                and rsi_val < RSI_OVERSOLD
                and not np.isnan(p_mpe)
                and not np.isnan(ma_val)
                and p_mpe <= price_thr
                and price > ma_val
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
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, h_onchain):
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

        df_test = df[test_start:test_end]
        p_test  = price_mpe[test_start:test_end]
        v_test  = vol_mpe[test_start:test_end]
        h_test  = h_onchain[test_start:test_end] if h_onchain is not None else None

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, p_test, v_test,
                h_test, p_train, v_train, oc_threshold,
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
    ref          = COINS[0]
    port_results = []
    for win in coin_results_by_mode[ref][mode]:
        ts, te = win["test_start"], win["test_end"]
        equities, all_pnls, all_held, total_trades = [], [], [], 0
        for sym in COINS:
            match = next((w for w in coin_results_by_mode[sym][mode]
                          if w["test_start"] == ts and w["test_end"] == te), None)
            if match is None:
                continue
            equities.append(match["equity"])
            total_trades += match["n_trades"]
            all_pnls.extend([t["pnl_pct"] for t in match["trades"]])
            all_held.extend([t["held_h"]  for t in match["trades"]])
        if not equities:
            continue
        n       = len(equities)
        port_eq = sum(eq / n for eq in equities)
        wr      = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0
        avg_h   = np.mean(all_held) if all_held else 0.0
        port_results.append({
            "label":      win["label"],
            "short_lbl":  win["short_lbl"],
            "test_start": ts,
            "equity":     port_eq,
            "metrics":    compute_metrics(port_eq),
            "n_trades":   total_trades,
            "win_rate":   wr,
            "avg_held_h": avg_h,
        })
    return port_results


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(port_results, name):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]
    pos     = sum(s > 0 for s in sharpes)
    total_n = sum(r["n_trades"] for r in port_results)
    avg_h   = np.mean([r["avg_held_h"] for r in port_results if r["n_trades"] > 0])
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'기간':<20} {'Sharpe':>8} {'수익률':>9} {'거래수':>7} {'승률':>7} {'평균보유':>8}")
    print(f"  {'-'*62}")
    for r, s in zip(port_results, sharpes):
        m      = r["metrics"]
        marker = "[+]" if s > 0 else "[-]"
        print(f"  {marker} {r['label']:<20} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}% {r['avg_held_h']:>7.1f}H")
    print(f"  {'-'*62}")
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  양수: {pos}/{len(sharpes)}  |  "
          f"총 거래: {total_n}  |  평균 보유: {avg_h:.1f}H")
    if total_n < 30:
        print(f"  ⚠ 총 거래수 {total_n} < 30 → 통계적 유의성 낮음")
    return np.mean(sharpes), pos, total_n, avg_h


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": f"A: RSI>50 OR 168H (기준)",
        "B": f"B: MPE>{MPE_EXIT_PCT}%ile OR 168H (신호소멸)",
        "C": f"C: MPE>{MPE_EXIT_PCT}%ile OR RSI>50 OR 168H (복합)",
        "D": f"D: TP+{TP_PCT}% OR RSI>50 OR 168H (목표가)",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#f78166"}

    labels  = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins  = len(labels)

    fig = plt.figure(figsize=(22, 20), facecolor="#0d1117")
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
    ax2.set_title("구간별 포트폴리오 Sharpe (청산 조건별)", color="#e6edf3")
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
    ax3.set_xticklabels(list(all_port.keys()), color="#8b949e")
    ax3.set_title("조건별 평균 Sharpe", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # 4. 평균 보유 시간
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    avg_helds = []
    for mode, prs in all_port.items():
        all_h = []
        for r in prs:
            all_h.append(r["avg_held_h"])
        avg_helds.append(np.mean([h for h in all_h if h > 0]))
    bars = ax4.bar(range(4), avg_helds, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v in zip(bars, avg_helds):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}H",
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.axhline(MAX_HOLD_H, color="#8b949e", linewidth=0.8, linestyle=":",
                label=f"최대 {MAX_HOLD_H}H")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(list(all_port.keys()), color="#8b949e")
    ax4.set_title("조건별 평균 보유 시간 (청산 속도)", color="#e6edf3")
    ax4.set_ylabel("평균 보유 시간 (H)", color="#8b949e")
    ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    fig.suptitle(
        "Exp28: 청산 전략 최적화 Walk-forward (O5 포트폴리오)\n"
        f"진입: Exp23-D 고정 / 청산: A(RSI>50) vs B(MPE소멸) vs C(복합) vs D(TP+{TP_PCT}%)\n"
        "이론: Leung&Li(2015) / Ning et al.(2024) / PE Bitcoin(2024)",
        color="#e6edf3", fontsize=11, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp28_exit_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp28: 청산 전략 최적화 Walk-forward")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print(f"  진입: Exp23-D 고정 (RSI<{RSI_OVERSOLD} + MPE<{ENTROPY_PCT}% + vol>={VOL_PCT_D}% + MA200 + 온체인<40%)")
    print(f"  청산 A: RSI>50 OR {MAX_HOLD_H}H               (기준)")
    print(f"  청산 B: MPE>{MPE_EXIT_PCT}%ile OR {MAX_HOLD_H}H         (신호 소멸)")
    print(f"  청산 C: MPE>{MPE_EXIT_PCT}%ile OR RSI>50 OR {MAX_HOLD_H}H (복합)")
    print(f"  청산 D: TP+{TP_PCT}% OR RSI>50 OR {MAX_HOLD_H}H       (목표가)")
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

    print("\n[Walk-forward 실행 중 (4개 청산 조건 × 5코인)...]")
    coin_results_by_mode = {}
    for sym in COINS:
        d       = coin_data[sym]
        results = run_coin_wf_all(sym, d["df"], d["price_mpe"], d["vol_mpe"], d["h"])
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            avg_h_list = [t["held_h"] for w in results[mode] for t in w["trades"]]
            avg_h = np.mean(avg_h_list) if avg_h_list else 0.0
            print(f"  [{LABEL[sym]:6s} {mode}] "
                  f"Sharpe={np.mean(sharpes):+.3f}  양수:{pos}/6  거래:{n_total}  "
                  f"평균보유:{avg_h:.1f}H")

    print("\n" + "=" * 70)
    print("  포트폴리오 결과 (O5 균등 배분)")
    print("=" * 70)

    all_port = {}
    summary  = {}
    for mode in ["A", "B", "C", "D"]:
        port = aggregate_portfolio(coin_results_by_mode, mode)
        all_port[mode] = port
        avg_s, pos_cnt, total_n, avg_h = print_results(port, f"청산 조건 {mode}")
        summary[mode] = {"avg_sharpe": avg_s, "pos_cnt": pos_cnt,
                         "total_n": total_n, "avg_held_h": avg_h}

    print("\n" + "=" * 70)
    print("  요약 비교 (기준: A = Exp23-D +1.291, 37건)")
    print(f"  {'조건':<5} | {'평균 Sharpe':>12} | {'개선폭':>8} | {'양수구간':>8} | {'거래수':>6} | {'평균보유':>8}")
    print("  " + "-" * 60)
    base_s = summary["A"]["avg_sharpe"]
    for mode in ["A", "B", "C", "D"]:
        s     = summary[mode]
        delta = s["avg_sharpe"] - base_s
        mark  = " ★" if delta > 0 else "  "
        print(f"  {mode}{mark}  | {s['avg_sharpe']:>+12.3f} | {delta:>+8.3f} | "
              f"{s['pos_cnt']}/6{'':<4} | {s['total_n']:>6} | {s['avg_held_h']:>6.1f}H")
    print("=" * 70)

    best = max(summary, key=lambda m: summary[m]["avg_sharpe"])
    print(f"\n  최적 청산 조건: {best} "
          f"(Sharpe={summary[best]['avg_sharpe']:+.3f}, "
          f"거래={summary[best]['total_n']}건, "
          f"평균보유={summary[best]['avg_held_h']:.1f}H)")

    print("\n[시각화 생성...]")
    plot_results(all_port)
    print("\nExp28 완료!")


if __name__ == "__main__":
    main()
