"""
실험 22: 엔트로피 정교화 — 동적 임계값 + 기울기 필터 Walk-forward 검증

배경 (논문 근거):
  - Lahmiri & Bekiros (2019, arXiv:1901.04967): 암호화폐 효율성이 시간에 따라 극적 변화
    → 고정 10%ile은 레짐 전환 시 신호 과다/부족 문제
  - Bekker et al. (2021, PMC9073522): rolling 분포 기반 임계값이 고정값보다 우수
  - Singha (2025, arXiv:2512.15720): MPE 기울기는 방향 예측 불가, 노이즈 필터로만 유용

비교 조건 (O5 포트폴리오: BTC, SOL, AVAX, ADA, DOT):
  A (기준):   train 기간 고정 10%ile  ← Exp10/21 기준선
  B (동적):   각 봉에서 최근 168H MPE 분포의 rolling 10%ile
  C (동적+기울기): B + MPE 기울기 < 0 필터 (저엔트로피로 하락 중인 경우만)
  D (동적강화): 각 봉에서 최근 168H MPE 분포의 rolling 5%ile (더 선별적)

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp22_dynamic_threshold_wf.py
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

# 동적 임계값 롤링 윈도우 (시간 단위)
DYNAMIC_WINDOW = 168   # 7일 = 168봉
SLOPE_WINDOW   = 24    # 기울기 측정 기간 (24H)
ENTROPY_PCT    = 10    # 기본 백분위수
ENTROPY_PCT_D  = 5     # 강화 백분위수 (조건 D)


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 전처리: 동적 임계값 & 기울기 시리즈 사전 계산 ─────────────────────────────
def precompute_dynamic(mpe: pd.Series) -> dict:
    """
    각 봉에서 과거 DYNAMIC_WINDOW 개 MPE 값의 rolling 백분위수와 기울기 계산.
    look-ahead 없음: bar i에서 mpe[i-window+1 : i+1] 사용.
    """
    mpe_arr = mpe.values
    n = len(mpe_arr)

    rolling_pct10 = np.full(n, np.nan)
    rolling_pct5  = np.full(n, np.nan)
    slope_24      = np.full(n, np.nan)

    for i in range(n):
        # 동적 임계값
        start = max(0, i - DYNAMIC_WINDOW + 1)
        window_vals = mpe_arr[start:i+1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) >= 50:  # 최소 50개 있어야 백분위수 의미 있음
            rolling_pct10[i] = np.percentile(valid, ENTROPY_PCT)
            rolling_pct5[i]  = np.percentile(valid, ENTROPY_PCT_D)

        # 기울기: 24봉 전 대비 현재 값
        if i >= SLOPE_WINDOW and not np.isnan(mpe_arr[i]) and not np.isnan(mpe_arr[i - SLOPE_WINDOW]):
            slope_24[i] = mpe_arr[i] - mpe_arr[i - SLOPE_WINDOW]

    return {
        "pct10": pd.Series(rolling_pct10, index=mpe.index),
        "pct5":  pd.Series(rolling_pct5,  index=mpe.index),
        "slope": pd.Series(slope_24,       index=mpe.index),
    }


# ── 단일 윈도우 실행 (조건별 분기) ───────────────────────────────────────────
def _run_window(df_test, mpe_test, h_test, mpe_train, oc_threshold,
                k_pct1, k_pct5, k_pct10,
                dynamic_pct10=None, dynamic_pct5=None, slope_series=None,
                mode="A"):
    """
    mode:
      A — train 기간 고정 10%ile (기준)
      B — rolling 168H 동적 10%ile
      C — rolling 168H 동적 10%ile + 기울기 < 0
      D — rolling 168H 동적 5%ile
    """
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    # 고정 임계값 (mode A 전용)
    fixed_thr = np.percentile(mpe_train.dropna(), ENTROPY_PCT)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity = 1.0
    position = 0
    entry_price = entry_hour = 0
    entry_kfrac = 0.0
    trades = []
    curve  = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        mpe_val = mpe_test.loc[idx] if idx in mpe_test.index else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        # 조건별 임계값 결정
        if mode == "A":
            mpe_thr = fixed_thr
            slope_ok = True
        elif mode == "B":
            mpe_thr  = dynamic_pct10.loc[idx] if idx in dynamic_pct10.index else np.nan
            slope_ok = True
        elif mode == "C":
            mpe_thr  = dynamic_pct10.loc[idx] if idx in dynamic_pct10.index else np.nan
            s_val    = slope_series.loc[idx]   if idx in slope_series.index else np.nan
            slope_ok = (not np.isnan(s_val)) and (s_val < 0)
        else:  # D
            mpe_thr  = dynamic_pct5.loc[idx] if idx in dynamic_pct5.index else np.nan
            slope_ok = True

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
                and not np.isnan(mpe_val)
                and not np.isnan(ma_val)
                and not np.isnan(mpe_thr)):
            if (mpe_val <= mpe_thr and price > ma_val
                    and oc_ok and sig_val == 1 and slope_ok):
                k_frac = _kelly(mpe_val)
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


# ── 단일 코인 Walk-forward (4가지 조건 동시 실행) ─────────────────────────────
def run_coin_wf_all(sym, df, mpe, h_onchain, dynamic):
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

        mpe_train = mpe[train_start:train_end].dropna()
        if len(mpe_train) < 200:
            cursor += test_delta
            continue

        k_pct1  = np.percentile(mpe_train, 1)
        k_pct5  = np.percentile(mpe_train, 5)
        k_pct10 = np.percentile(mpe_train, 10)

        oc_threshold = None
        if h_onchain is not None:
            hoc_train = h_onchain[train_start:train_end].dropna()
            if len(hoc_train) > 0:
                oc_threshold = np.percentile(hoc_train, 40)

        df_test  = df[test_start:test_end]
        mpe_test = mpe[test_start:test_end]
        h_test   = h_onchain[test_start:test_end] if h_onchain is not None else None

        dpct10_test = dynamic["pct10"][test_start:test_end]
        dpct5_test  = dynamic["pct5"][test_start:test_end]
        slope_test  = dynamic["slope"][test_start:test_end]

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, mpe_test, h_test, mpe_train, oc_threshold,
                k_pct1, k_pct5, k_pct10,
                dynamic_pct10=dpct10_test, dynamic_pct5=dpct5_test,
                slope_series=slope_test, mode=mode,
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
def aggregate_portfolio(coin_results_by_mode: dict, mode: str) -> list:
    """mode별 5코인 포트폴리오 집계"""
    ref = COINS[0]
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

        n = len(equities)
        port_eq  = sum(eq / n for eq in equities)
        metrics  = compute_metrics(port_eq)
        wr       = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0

        port_results.append({
            "label":      win["label"],
            "short_lbl":  win["short_lbl"],
            "test_start": ts,
            "equity":     port_eq,
            "metrics":    metrics,
            "n_trades":   total_trades,
            "win_rate":   wr,
        })

    return port_results


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(port_results, name):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]
    pos = sum(s > 0 for s in sharpes)
    total_n = sum(r["n_trades"] for r in port_results)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'기간':<20} {'Sharpe':>8} {'수익률':>9} {'거래수':>7} {'승률':>7}")
    print(f"  {'-'*55}")
    for r, s in zip(port_results, sharpes):
        m = r["metrics"]
        marker = "[+]" if s > 0 else "[-]"
        print(f"  {marker} {r['label']:<20} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}%")
    print(f"  {'-'*55}")
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  양수: {pos}/{len(sharpes)}  |  총 거래: {total_n}")
    return np.mean(sharpes), pos, total_n


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port: dict):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": "A: 고정 10%ile (기준)",
        "B": "B: 동적 168H 10%ile",
        "C": "C: 동적 10%ile + 기울기",
        "D": "D: 동적 168H 5%ile",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#58a6ff"}

    labels = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins = len(labels)

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. 포트폴리오 누적 수익 비교 (전체 WF 구간)
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    win_colors = plt.cm.tab20(np.linspace(0, 1, n_wins))
    for mode, prs in all_port.items():
        lw    = 2.2 if mode != "A" else 1.0
        ls    = "-"  if mode != "A" else ":"
        alpha = 0.9  if mode != "A" else 0.5
        for i, r in enumerate(prs):
            ax1.plot(r["equity"].index, r["equity"].values,
                     color=palette[mode], linewidth=lw, linestyle=ls, alpha=alpha,
                     label=mode_names[mode] if i == 0 else "_")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.6, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 비교 (점선=A 기준)", color="#e6edf3")
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=2, loc="upper left")

    # 2. 구간별 Sharpe 비교
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)
    x = np.arange(n_wins)
    w = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    for ni, (mode, prs) in enumerate(all_port.items()):
        sharpes = [float(r["metrics"]["Sharpe"]) for r in prs]
        bars = ax2.bar(x + offsets[ni], sharpes, w,
                       color=palette[mode], alpha=0.85,
                       label=f"{mode_names[mode]}  avg {np.mean(sharpes):+.3f}")
        for bar, v in zip(bars, sharpes):
            yp = v + 0.02 if v >= 0 else v - 0.09
            ax2.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.2f}",
                     ha="center", fontsize=7, color="#e6edf3", fontweight="bold")

    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", rotation=30, ha="right")
    ax2.set_title("구간별 포트폴리오 Sharpe (A~D 조건 비교)", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 3. 평균 Sharpe 요약
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    avg_sharpes = [np.mean([float(r["metrics"]["Sharpe"]) for r in prs])
                   for prs in all_port.values()]
    pos_counts  = [sum(float(r["metrics"]["Sharpe"]) > 0 for r in prs)
                   for prs in all_port.values()]
    bar_colors  = [palette[m] for m in all_port.keys()]
    bars = ax3.bar(range(4), avg_sharpes, color=bar_colors, alpha=0.85)
    for bar, v, pos in zip(bars, avg_sharpes, pos_counts):
        yp = v + 0.01 if v >= 0 else v - 0.07
        ax3.text(bar.get_x() + bar.get_width()/2, yp,
                 f"{v:+.3f}\n({pos}/{n_wins})", ha="center", fontsize=9,
                 color="#e6edf3", fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([mode_names[m] for m in all_port.keys()],
                        color="#8b949e", fontsize=8)
    ax3.set_title("조건별 평균 Sharpe (괄호: 양수 구간 수)", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")

    # 4. 거래수 비교
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(4), total_trades, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([mode_names[m] for m in all_port.keys()],
                        color="#8b949e", fontsize=8)
    ax4.set_title("조건별 총 거래수 (빈도 vs 품질 트레이드오프)", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")

    fig.suptitle(
        "Exp22: 엔트로피 동적 임계값 + 기울기 필터 검증 (O5 포트폴리오)\n"
        "A(고정10%) vs B(동적10%) vs C(동적10%+기울기) vs D(동적5%)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp22_dynamic_threshold_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp22: 동적 임계값 + 기울기 필터 Walk-forward 검증")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print(f"  조건 A: train 기간 고정 10%ile (기준)")
    print(f"  조건 B: rolling {DYNAMIC_WINDOW}H 동적 10%ile")
    print(f"  조건 C: B + MPE 기울기 < 0 ({SLOPE_WINDOW}H)")
    print(f"  조건 D: rolling {DYNAMIC_WINDOW}H 동적 5%ile")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인 데이터 + 동적 임계값 사전 계산...]")
    coin_data = {}
    for sym in COINS:
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168,
                          cache_key=f"{sym}_1h_{START}_{END}")
        h = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        dynamic = precompute_dynamic(mpe)
        coin_data[sym] = {"df": df, "mpe": mpe, "h": h, "dynamic": dynamic}
        print(f"  [{LABEL[sym]:6s}] 데이터 {len(df):,}봉 | "
              f"동적 임계값 유효 봉: {dynamic['pct10'].notna().sum():,}개")

    print("\n[Walk-forward 실행 중 (4개 조건 × 5코인)...]")
    coin_results_by_mode = {}
    for sym in COINS:
        d = coin_data[sym]
        results = run_coin_wf_all(sym, d["df"], d["mpe"], d["h"], d["dynamic"])
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            print(f"  [{LABEL[sym]:6s}] {mode}: 평균 Sharpe {np.mean(sharpes):+.3f} | "
                  f"양수 {pos}/{len(sharpes)} | 거래 {n_total}")

    print("\n[포트폴리오 집계...]")
    all_port = {}
    summary  = {}
    baseline = None

    for mode in ["A", "B", "C", "D"]:
        port = aggregate_portfolio(coin_results_by_mode, mode)
        all_port[mode] = port
        avg_s, pos, total_n = print_results(
            port,
            {"A": "A: 고정 10%ile (기준)",
             "B": "B: 동적 168H 10%ile",
             "C": "C: 동적 10%ile + 기울기 < 0",
             "D": "D: 동적 168H 5%ile"}[mode],
        )
        summary[mode] = {"avg_sharpe": avg_s, "pos": pos, "total_n": total_n}
        if mode == "A":
            baseline = avg_s

    # 최종 비교
    print(f"\n{'='*70}")
    print("최종 비교 요약")
    print(f"{'='*70}")
    mode_desc = {
        "A": "A 고정10%ile (기준)  ",
        "B": "B 동적10%ile        ",
        "C": "C 동적10%+기울기    ",
        "D": "D 동적5%ile         ",
    }
    for mode, s in summary.items():
        diff   = s["avg_sharpe"] - baseline
        n_wins = len(all_port[mode])
        if mode == "A":
            tag = "  (기준)"
        elif diff > 0.05:
            tag = "★ 개선"
        elif diff < -0.05:
            tag = "▼ 열위"
        else:
            tag = "≈ 동일"
        print(f"  {tag} {mode_desc[mode]}  Sharpe {s['avg_sharpe']:+.3f}  "
              f"양수 {s['pos']}/{n_wins}  거래 {s['total_n']}  "
              f"{'(' + ('+' if diff >= 0 else '') + f'{diff:.3f} vs A)' if mode != 'A' else ''}")

    best = max(summary.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"\n  최고 조건: {best[0]}  (Sharpe {best[1]['avg_sharpe']:+.3f})")

    if best[0] == "A":
        print("  → 동적 임계값 효과 없음 — 현재 고정 임계값 유지")
    elif best[0] == "B":
        print("  → 동적 168H 10%ile이 기준 대비 개선 — 전략 업데이트 권장")
    elif best[0] == "C":
        print("  → 동적 임계값 + 기울기 필터 조합이 최고 — 거래수 감소 감안")
    else:
        print("  → 동적 5%ile이 최고 — 더 선별적 진입 (거래수 확인 필요)")

    plot_results(all_port)


if __name__ == "__main__":
    main()
