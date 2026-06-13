"""
실험 21: ADA 교체 후보 Walk-forward 검증

배경:
  Exp20에서 ADA가 5코인 중 Sharpe 최하위 (0.660).
  교체 후보: SUSHI(2.109), GRT(1.998), UNI(1.903) — IS 성과 + 거래수 충분
  FIL/XRP는 Exp14에서 OOS 실패 → 제외

비교:
  O5 (기준):       BTC, SOL, AVAX, ADA, DOT     ← Exp10 WF Sharpe +0.529
  후보 A (ADA→SUSHI): BTC, SOL, AVAX, SUSHI, DOT
  후보 B (ADA→GRT):   BTC, SOL, AVAX, GRT, DOT
  후보 C (ADA→UNI):   BTC, SOL, AVAX, UNI, DOT

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
지표: 포트폴리오 Sharpe, 양수 구간 비율, 총 거래수

실행: py exp21_ada_replacement_wf.py
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
from src.analysis.h4_backtest import compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi, generate_signals

RESULTS_DIR = Path("results")
START, END   = "2021-01-01", "2025-01-01"

TRAIN_MONTHS = 12
TEST_MONTHS  = 6

ORIGINAL_5   = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT",  "DOTUSDT"]
CANDIDATE_A  = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "SUSHIUSDT","DOTUSDT"]
CANDIDATE_B  = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "GRTUSDT",  "DOTUSDT"]
CANDIDATE_C  = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "UNIUSDT",  "DOTUSDT"]

PORTFOLIOS = {
    "O5 (기준)":       ORIGINAL_5,
    "A (ADA→SUSHI)": CANDIDATE_A,
    "B (ADA→GRT)":   CANDIDATE_B,
    "C (ADA→UNI)":   CANDIDATE_C,
}

LABEL = {
    "BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
    "ADAUSDT": "ADA", "DOTUSDT": "DOT", "SUSHIUSDT": "SUSHI",
    "GRTUSDT": "GRT", "UNIUSDT": "UNI",
}

ALL_SYMS = list({s for coins in PORTFOLIOS.values() for s in coins})


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, mpe_test, h_test, mpe_train, oc_threshold,
                k_pct1, k_pct5, k_pct10):
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()
    mpe_thr = np.percentile(mpe_train.dropna(), ENTROPY_PCT)

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

        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            if (mpe_val <= mpe_thr and price > ma_val and oc_ok and sig_val == 1):
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


# ── 단일 코인 Walk-forward ────────────────────────────────────────────────────
def run_coin_wf(sym, df, mpe, h_onchain):
    results = []
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

        if len(df_test) < 168:
            cursor += test_delta
            continue

        equity, trades = _run_window(df_test, mpe_test, h_test,
                                     mpe_train, oc_threshold,
                                     k_pct1, k_pct5, k_pct10)

        metrics = compute_metrics(equity)
        label   = (f"{test_start.strftime('%Y-%m')} ~ "
                   f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        results.append({
            "label":      label,
            "short_lbl":  test_start.strftime("%Y-%m"),
            "test_start": test_start,
            "test_end":   test_end,
            "equity":     equity,
            "metrics":    metrics,
            "trades":     trades,
            "n_trades":   len(trades),
        })
        cursor += test_delta

    return results


# ── 포트폴리오 집계 ───────────────────────────────────────────────────────────
def aggregate_portfolio(coin_results: dict, coins: list) -> list:
    ref = next(s for s in coins if s in coin_results)
    port_results = []

    for win in coin_results[ref]:
        ts, te = win["test_start"], win["test_end"]
        equities, all_pnls, total_trades = [], [], 0

        for sym in coins:
            if sym not in coin_results:
                continue
            match = next((w for w in coin_results[sym]
                          if w["test_start"] == ts and w["test_end"] == te), None)
            if match is None:
                continue
            equities.append(match["equity"])
            total_trades += match["n_trades"]
            all_pnls.extend([t["pnl_pct"] for t in match["trades"]])

        if not equities:
            continue

        n = len(equities)
        port_eq = sum(eq / n for eq in equities)
        metrics = compute_metrics(port_eq)
        wr = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0

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
def print_results(port_results: list, name: str):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]
    pos = sum(s > 0 for s in sharpes)

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
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  "
          f"양수: {pos}/{len(sharpes)}  |  총 거래: {sum(r['n_trades'] for r in port_results)}")
    return np.mean(sharpes), pos


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port: dict):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    names  = list(all_port.keys())
    n_wins = len(next(iter(all_port.values())))
    labels = [r["short_lbl"] for r in next(iter(all_port.values()))]

    fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    palette = {"O5 (기준)": "#8b949e",
               "A (ADA→SUSHI)": "#56d364",
               "B (ADA→GRT)":   "#58a6ff",
               "C (ADA→UNI)":   "#f0c040"}

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. 포트폴리오 누적 수익 (구간별) — O5 vs 최고 후보
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    win_colors = plt.cm.plasma(np.linspace(0.15, 0.88, n_wins))
    for name, prs in all_port.items():
        lw  = 2.2 if name != "O5 (기준)" else 1.0
        ls  = "-"  if name != "O5 (기준)" else ":"
        alpha = 0.9 if name != "O5 (기준)" else 0.5
        for i, r in enumerate(prs):
            eq = r["equity"]
            ax1.plot(eq.index, eq.values,
                     color=palette.get(name, win_colors[i]),
                     linewidth=lw, linestyle=ls, alpha=alpha,
                     label=f"{name} {r['label']}" if i == 0 else "_")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.6, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 (점선=O5 기준 / 실선=후보)", color="#e6edf3")
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=2, loc="upper left")

    # 2. 구간별 Sharpe 비교 막대
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)
    x = np.arange(n_wins)
    w = 0.20
    offsets = np.linspace(-(len(names)-1)/2, (len(names)-1)/2, len(names)) * w

    for ni, (name, prs) in enumerate(all_port.items()):
        sharpes = [float(r["metrics"]["Sharpe"]) for r in prs]
        bars = ax2.bar(x + offsets[ni], sharpes, w,
                       color=palette.get(name, "#aaa"),
                       alpha=0.85, label=f"{name} 평균 {np.mean(sharpes):+.3f}")
        for bar, v in zip(bars, sharpes):
            yp = v + 0.02 if v >= 0 else v - 0.08
            ax2.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.2f}",
                     ha="center", fontsize=7.5, color="#e6edf3", fontweight="bold")

    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", rotation=30, ha="right")
    ax2.set_title("구간별 포트폴리오 Sharpe 비교", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 3. 평균 Sharpe 요약 막대
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    avg_sharpes = [np.mean([float(r["metrics"]["Sharpe"]) for r in prs])
                   for prs in all_port.values()]
    pos_counts  = [sum(float(r["metrics"]["Sharpe"]) > 0 for r in prs)
                   for prs in all_port.values()]
    bar_colors  = [palette.get(n, "#aaa") for n in names]
    bars = ax3.bar(range(len(names)), avg_sharpes, color=bar_colors, alpha=0.85)
    for bar, v, pos in zip(bars, avg_sharpes, pos_counts):
        yp = v + 0.01 if v >= 0 else v - 0.06
        ax3.text(bar.get_x() + bar.get_width()/2, yp,
                 f"{v:+.3f}\n({pos}/{n_wins})", ha="center", fontsize=9,
                 color="#e6edf3", fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, color="#8b949e", fontsize=9)
    ax3.set_title("포트폴리오별 평균 Sharpe (숫자: 양수 구간)", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")

    # 4. 거래수 비교
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(len(names)), total_trades, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, color="#8b949e", fontsize=9)
    ax4.set_title("포트폴리오별 총 거래수 (전체 WF 기간)", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")

    fig.suptitle("Exp21: ADA 교체 후보 Walk-forward 검증\n"
                 "O5(기준) vs A(SUSHI) vs B(GRT) vs C(UNI) | 학습 12M → 테스트 6M",
                 color="#e6edf3", fontsize=12, fontweight="bold", y=1.01)

    path = RESULTS_DIR / "exp21_ada_replacement_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp21: ADA 교체 후보 Walk-forward 검증")
    print("  비교 포트폴리오:")
    for name, coins in PORTFOLIOS.items():
        print(f"    {name}: {', '.join(LABEL.get(s,s) for s in coins)}")
    print(f"  기간: {START} ~ {END}")
    print(f"  방법: 학습 {TRAIN_MONTHS}M → 테스트 {TEST_MONTHS}M (슬라이딩)")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인 데이터 로드 ({len(ALL_SYMS)}개)...]")
    coin_data = {}
    for sym in ALL_SYMS:
        try:
            df  = collect(sym, "1h", START, END)
            mpe = rolling_mpe(df["close"], window=168,
                              cache_key=f"{sym}_1h_{START}_{END}")
            h = combined_onchain_entropy(
                funding_entropy(funding, df.index),
                fear_greed_entropy(fg, df.index),
            )
            coin_data[sym] = {"df": df, "mpe": mpe, "h": h}
            print(f"  [{LABEL.get(sym,sym):6s}] {len(df):,}봉")
        except Exception as e:
            print(f"  [{LABEL.get(sym,sym):6s}] 실패: {e}")

    print("\n[Walk-forward 실행 중...]")
    coin_results = {}
    for sym in ALL_SYMS:
        if sym not in coin_data:
            continue
        d = coin_data[sym]
        wf = run_coin_wf(sym, d["df"], d["mpe"], d["h"])
        coin_results[sym] = wf
        total_n = sum(w["n_trades"] for w in wf)
        sharpes = [float(w["metrics"]["Sharpe"]) for w in wf]
        print(f"  [{LABEL.get(sym,sym):6s}] 평균 Sharpe {np.mean(sharpes):+.3f} | "
              f"양수 {sum(s>0 for s in sharpes)}/{len(sharpes)} | 총 거래 {total_n}")

    print("\n[포트폴리오 집계...]")
    all_port = {}
    summary  = {}

    for name, coins in PORTFOLIOS.items():
        avail = [s for s in coins if s in coin_results]
        if len(avail) < len(coins):
            missing = [LABEL.get(s,s) for s in coins if s not in avail]
            print(f"  {name}: 데이터 없음 → {missing} — 건너뜀")
            continue
        port = aggregate_portfolio(coin_results, avail)
        all_port[name] = port
        avg_s, pos = print_results(port, name)
        summary[name] = {"avg_sharpe": avg_s, "pos": pos, "n_wins": len(port)}

    # 최종 비교
    print(f"\n{'='*70}")
    print("최종 비교 요약")
    print(f"{'='*70}")
    baseline = summary.get("O5 (기준)", {}).get("avg_sharpe", 0)
    for name, s in summary.items():
        diff = s["avg_sharpe"] - baseline
        marker = "★ 개선" if diff > 0.05 and name != "O5 (기준)" else ("  " if name == "O5 (기준)" else "▼ 열위")
        print(f"  {marker} {name:<20} Sharpe {s['avg_sharpe']:+.3f}  "
              f"양수 {s['pos']}/{s['n_wins']}  "
              f"{'(' + ('+' if diff >= 0 else '') + f'{diff:.3f} vs O5)' if name != 'O5 (기준)' else '(기준)'}")

    best = max(summary.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"\n  최고 포트폴리오: {best[0]}  (Sharpe {best[1]['avg_sharpe']:+.3f})")
    if best[0] != "O5 (기준)":
        print(f"  → ADA 교체 효과: {best[1]['avg_sharpe'] - baseline:+.3f} Sharpe 개선")
        print(f"  → 신규 포트폴리오 확정 검토 권장")
    else:
        print(f"  → ADA 교체 효과 없음 — 현재 O5 유지")

    if len(all_port) >= 2:
        plot_results(all_port)


if __name__ == "__main__":
    main()
