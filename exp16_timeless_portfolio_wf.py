"""
실험 16: 타임리스 8코인 포트폴리오 Walk-forward

목적: Exp14에서 IS+/OOS+ 코인들로 포트폴리오 구성 후
     Walk-forward로 통계적 유의성 검증

배경:
  - Exp14에서 타임리스 13코인 확인 (IS+/OOS+)
  - 단, OOS 거래수 1~5건으로 통계 불충분
  - 포트폴리오 집계 → 거래수 증가 → 통계력 확보

코인 선정 기준 (Exp14에서):
  1. IS Sharpe > 0 (전략이 IS에서 작동)
  2. OOS Sharpe > 0 (OOS에서도 작동 = 타임리스)
  3. IS 거래수 >= 15 (최소 통계)
  4. 순수 시장 코인 (거래소 토큰/게임 제외)
  -> LTC, SOL, ADA, DOT, ALGO, XRP, GRT, AVAX

비교 대상:
  - 원래 5코인 (BTC, SOL, AVAX, ADA, DOT) [Exp10]
  - 타임리스 8코인 (BTC 제외, + LTC/ALGO/XRP/GRT)

실행: py exp16_timeless_portfolio_wf.py
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

# Exp14 타임리스 코인 (IS Sharpe > 0, OOS Sharpe > 0, IS 거래 >= 15, 순수 시장)
TIMELESS_8 = ["LTCUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "ALGOUSDT",
               "XRPUSDT", "GRTUSDT", "AVAXUSDT"]

# Exp10 원래 5코인 (비교 기준)
ORIGINAL_5 = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]

LABEL = {
    "LTCUSDT": "LTC", "SOLUSDT": "SOL", "ADAUSDT": "ADA", "DOTUSDT": "DOT",
    "ALGOUSDT": "ALGO", "XRPUSDT": "XRP", "GRTUSDT": "GRT", "AVAXUSDT": "AVAX",
    "BTCUSDT": "BTC",
}

TRAIN_MONTHS = 12
TEST_MONTHS  = 6


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우 전략 실행 (최적화: Kelly 임계값 사전 계산) ──────────────────
def _run_window(df_test, mpe_test, h_test,
                mpe_train, oc_threshold, k_pct1, k_pct5, k_pct10):
    """
    Kelly 임계값(k_pct*)을 train 기간에서 사전 계산하여 루프 외부에서 전달
    """
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()
    mpe_threshold = np.percentile(mpe_train.dropna(), ENTROPY_PCT)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity = 1.0
    position = 0
    entry_price = 0.0
    entry_hour  = 0
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
            if (mpe_val <= mpe_threshold and price > ma_val and oc_ok
                    and sig_val == 1):
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


# ── 단일 코인 Walk-forward ───────────────────────────────────────────────────
def run_coin_wf(sym, df, mpe, h_onchain):
    results = []
    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta  = pd.DateOffset(months=TEST_MONTHS)
    step_delta  = pd.DateOffset(months=TEST_MONTHS)

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
            cursor += step_delta
            continue

        # Kelly 임계값 사전 계산 (train 기간 기준)
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
            cursor += step_delta
            continue

        equity, trades = _run_window(df_test, mpe_test, h_test,
                                     mpe_train, oc_threshold,
                                     k_pct1, k_pct5, k_pct10)

        metrics = compute_metrics(equity)
        bah     = df_test["close"] / df_test["close"].iloc[0]
        label   = f"{test_start.strftime('%Y-%m')} ~ {(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}"

        results.append({
            "label":      label,
            "short_lbl":  test_start.strftime("%Y-%m"),
            "test_start": test_start,
            "test_end":   test_end,
            "equity":     equity,
            "bah":        bah,
            "metrics":    metrics,
            "trades":     trades,
            "n_trades":   len(trades),
        })
        cursor += step_delta

    return results


# ── 포트폴리오 집계 ──────────────────────────────────────────────────────────
def aggregate_portfolio(coin_results: dict, coins: list) -> list:
    """코인별 walk-forward 결과를 포트폴리오로 집계 (균등 배분)"""
    # 공통 윈도우 식별 (첫 번째 코인 기준)
    ref_coin = coins[0]
    if ref_coin not in coin_results:
        ref_coin = [c for c in coins if c in coin_results][0]

    port_results = []
    for win in coin_results[ref_coin]:
        ts, te = win["test_start"], win["test_end"]
        label  = win["label"]
        short  = win["short_lbl"]

        equities = []
        total_trades = 0
        all_pnls = []

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

        # 균등 배분 포트폴리오 equity (각 코인 1/n 비중)
        n = len(equities)
        port_eq = sum(eq / n for eq in equities)
        bah_eq  = win["bah"]  # BTC BaH 기준 유지

        metrics = compute_metrics(port_eq)
        wr = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0

        port_results.append({
            "label":        label,
            "short_lbl":    short,
            "test_start":   ts,
            "equity":       port_eq,
            "bah":          bah_eq,
            "metrics":      metrics,
            "n_trades":     total_trades,
            "n_coins":      n,
            "win_rate":     wr,
        })

    return port_results


# ── 결과 출력 ────────────────────────────────────────────────────────────────
def print_portfolio_results(port_results: list, name: str):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]

    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")
    print(f"{'기간':>20} {'Sharpe':>8} {'수익률':>9} {'거래수':>7} {'승률':>7}")
    print(f"{'-'*55}")
    for r, s in zip(port_results, sharpes):
        marker = "[+]" if s > 0 else "[-]"
        m = r["metrics"]
        print(f"  {marker} {r['label']:<18} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}%")
    print(f"{'-'*55}")
    pos = sum(s > 0 for s in sharpes)
    print(f"  평균 Sharpe: {np.mean(sharpes):.3f}  |  "
          f"양수 비율: {pos}/{len(sharpes)}  |  "
          f"총 거래: {sum(r['n_trades'] for r in port_results)}")


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_comparison(results_8: list, results_5: list, coin_results: dict):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    sharpes_8 = [float(r["metrics"]["Sharpe"]) for r in results_8]
    sharpes_5 = [float(r["metrics"]["Sharpe"]) for r in results_5]
    labels    = [r["short_lbl"] for r in results_8]
    n         = len(results_8)

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32)

    # 1. 포트폴리오 누적 수익 비교
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22"); ax1.spines[:].set_color("#30363d")
    ax1.tick_params(colors="#8b949e")

    colors_8 = plt.cm.plasma(np.linspace(0.15, 0.88, n))
    colors_5 = plt.cm.cool(np.linspace(0.2, 0.85, n))

    for i, (r8, r5) in enumerate(zip(results_8, results_5)):
        ax1.plot(r8["equity"].index, r8["equity"].values,
                 color=colors_8[i], linewidth=2.0,
                 label=f"T8 {r8['label']} Sharpe={float(r8['metrics']['Sharpe']):.3f}")
        ax1.plot(r5["equity"].index, r5["equity"].values,
                 color=colors_5[i], linewidth=1.2, linestyle="--", alpha=0.7)

    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("누적 수익 배율", color="#8b949e")
    ax1.set_title("Timeless 8-coin vs Original 5-coin Walk-forward (실선=T8 / 점선=O5)",
                  color="#e6edf3", fontsize=11)
    ax1.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=2)
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # 2. Sharpe 비교 막대
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161b22"); ax2.spines[:].set_color("#30363d")
    ax2.tick_params(colors="#8b949e")

    x = np.arange(n)
    w = 0.38
    bars8 = ax2.bar(x - w/2, sharpes_8, w,
                    color=["#3fb950" if s > 0 else "#ff7b72" for s in sharpes_8],
                    alpha=0.9, label="Timeless 8코인")
    bars5 = ax2.bar(x + w/2, sharpes_5, w,
                    color=["#58a6ff" if s > 0 else "#d2a8ff" for s in sharpes_5],
                    alpha=0.7, label="Original 5코인")

    for bar, v in zip(bars8, sharpes_8):
        yp = v + 0.01 if v >= 0 else v - 0.03
        ax2.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.2f}",
                 ha="center", fontsize=7.5, color="#e6edf3", fontweight="bold")

    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.axhline(np.mean(sharpes_8), color="#3fb950", linewidth=1.3,
                linestyle="--", label=f"T8 평균 {np.mean(sharpes_8):.3f}")
    ax2.axhline(np.mean(sharpes_5), color="#58a6ff", linewidth=1.3,
                linestyle=":", label=f"O5 평균 {np.mean(sharpes_5):.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")
    ax2.set_title("기간별 Sharpe: T8 vs O5", color="#e6edf3", fontsize=10)
    ax2.legend(fontsize=7.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    # 3. 코인별 WF Sharpe 히트맵
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161b22"); ax3.spines[:].set_color("#30363d")

    coin_labels = [LABEL.get(s, s) for s in TIMELESS_8 if s in coin_results]
    win_labels  = [r["short_lbl"] for r in coin_results[TIMELESS_8[0]]]
    matrix = np.zeros((len(coin_labels), len(win_labels)))

    for ci, sym in enumerate([s for s in TIMELESS_8 if s in coin_results]):
        for wi, win in enumerate(coin_results[sym]):
            matrix[ci, wi] = float(win["metrics"]["Sharpe"])

    vmax = max(abs(matrix).max(), 0.5)
    im = ax3.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax3, fraction=0.03)

    ax3.set_xticks(range(len(win_labels)))
    ax3.set_xticklabels(win_labels, color="#8b949e", fontsize=7, rotation=45, ha="right")
    ax3.set_yticks(range(len(coin_labels)))
    ax3.set_yticklabels(coin_labels, color="#e6edf3", fontsize=8)
    ax3.set_title("코인별 WF Sharpe 히트맵", color="#e6edf3", fontsize=10)

    for ci in range(len(coin_labels)):
        for wi in range(len(win_labels)):
            ax3.text(wi, ci, f"{matrix[ci,wi]:.2f}",
                     ha="center", va="center", fontsize=6.5,
                     color="black" if abs(matrix[ci,wi]) < vmax*0.7 else "white")

    # 4. 거래수 & 요약 테이블
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22"); ax4.spines[:].set_color("#30363d")
    ax4.tick_params(colors="#8b949e")

    n_trades_8 = [r["n_trades"] for r in results_8]
    n_trades_5 = [r["n_trades"] for r in results_5]
    ax4.bar(x - w/2, n_trades_8, w, color="#3fb950", alpha=0.8, label="Timeless 8코인")
    ax4.bar(x + w/2, n_trades_5, w, color="#58a6ff", alpha=0.7, label="Original 5코인")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, color="#8b949e", fontsize=8, rotation=30, ha="right")
    ax4.set_ylabel("거래수", color="#8b949e")
    ax4.set_title("기간별 포트폴리오 거래수", color="#e6edf3", fontsize=10)
    ax4.legend(fontsize=7.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    # 5. 요약 테이블
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22"); ax5.spines[:].set_color("#30363d")
    ax5.axis("off")

    pos8 = sum(s > 0 for s in sharpes_8)
    pos5 = sum(s > 0 for s in sharpes_5)
    rows = [
        ["지표", "Timeless 8코인", "Original 5코인"],
        ["평균 Sharpe", f"{np.mean(sharpes_8):.3f}", f"{np.mean(sharpes_5):.3f}"],
        ["Sharpe 표준편차", f"{np.std(sharpes_8):.3f}", f"{np.std(sharpes_5):.3f}"],
        ["양수 Sharpe", f"{pos8}/{n} ({pos8/n*100:.0f}%)", f"{pos5}/{n} ({pos5/n*100:.0f}%)"],
        ["총 거래수", str(sum(n_trades_8)), str(sum(n_trades_5))],
        ["코인 수", f"{len(TIMELESS_8)}개", "5개"],
    ]

    tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax5.set_title("요약 비교", color="#e6edf3", fontsize=10, pad=8)

    fig.suptitle("Exp16: Timeless 8-coin vs Original 5-coin Walk-forward",
                 color="#e6edf3", fontsize=13, fontweight="bold")

    path = RESULTS_DIR / "exp16_timeless_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장: {path}")
    plt.show()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Exp16: Timeless 8코인 포트폴리오 Walk-forward")
    print(f"코인: {', '.join(LABEL.get(s,s) for s in TIMELESS_8)}")
    print(f"기간: {START} ~ {END}")
    print(f"학습: {TRAIN_MONTHS}개월 / 테스트: {TEST_MONTHS}개월")
    print("=" * 65)

    # 온체인 데이터 (BTC 기반)
    print("\n[온체인 데이터 로드...]")
    funding   = collect_funding_rate("BTCUSDT", START, END)
    fg        = collect_fear_greed(START, END)

    all_syms = list(set(TIMELESS_8 + ORIGINAL_5))
    coin_data = {}

    print(f"\n[코인 데이터 로드 ({len(all_syms)}개)...]")
    for sym in all_syms:
        try:
            df  = collect(sym, "1h", START, END)
            mpe = rolling_mpe(df["close"], window=168,
                              cache_key=f"{sym}_1h_{START}_{END}")
            h_funding  = funding_entropy(funding, df.index)
            h_fg       = fear_greed_entropy(fg, df.index)
            h_onchain  = combined_onchain_entropy(h_funding, h_fg)
            coin_data[sym] = {"df": df, "mpe": mpe, "h": h_onchain}
            print(f"  [{LABEL.get(sym,sym):6s}] {len(df):,}봉")
        except Exception as e:
            print(f"  [{LABEL.get(sym,sym):6s}] 실패: {e}")

    print("\n[Walk-forward 실행 중...]")
    coin_results = {}
    for sym in TIMELESS_8:
        if sym not in coin_data:
            continue
        d = coin_data[sym]
        wf = run_coin_wf(sym, d["df"], d["mpe"], d["h"])
        coin_results[sym] = wf
        total_n = sum(w["n_trades"] for w in wf)
        sharpes = [float(w["metrics"]["Sharpe"]) for w in wf]
        avg_s   = np.mean(sharpes)
        pos_s   = sum(s > 0 for s in sharpes)
        print(f"  [{LABEL.get(sym,sym):6s}] 평균 Sharpe {avg_s:+.3f} | "
              f"양수 {pos_s}/{len(sharpes)} | 총 거래 {total_n}")

    # Original 5코인 WF
    coin_results_5 = {}
    for sym in ORIGINAL_5:
        if sym not in coin_data:
            continue
        d = coin_data[sym]
        wf = run_coin_wf(sym, d["df"], d["mpe"], d["h"])
        coin_results_5[sym] = wf

    # 포트폴리오 집계
    print("\n[포트폴리오 집계...]")
    avail_t8 = [s for s in TIMELESS_8 if s in coin_results]
    avail_o5 = [s for s in ORIGINAL_5 if s in coin_results_5]

    port_t8 = aggregate_portfolio(coin_results, avail_t8)
    port_o5 = aggregate_portfolio(coin_results_5, avail_o5)

    print_portfolio_results(port_t8, f"Timeless 8코인 포트폴리오 WF ({', '.join(LABEL.get(s,s) for s in avail_t8)})")
    print_portfolio_results(port_o5, f"Original 5코인 포트폴리오 WF ({', '.join(LABEL.get(s,s) for s in avail_o5)})")

    # 최종 비교
    sharpes_t8 = [float(r["metrics"]["Sharpe"]) for r in port_t8]
    sharpes_o5 = [float(r["metrics"]["Sharpe"]) for r in port_o5]

    print(f"\n{'='*65}")
    print(f"최종 비교")
    print(f"{'='*65}")
    print(f"  Timeless 8코인 평균 Sharpe: {np.mean(sharpes_t8):.3f}  "
          f"(양수 {sum(s>0 for s in sharpes_t8)}/{len(sharpes_t8)})")
    print(f"  Original 5코인  평균 Sharpe: {np.mean(sharpes_o5):.3f}  "
          f"(양수 {sum(s>0 for s in sharpes_o5)}/{len(sharpes_o5)})")

    if np.mean(sharpes_t8) > np.mean(sharpes_o5):
        print(f"\n  [결론] Timeless 8코인이 Original 5코인 대비 "
              f"+{np.mean(sharpes_t8) - np.mean(sharpes_o5):.3f} Sharpe 우위")
    else:
        print(f"\n  [결론] Original 5코인이 Timeless 8코인 대비 "
              f"+{np.mean(sharpes_o5) - np.mean(sharpes_t8):.3f} Sharpe 우위")

    plot_comparison(port_t8, port_o5, coin_results)


if __name__ == "__main__":
    main()
