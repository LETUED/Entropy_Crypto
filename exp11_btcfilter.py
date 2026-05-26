"""
실험 11: BTC 하락장 필터 + 5코인 Walk-forward
목적: 2022-07~12 크래시 구간 방어 — BTC MA200 아래이면 포트폴리오 전체 진입 금지

근거:
  2022-07~12에서 AVAX(-1.41), ADA(-1.41), DOT(-1.46) 동시 손실
  당시 BTC는 MA200 훨씬 아래 → 시스템 리스크 구간
  포트폴리오 레벨 필터로 이 구간 전체를 차단

설계:
  BTC_MA200_FILTER = True  → BTC 가격 < BTC MA200 이면 전코인 진입 금지
  BTC_MA200_FILTER = False → 기존 5코인 WF (Exp10 결과)

비교: Exp10(필터없음) vs Exp11(BTC MA200 필터)

실행: py run_btcfilter_walkforward.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.walkforward import run_walkforward, print_walkforward
from src.analysis.h4_backtest import compute_metrics, MA_PERIOD

RESULTS_DIR = Path("results")
START, END   = "2021-01-01", "2025-01-01"

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
COIN_LABELS = {
    "BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
    "ADAUSDT": "ADA", "DOTUSDT": "DOT",
}
COIN_COLORS = ["#f0c040", "#56d364", "#d2a8ff", "#ffab70", "#58a6ff"]

TRAIN_MONTHS = 12
TEST_MONTHS  = 6


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def make_btc_filter(btc_df: pd.DataFrame) -> pd.Series:
    """BTC 가격 > BTC MA200 이면 True (진입 허용), 아니면 False (진입 금지)"""
    ma200 = btc_df["close"].rolling(MA_PERIOD).mean()
    return (btc_df["close"] > ma200).rename("btc_above_ma200")


def run_coins(funding, fg, btc_filter=None):
    """5코인 Walk-forward 실행. btc_filter=None이면 필터 없음."""
    coin_results = {}
    for sym in COINS:
        label = COIN_LABELS[sym]
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168)
        h_onchain = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        # BTC 필터는 공통 인덱스로 정렬
        mkt = btc_filter.reindex(df.index).ffill() if btc_filter is not None else None

        results = run_walkforward(df, mpe, h_onchain,
                                  train_months=TRAIN_MONTHS,
                                  test_months=TEST_MONTHS,
                                  step_months=TEST_MONTHS,
                                  market_filter=mkt)
        coin_results[sym] = results
        sharpes = [float(r["metrics"]["Sharpe"]) for r in results]
        pos = sum(s > 0 for s in sharpes)
        print(f"  [{label}] 평균 Sharpe {np.mean(sharpes):+.3f} | {pos}/{len(sharpes)} 양수")
    return coin_results


def aggregate(coin_results):
    """코인별 구간 결과 통합"""
    ref = coin_results["BTCUSDT"]
    windows = []
    for btc_win in ref:
        ts, te = btc_win["test_start"], btc_win["test_end"]
        sl = btc_win["short_label"]
        sharpes, equity_list = [], []
        for sym in COINS:
            match = next((w for w in coin_results[sym]
                          if w["test_start"] == ts and w["test_end"] == te), None)
            if match:
                sharpes.append(float(match["metrics"]["Sharpe"]))
                equity_list.append(match["equity"])
        if not sharpes:
            continue
        port = pd.concat(equity_list, axis=1).dropna().mean(axis=1) if equity_list else None
        windows.append({
            "label":       btc_win["label"],
            "short_label": sl,
            "avg_sharpe":  np.mean(sharpes),
            "sharpes":     sharpes,
            "port_equity": port,
            "port_metrics": compute_metrics(port) if port is not None else {},
        })
    return windows


def print_comparison(wins_no, wins_btc):
    """필터 없음 vs BTC 필터 구간별 비교"""
    print("\n" + "=" * 90)
    print(f"{'구간':<22} | {'필터없음 평균':>13} {'포트Sharpe':>11} | "
          f"{'BTC필터 평균':>13} {'포트Sharpe':>11} | {'포트 차이':>10}")
    print("-" * 90)

    avg_port_no, avg_port_btc = [], []
    for w_no, w_bt in zip(wins_no, wins_btc):
        pn = float(str(w_no["port_metrics"].get("Sharpe", "0")))
        pb = float(str(w_bt["port_metrics"].get("Sharpe", "0")))
        diff = pb - pn
        sign = "+" if diff >= 0 else ""
        marker = "★" if pb > pn else "▼" if pb < pn else " "
        print(f"{marker} {w_no['label']:<20} | {w_no['avg_sharpe']:>+13.3f} {pn:>+11.3f} | "
              f"{w_bt['avg_sharpe']:>+13.3f} {pb:>+11.3f} | {sign}{diff:>+9.3f}")
        avg_port_no.append(pn); avg_port_btc.append(pb)

    print("=" * 90)
    dn = np.mean(avg_port_no); db = np.mean(avg_port_btc)
    diff_avg = db - dn
    sign = "+" if diff_avg >= 0 else ""
    print(f"{'평균':<22} | {'':>13} {dn:>+11.3f} | {'':>13} {db:>+11.3f} | "
          f"{sign}{diff_avg:>+9.3f}")
    print(f"\n  필터없음 포트 양수 구간: {sum(s>0 for s in avg_port_no)}/{len(avg_port_no)}")
    print(f"  BTC필터  포트 양수 구간: {sum(s>0 for s in avg_port_btc)}/{len(avg_port_btc)}")


def plot_comparison(wins_no, wins_btc):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    n = len(wins_no)
    wlabels = [w["short_label"] for w in wins_no]
    ports_no  = [float(str(w["port_metrics"].get("Sharpe","0"))) for w in wins_no]
    ports_btc = [float(str(w["port_metrics"].get("Sharpe","0"))) for w in wins_btc]
    avgs_no   = [w["avg_sharpe"] for w in wins_no]
    avgs_btc  = [w["avg_sharpe"] for w in wins_btc]

    fig = plt.figure(figsize=(22, 20), facecolor="#0d1117")
    fig.suptitle("실험 11: BTC MA200 하락장 필터 효과 (5코인 포트폴리오 WF)\n"
                 "필터없음(Exp10) vs BTC MA200 필터(Exp11)",
                 color="#e6edf3", fontsize=13, y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2.5, 1.3, 1.3])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--")

    win_colors = plt.cm.plasma(np.linspace(0.15, 0.88, n))

    # ── 1. 포트폴리오 누적 수익 비교 ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    for i, (w_no, w_bt) in enumerate(zip(wins_no, wins_btc)):
        color = win_colors[i]
        if w_no["port_equity"] is not None:
            ax1.plot(w_no["port_equity"].index, w_no["port_equity"].values,
                     color=color, linewidth=1.0, alpha=0.4, linestyle="--",
                     label=f"_{w_no['short_label']} 필터없음")
        if w_bt["port_equity"] is not None:
            ax1.plot(w_bt["port_equity"].index, w_bt["port_equity"].values,
                     color=color, linewidth=2.0, alpha=0.9,
                     label=f"{w_bt['label']} BTC필터(S={w_bt['avg_sharpe']:+.2f})")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 (실선=BTC필터 / 점선=필터없음)", color="#e6edf3")
    ax1.set_ylabel("누적 배율", color="#8b949e")
    handles = [h for h in ax1.get_legend_handles_labels()[0]
               if not ax1.get_legend_handles_labels()[1][
                   ax1.get_legend_handles_labels()[0].index(h)].startswith("_")]
    labels_leg = [l for l in ax1.get_legend_handles_labels()[1] if not l.startswith("_")]
    ax1.legend(handles, labels_leg, fontsize=7.5, facecolor="#21262d",
               edgecolor="#30363d", labelcolor="#e6edf3", loc="upper left", ncol=2)

    # ── 2. 포트폴리오 Sharpe 비교 바 ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    x = np.arange(n); w = 0.35
    b1 = ax2.bar(x - w/2, ports_no,  w, color="#79c0ff", alpha=0.85, label="필터없음 (Exp10)")
    b2 = ax2.bar(x + w/2, ports_btc, w, color="#56d364", alpha=0.85, label="BTC필터 (Exp11)")
    for bar, val in zip(list(b1)+list(b2), ports_no+ports_btc):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.03 if val >= 0 else -0.12),
                 f"{val:+.2f}", ha="center", color="#e6edf3", fontsize=8.5)
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.axhline(np.mean(ports_no),  color="#79c0ff", linewidth=1.0, linestyle="--",
                alpha=0.6, label=f"평균 필터없음 {np.mean(ports_no):+.3f}")
    ax2.axhline(np.mean(ports_btc), color="#56d364", linewidth=1.0, linestyle="--",
                alpha=0.6, label=f"평균 BTC필터 {np.mean(ports_btc):+.3f}")
    ax2.set_xticks(x); ax2.set_xticklabels(wlabels, rotation=30)
    ax2.set_title("구간별 포트폴리오 Sharpe 비교", color="#e6edf3")
    ax2.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 3. 5코인 평균 Sharpe 비교 바 ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    b3 = ax3.bar(x - w/2, avgs_no,  w, color="#79c0ff", alpha=0.85, label="필터없음 (Exp10)")
    b4 = ax3.bar(x + w/2, avgs_btc, w, color="#56d364", alpha=0.85, label="BTC필터 (Exp11)")
    for bar, val in zip(list(b3)+list(b4), avgs_no+avgs_btc):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.03 if val >= 0 else -0.12),
                 f"{val:+.2f}", ha="center", color="#e6edf3", fontsize=8.5)
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.axhline(np.mean(avgs_no),  color="#79c0ff", linewidth=1.0, linestyle="--", alpha=0.6)
    ax3.axhline(np.mean(avgs_btc), color="#56d364", linewidth=1.0, linestyle="--", alpha=0.6)
    ax3.set_xticks(x); ax3.set_xticklabels(wlabels, rotation=30)
    ax3.set_title("구간별 5코인 평균 Sharpe 비교", color="#e6edf3")
    ax3.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. BTC MA200 구간 비율 시각화 ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style(ax4)
    diff_port = [pb - pn for pb, pn in zip(ports_btc, ports_no)]
    colors_d  = ["#56d364" if d >= 0 else "#ff7b72" for d in diff_port]
    bars_d = ax4.bar(range(n), diff_port, color=colors_d, alpha=0.85)
    for bar, val in zip(bars_d, diff_port):
        ax4.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.02 if val >= 0 else -0.12),
                 f"{val:+.2f}", ha="center", color="#e6edf3", fontsize=9)
    ax4.axhline(0, color="#8b949e", linewidth=0.8)
    ax4.axhline(np.mean(diff_port), color="#f7931a", linewidth=1.3, linestyle="--",
                label=f"평균 차이 {np.mean(diff_port):+.3f}")
    ax4.set_xticks(range(n)); ax4.set_xticklabels(wlabels, rotation=30)
    ax4.set_title("BTC 필터 Sharpe 개선량 (BTC필터 - 필터없음)", color="#e6edf3")
    ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 5. 요약 테이블 ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style(ax5); ax5.axis("off")
    pos_no  = sum(s > 0 for s in ports_no)
    pos_btc = sum(s > 0 for s in ports_btc)
    rows = [
        ["지표",                    "필터없음 (Exp10)", "BTC필터 (Exp11)"],
        ["평균 포트폴리오 Sharpe",  f"{np.mean(ports_no):+.3f}", f"{np.mean(ports_btc):+.3f}"],
        ["포트 양수 구간",          f"{pos_no}/{n} ({pos_no/n*100:.0f}%)",
                                    f"{pos_btc}/{n} ({pos_btc/n*100:.0f}%)"],
        ["평균 5코인 Sharpe",       f"{np.mean(avgs_no):+.3f}", f"{np.mean(avgs_btc):+.3f}"],
        ["Sharpe 표준편차",         f"{np.std(ports_no):.3f}", f"{np.std(ports_btc):.3f}"],
        ["BTC필터 평균 개선",       "—", f"{np.mean(diff_port):+.3f}"],
    ]
    tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if r == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax5.set_title("필터 효과 요약", color="#e6edf3", fontsize=10, pad=8)

    plt.savefig(RESULTS_DIR / "btcfilter_walkforward.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/btcfilter_walkforward.png")


def main():
    print("=" * 70)
    print("실험 11: BTC MA200 하락장 필터 + 5코인 Walk-forward")
    print(f"  코인: BTC · SOL · AVAX · ADA · DOT")
    print(f"  기간: {START} ~ {END}")
    print(f"  필터: BTC 가격 < BTC MA200 → 포트폴리오 전체 진입 금지")
    print("=" * 70)

    print("\n[공통 온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[BTC MA200 필터 계산...]")
    btc_df     = collect("BTCUSDT", "1h", START, END)
    btc_filter = make_btc_filter(btc_df)
    btc_above  = btc_filter.mean() * 100
    print(f"  BTC MA200 위 비율: {btc_above:.1f}% (아래 {100-btc_above:.1f}%)")

    # 필터 없음 (Exp10 재실행)
    print("\n[필터없음 (Exp10 재실행)...]")
    coin_results_no = run_coins(funding, fg, btc_filter=None)
    wins_no = aggregate(coin_results_no)

    # BTC MA200 필터 적용 (Exp11)
    print("\n[BTC MA200 필터 적용 (Exp11)...]")
    coin_results_btc = run_coins(funding, fg, btc_filter=btc_filter)
    wins_btc = aggregate(coin_results_btc)

    print_comparison(wins_no, wins_btc)
    plot_comparison(wins_no, wins_btc)


if __name__ == "__main__":
    main()
