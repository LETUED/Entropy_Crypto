"""
실험 13: 코인 확장 포트폴리오 (5코인 → 10코인)
목적: 연간 거래 빈도 증가 + 새 코인의 기여도 평가

확장 코인: BTC SOL AVAX ADA DOT  (기존 5)
           ETH BNB DOGE LINK MATIC (신규 5)

설계:
  총 자본: 10,000 USDT (코인당 1,000 USDT × 10)
  전략: 저엔트로피 롱 + Kelly + MA200 + 온체인 (Exp12 동일)
  비교:
    A. 기존 5코인 포트폴리오 (Exp12 재현)
    B. 확장 10코인 포트폴리오 (신규)

핵심 출력:
  1. 코인별 Sharpe / PnL / 거래수 히트맵
  2. 5코인 vs 10코인 누적 수익 비교
  3. 연간 거래 빈도 비교
  4. 확장 포트폴리오 운용 가이드

실행: py exp13_expanded_portfolio.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import kelly_size, compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H

RESULTS_DIR   = Path("results")
START, END    = "2021-01-01", "2025-01-01"
TOTAL_CAPITAL = 10_000   # USDT

# 기존 5코인
COINS_5 = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]

# 확장 10코인 (기존 5 + 신규 5)
COINS_10 = [
    "BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
    "ETHUSDT", "BNBUSDT",  "DOGEUSDT", "LINKUSDT", "MATICUSDT",
]

COIN_LABELS = {
    "BTCUSDT": "BTC",  "SOLUSDT": "SOL",   "AVAXUSDT": "AVAX",
    "ADAUSDT": "ADA",  "DOTUSDT": "DOT",   "ETHUSDT":  "ETH",
    "BNBUSDT": "BNB",  "DOGEUSDT": "DOGE", "LINKUSDT": "LINK",
    "MATICUSDT": "MATIC",
}

COIN_COLORS = {
    "BTCUSDT":  "#f0c040", "SOLUSDT":  "#56d364", "AVAXUSDT": "#d2a8ff",
    "ADAUSDT":  "#ffab70", "DOTUSDT":  "#58a6ff", "ETHUSDT":  "#79c0ff",
    "BNBUSDT":  "#f78166", "DOGEUSDT": "#3fb950", "LINKUSDT": "#bc8cff",
    "MATICUSDT":"#ff7b72",
}


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────────────────
# 코인 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────

def prepare_coin(sym: str, df, mpe, h_onchain) -> dict:
    """단일 코인 전략 파라미터 사전 계산"""
    rsi    = compute_rsi(df["close"])
    ma200  = df["close"].rolling(MA_PERIOD).mean()
    sig    = generate_signals(rsi)
    thresh = np.percentile(mpe.dropna(), ENTROPY_PCT)
    oc_thresh = np.percentile(h_onchain.dropna(), 40) if h_onchain is not None else None
    return {
        "df": df, "mpe": mpe, "h_oc": h_onchain,
        "rsi": rsi, "ma200": ma200, "sig": sig,
        "thresh": thresh, "oc_thresh": oc_thresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 포트폴리오 시뮬레이터 (Exp12와 동일 로직, 코인 리스트만 가변)
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio(coin_data: dict, total_capital: float = TOTAL_CAPITAL) -> dict:
    """
    멀티코인 동시 운용 시뮬레이터.
    coin_data: {symbol: prepared_dict (from prepare_coin)}
    총 자본을 코인 수로 균등 배분, 각 코인 독립 운용.
    """
    symbols    = list(coin_data.keys())
    n_coins    = len(symbols)
    coin_cap   = total_capital / n_coins

    coin_equity = {sym: coin_cap for sym in symbols}
    positions   = {}
    all_trades  = []

    # 공통 시간축 (BTC 기준)
    common_idx = coin_data[symbols[0]]["df"].index

    portfolio_curve  = []
    deployed_curve   = []
    concurrent_curve = []
    coin_eq_curve    = {sym: [] for sym in symbols}

    for i, idx in enumerate(common_idx):
        deployed_usdt = 0.0

        for sym in symbols:
            prep  = coin_data[sym]
            df    = prep["df"]
            if idx not in df.index:
                coin_eq_curve[sym].append(coin_equity[sym])
                continue

            price   = df["close"].loc[idx]
            rsi_val = prep["rsi"].loc[idx]  if idx in prep["rsi"].index  else 50.0
            ma_val  = prep["ma200"].loc[idx] if idx in prep["ma200"].index else np.nan
            mpe_val = prep["mpe"].loc[idx]   if idx in prep["mpe"].index  else np.nan
            sig_val = prep["sig"].loc[idx]   if idx in prep["sig"].index  else 0

            oc_ok = True
            if prep["h_oc"] is not None and prep["oc_thresh"] is not None:
                oc_val = prep["h_oc"].loc[idx] if idx in prep["h_oc"].index else 0
                oc_ok  = oc_val <= prep["oc_thresh"]

            # ── 청산 ─────────────────────────────────────────────────────
            if sym in positions:
                pos  = positions[sym]
                held = i - pos["entry_hour"]
                ep   = pos["entry_price"]
                if rsi_val > 50 or held >= MAX_HOLD_H:
                    pnl_pct  = (price - ep) / ep
                    pnl_usdt = pnl_pct * pos["pos_size_usdt"]
                    fee      = pos["pos_size_usdt"] * FEE_RATE
                    coin_equity[sym] += pnl_usdt - fee
                    all_trades.append({
                        "symbol":      sym,
                        "entry_time":  pos["entry_time"],
                        "exit_time":   idx,
                        "entry_price": ep,
                        "exit_price":  price,
                        "pnl_pct":     pnl_pct,
                        "pnl_usdt":    pnl_usdt - fee,
                        "kelly_frac":  pos["kelly_frac"],
                        "held_h":      held,
                    })
                    del positions[sym]

            # ── 진입 ─────────────────────────────────────────────────────
            if sym not in positions:
                if not np.isnan(mpe_val) and not np.isnan(ma_val):
                    is_low   = mpe_val <= prep["thresh"]
                    above_ma = price > ma_val
                    k_frac   = kelly_size(mpe_val, prep["mpe"])
                    if sig_val == 1 and is_low and above_ma and k_frac > 0 and oc_ok:
                        pos_usdt  = coin_equity[sym] * k_frac
                        entry_fee = pos_usdt * FEE_RATE
                        coin_equity[sym] -= entry_fee
                        positions[sym] = {
                            "entry_price":   price * (1 + FEE_RATE),
                            "entry_time":    idx,
                            "entry_hour":    i,
                            "pos_size_usdt": pos_usdt,
                            "kelly_frac":    k_frac,
                        }

            # 미실현 평가
            eq = coin_equity[sym]
            if sym in positions:
                pos    = positions[sym]
                unreal = (price - pos["entry_price"]) / pos["entry_price"] * pos["pos_size_usdt"]
                eq    += unreal
                deployed_usdt += pos["pos_size_usdt"]
            coin_eq_curve[sym].append(eq)

        total_eq = sum(
            coin_equity[s] + (
                ((coin_data[s]["df"]["close"].loc[idx] - positions[s]["entry_price"])
                 / positions[s]["entry_price"] * positions[s]["pos_size_usdt"])
                if s in positions and idx in coin_data[s]["df"].index else 0
            )
            for s in symbols
        )
        portfolio_curve.append(total_eq)
        deployed_curve.append(deployed_usdt)
        concurrent_curve.append(len(positions))

    trades_df   = pd.DataFrame(all_trades)
    port_equity = pd.Series(portfolio_curve, index=common_idx)
    deploy_ser  = pd.Series(deployed_curve,  index=common_idx)
    conc_ser    = pd.Series(concurrent_curve, index=common_idx)
    coin_eq_ser = {
        sym: pd.Series(coin_eq_curve[sym], index=common_idx[:len(coin_eq_curve[sym])])
        for sym in symbols
    }

    metrics = compute_metrics(port_equity / total_capital)
    return {
        "portfolio_equity": port_equity,
        "deployed":         deploy_ser,
        "concurrent":       conc_ser,
        "trades":           trades_df,
        "coin_equity":      coin_eq_ser,
        "final_capital":    port_equity.iloc[-1],
        "coin_capital":     coin_cap,
        "metrics":          metrics,
        "symbols":          symbols,
        "total_capital":    total_capital,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, label: str):
    trades  = result["trades"]
    metrics = result["metrics"]
    cap     = result["total_capital"]
    n_coins = len(result["symbols"])
    coin_cap= result["coin_capital"]
    closed  = trades.dropna(subset=["pnl_usdt"]) if len(trades) else pd.DataFrame()
    annual_n = len(closed) / 4 if len(closed) else 0

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  총 자본: {cap:,} USDT  |  코인당: {coin_cap:,.0f} USDT × {n_coins}개")
    print(f"{'=' * 70}")
    print(f"\n[포트폴리오 성과]")
    print(f"  총 수익률  : {metrics['총 수익률']}")
    print(f"  Sharpe     : {metrics['Sharpe']}")
    print(f"  최대 낙폭  : {metrics['최대 낙폭']}")
    print(f"  최종 자산  : {result['final_capital']:,.0f} USDT  "
          f"(+{result['final_capital']-cap:,.0f} USDT)")

    print(f"\n[거래 통계]")
    if len(closed):
        wr       = (closed["pnl_pct"] > 0).mean() * 100
        avg_pct  = closed["pnl_pct"].mean() * 100
        avg_usdt = closed["pnl_usdt"].mean()
        total_p  = closed["pnl_usdt"].sum()
        avg_held = closed["held_h"].mean()
        print(f"  총 거래 수   : {len(closed)}건 (4년)")
        print(f"  연간 거래 수 : {annual_n:.1f}건/년")
        print(f"  승률         : {wr:.1f}%")
        print(f"  평균 수익    : {avg_pct:+.3f}% / {avg_usdt:+.2f} USDT/건")
        print(f"  총 수익      : {total_p:+,.2f} USDT")
        print(f"  평균 보유    : {avg_held:.1f}H")

    print(f"\n[코인별 기여도]")
    print(f"  {'코인':<6} {'거래':>6} {'승률':>8} {'평균PnL%':>10} {'평균PnL(U)':>12} {'총PnL(U)':>12}")
    print(f"  {'-'*60}")
    for sym in result["symbols"]:
        sub = closed[closed["symbol"] == sym] if len(closed) else pd.DataFrame()
        if not len(sub):
            print(f"  {COIN_LABELS[sym]:<6} {'0':>6}")
            continue
        wr_c   = (sub["pnl_pct"] > 0).mean() * 100
        avg_c  = sub["pnl_pct"].mean() * 100
        avgu_c = sub["pnl_usdt"].mean()
        tot_c  = sub["pnl_usdt"].sum()
        marker = " [+]" if tot_c > 0 else " [-]"
        print(f"  {COIN_LABELS[sym]:<6} {len(sub):>6}건 {wr_c:>7.1f}% "
              f"{avg_c:>+9.3f}% {avgu_c:>+11.2f} {tot_c:>+11.2f}{marker}")

    # 동시 포지션 / 배포율
    conc    = result["concurrent"]
    dep_pct = result["deployed"] / cap * 100
    print(f"\n[자본 효율]")
    print(f"  평균 배포율: {dep_pct.mean():.1f}%")
    print(f"  최대 배포율: {dep_pct.max():.1f}%")
    print(f"  최대 동시 포지션: {conc.max()}개")


def print_comparison(r5: dict, r10: dict):
    """5코인 vs 10코인 핵심 비교"""
    def n_trades(r):
        t = r["trades"].dropna(subset=["pnl_usdt"]) if len(r["trades"]) else pd.DataFrame()
        return len(t)
    def wr(r):
        t = r["trades"].dropna(subset=["pnl_usdt"]) if len(r["trades"]) else pd.DataFrame()
        return (t["pnl_pct"] > 0).mean() * 100 if len(t) else 0.0

    print(f"\n{'=' * 70}")
    print(f"  실험 13 비교 요약 — 5코인 vs 10코인")
    print(f"{'=' * 70}")
    rows = [
        ("지표",                  "5코인 (Exp12)",                    "10코인 (Exp13)"),
        ("Sharpe",                r5["metrics"]["Sharpe"],             r10["metrics"]["Sharpe"]),
        ("최대 낙폭",             r5["metrics"]["최대 낙폭"],          r10["metrics"]["최대 낙폭"]),
        ("총 수익률",             r5["metrics"]["총 수익률"],          r10["metrics"]["총 수익률"]),
        ("총 거래수 (4년)",       f"{n_trades(r5)}건",                 f"{n_trades(r10)}건"),
        ("연간 거래수",           f"{n_trades(r5)/4:.1f}건/년",        f"{n_trades(r10)/4:.1f}건/년"),
        ("승률",                  f"{wr(r5):.1f}%",                   f"{wr(r10):.1f}%"),
        ("평균 배포율",           f"{r5['deployed'].div(r5['total_capital']).mean()*100:.1f}%",
                                  f"{r10['deployed'].div(r10['total_capital']).mean()*100:.1f}%"),
        ("최대 동시 포지션",      f"{r5['concurrent'].max()}개",        f"{r10['concurrent'].max()}개"),
    ]
    header = rows[0]
    print(f"  {header[0]:<20} {header[1]:<22} {header[2]}")
    print(f"  {'-'*60}")
    for row in rows[1:]:
        print(f"  {row[0]:<20} {str(row[1]):<22} {row[2]}")


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(r5: dict, r10: dict, all_coin_data: dict):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(24, 28), facecolor="#0d1117")
    fig.suptitle(
        "실험 13: 코인 확장 포트폴리오 (5코인 → 10코인)\n"
        f"5코인 Sharpe {r5['metrics']['Sharpe']}  vs  "
        f"10코인 Sharpe {r10['metrics']['Sharpe']}",
        color="#e6edf3", fontsize=13, y=0.99
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2.5, 1.5, 1.5, 1.5])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--")

    def _xfmt(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color="#8b949e", fontsize=7)

    # ── 1. 포트폴리오 누적 자산 비교 ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    eq5  = r5["portfolio_equity"]
    eq10 = r10["portfolio_equity"]
    ax1.plot(eq5.index,  eq5.values,  color="#56d364", linewidth=2.5, label="5코인 포트폴리오")
    ax1.plot(eq10.index, eq10.values, color="#f0c040", linewidth=2.5,
             linestyle="--", label="10코인 포트폴리오")
    ax1.axhline(TOTAL_CAPITAL, color="#8b949e", linewidth=0.8, linestyle=":", label="원금")
    # 개별 코인 얇게 (신규 5개만)
    for sym in ["ETHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT"]:
        eq = r10["coin_equity"].get(sym)
        if eq is not None:
            ax1.plot(eq.index, eq.values,
                     color=COIN_COLORS[sym], linewidth=0.7, alpha=0.45,
                     label=f"{COIN_LABELS[sym]}")
    ax1.set_ylabel("자산 (USDT)", color="#8b949e")
    ax1.set_title("5코인 vs 10코인 포트폴리오 누적 자산 (점선=10코인 / 얇은선=신규 코인별)",
                  color="#e6edf3")
    ax1.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=4)
    _xfmt(ax1)

    # ── 2. 코인별 총 PnL (10코인) ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    closed10 = r10["trades"].dropna(subset=["pnl_usdt"]) if len(r10["trades"]) else pd.DataFrame()
    if len(closed10):
        syms   = r10["symbols"]
        totals = [closed10[closed10["symbol"]==s]["pnl_usdt"].sum() for s in syms]
        colors_bar = ["#56d364" if t >= 0 else "#ff7b72" for t in totals]
        bars = ax2.bar([COIN_LABELS[s] for s in syms], totals,
                       color=colors_bar, alpha=0.85)
        for bar, val in zip(bars, totals):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height() + (3 if val >= 0 else -25),
                     f"{val:+,.0f}", ha="center", color="#e6edf3", fontsize=8)
        ax2.axhline(0, color="#8b949e", linewidth=0.8)
        ax2.set_title("10코인별 총 PnL (USDT) — 초록=수익 / 빨강=손실", color="#e6edf3")
        ax2.set_ylabel("PnL (USDT)", color="#8b949e")
        ax2.tick_params(axis="x", colors="#8b949e", labelsize=8)

    # ── 3. 코인별 거래수 (신규 5 vs 기존 5) ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    if len(closed10):
        syms = r10["symbols"]
        n_list = [len(closed10[closed10["symbol"]==s]) for s in syms]
        colors_n = ["#f0c040" if s in COINS_5 else "#79c0ff" for s in syms]
        bars = ax3.bar([COIN_LABELS[s] for s in syms], n_list, color=colors_n, alpha=0.85)
        for bar, n in zip(bars, n_list):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     str(n), ha="center", color="#e6edf3", fontsize=9)
        ax3.set_title("코인별 총 거래수 (노란=기존 5 / 파란=신규 5)", color="#e6edf3")
        ax3.set_ylabel("거래수 (건)", color="#8b949e")
        ax3.tick_params(axis="x", colors="#8b949e", labelsize=8)
        from matplotlib.patches import Patch
        ax3.legend(handles=[
            Patch(facecolor="#f0c040", label="기존 5코인"),
            Patch(facecolor="#79c0ff", label="신규 5코인"),
        ], fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. 코인별 Sharpe (개인 equity 기반) ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style(ax4)
    sharpes = []
    for sym in r10["symbols"]:
        eq   = r10["coin_equity"][sym]
        init = r10["coin_capital"]
        m    = compute_metrics(eq / init)
        try:
            s = float(m["Sharpe"])
        except Exception:
            s = 0.0
        sharpes.append(s)
    colors_s = ["#56d364" if s >= 0 else "#ff7b72" for s in sharpes]
    bars4 = ax4.bar([COIN_LABELS[s] for s in r10["symbols"]], sharpes,
                    color=colors_s, alpha=0.85)
    for bar, val in zip(bars4, sharpes):
        ax4.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.02 if val >= 0 else -0.08),
                 f"{val:+.3f}", ha="center", color="#e6edf3", fontsize=8)
    ax4.axhline(0, color="#8b949e", linewidth=0.8)
    ax4.set_title("코인별 개인 Sharpe (10코인)", color="#e6edf3")
    ax4.set_ylabel("Sharpe", color="#8b949e")
    ax4.tick_params(axis="x", colors="#8b949e", labelsize=8)

    # ── 5. 연간 거래 빈도 비교 ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style(ax5)
    closed5 = r5["trades"].dropna(subset=["pnl_usdt"]) if len(r5["trades"]) else pd.DataFrame()
    for yrs in range(4):
        y_start = pd.Timestamp(f"202{yrs+1}-01-01")
        y_end   = pd.Timestamp(f"202{yrs+2}-01-01")
        n5  = len(closed5[(closed5["entry_time"] >= y_start) & (closed5["entry_time"] < y_end)]) if len(closed5) else 0
        n10 = len(closed10[(closed10["entry_time"] >= y_start) & (closed10["entry_time"] < y_end)]) if len(closed10) else 0
        x   = yrs * 3
        ax5.bar(x,   n5,  color="#56d364", alpha=0.85, width=1.2, label="5코인" if yrs==0 else "")
        ax5.bar(x+1.3, n10, color="#f0c040", alpha=0.85, width=1.2, label="10코인" if yrs==0 else "")
        ax5.text(x,     n5+0.2,  str(n5),  ha="center", color="#e6edf3", fontsize=8)
        ax5.text(x+1.3, n10+0.2, str(n10), ha="center", color="#e6edf3", fontsize=8)
    ax5.set_xticks([1, 4, 7, 10])
    ax5.set_xticklabels(["2021", "2022", "2023", "2024"], color="#8b949e")
    ax5.set_title("연도별 거래 빈도 (초록=5코인 / 노란=10코인)", color="#e6edf3")
    ax5.set_ylabel("거래수 (건)", color="#8b949e")
    ax5.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 6. 요약 비교 테이블 ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    _style(ax6); ax6.axis("off")

    def _wr(r):
        t = r["trades"].dropna(subset=["pnl_usdt"]) if len(r["trades"]) else pd.DataFrame()
        return f"{(t['pnl_pct']>0).mean()*100:.1f}%" if len(t) else "—"
    def _n(r):
        t = r["trades"].dropna(subset=["pnl_usdt"]) if len(r["trades"]) else pd.DataFrame()
        return len(t)

    rows_tbl = [
        ["지표",              "5코인 (Exp12)",                  "10코인 (Exp13)",             "판정"],
        ["Sharpe",            r5["metrics"]["Sharpe"],           r10["metrics"]["Sharpe"],      ""],
        ["최대 낙폭",         r5["metrics"]["최대 낙폭"],        r10["metrics"]["최대 낙폭"],   ""],
        ["총 수익률",         r5["metrics"]["총 수익률"],        r10["metrics"]["총 수익률"],   ""],
        ["총 거래수 (4년)",   f"{_n(r5)}건",                    f"{_n(r10)}건",               ""],
        ["연간 거래수",       f"{_n(r5)/4:.1f}건/년",           f"{_n(r10)/4:.1f}건/년",      ""],
        ["승률",              _wr(r5),                           _wr(r10),                     ""],
        ["평균 배포율",       f"{r5['deployed'].div(r5['total_capital']).mean()*100:.1f}%",
                              f"{r10['deployed'].div(r10['total_capital']).mean()*100:.1f}%",  ""],
        ["코인당 배분",       f"{r5['coin_capital']:,.0f} USDT",
                              f"{r10['coin_capital']:,.0f} USDT",                             ""],
    ]

    # 판정 자동 입력
    for row in rows_tbl[1:]:
        try:
            v5  = float(str(row[1]).replace("%","").replace("건","").replace("/년","").replace("USDT","").strip())
            v10 = float(str(row[2]).replace("%","").replace("건","").replace("/년","").replace("USDT","").strip())
        except Exception:
            continue
        if row[0] in ("Sharpe", "총 수익률", "승률", "총 거래수 (4년)", "연간 거래수", "평균 배포율"):
            row[3] = "10코인 ↑" if v10 > v5 else ("동일" if v10 == v5 else "5코인 ↑")
        elif row[0] == "최대 낙폭":
            row[3] = "10코인 ↑" if v10 > v5 else ("동일" if v10 == v5 else "5코인 ↑")

    tbl = ax6.table(cellText=rows_tbl[1:], colLabels=rows_tbl[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#21262d")
        else:
            cell.set_facecolor("#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax6.set_title("5코인 vs 10코인 비교 요약", color="#e6edf3", fontsize=10, pad=8)

    out_path = RESULTS_DIR / "exp13_expanded_portfolio.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("실험 13: 코인 확장 포트폴리오 (5코인 → 10코인)")
    print(f"  총 자본  : {TOTAL_CAPITAL:,} USDT")
    print(f"  기존 5코인: {' · '.join(COIN_LABELS[s] for s in COINS_5)}")
    print(f"  신규 5코인: ETH · BNB · DOGE · LINK · MATIC")
    print(f"  기간     : {START} ~ {END}")
    print(f"  전략     : 저엔트로피 롱 + Kelly + MA200 + 온체인 (Exp12 동일)")
    print("=" * 70)

    # 온체인 데이터
    print("\n[온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    # 10개 코인 데이터 수집
    print("\n[코인 데이터 수집 + MPE 계산 (캐시 적용)...]")
    all_coin_data = {}
    for sym in COINS_10:
        label = COIN_LABELS[sym]
        print(f"  [{label}] 수집 중...")
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168,
                          cache_key=f"{sym}_1h_{START}_{END}")
        h_onchain = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        all_coin_data[sym] = prepare_coin(sym, df, mpe, h_onchain)

    # A. 5코인 포트폴리오 (Exp12 재현)
    print("\n[5코인 포트폴리오 시뮬레이션 (Exp12 재현)...]")
    coin_data_5 = {sym: all_coin_data[sym] for sym in COINS_5}
    result_5 = run_portfolio(coin_data_5, TOTAL_CAPITAL)

    # B. 10코인 포트폴리오
    print("\n[10코인 포트폴리오 시뮬레이션...]")
    result_10 = run_portfolio(all_coin_data, TOTAL_CAPITAL)

    # 결과 출력
    print_report(result_5,  "A. 5코인 포트폴리오 (Exp12 재현)")
    print_report(result_10, "B. 10코인 확장 포트폴리오 (Exp13)")
    print_comparison(result_5, result_10)

    # 시각화
    plot_results(result_5, result_10, all_coin_data)

    # 결론 요약
    n5  = len(result_5["trades"].dropna(subset=["pnl_usdt"])) if len(result_5["trades"]) else 0
    n10 = len(result_10["trades"].dropna(subset=["pnl_usdt"])) if len(result_10["trades"]) else 0
    s5  = result_5["metrics"]["Sharpe"]
    s10 = result_10["metrics"]["Sharpe"]

    print(f"\n{'=' * 70}")
    print(f"  실험 13 결론")
    print(f"{'=' * 70}")
    print(f"  거래 빈도: {n5}건 → {n10}건 (+{n10-n5}건, +{(n10-n5)/n5*100:.0f}%)" if n5 else "")
    print(f"  연간 빈도: {n5/4:.1f}건/년 → {n10/4:.1f}건/년")
    print(f"  Sharpe:   {s5} → {s10}")
    if float(str(s10)) > float(str(s5)):
        print(f"  → 코인 확장이 성과 개선에 기여. 10코인 포트폴리오 권장.")
    elif float(str(s10)) < float(str(s5)):
        print(f"  → 신규 코인 중 해악 코인 있음. 개별 Sharpe 확인 후 선별 필요.")
    else:
        print(f"  → 성과 동일. 거래 빈도 증가에 의의.")


if __name__ == "__main__":
    main()
