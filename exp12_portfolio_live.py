"""
실전 운용 포트폴리오 시뮬레이터 (실험 12)
목적: 5코인 동시 운용 — 공유 자본 풀에서 신호 발생 코인에 동시 진입

설계:
  총 자본: TOTAL_CAPITAL (기본 10,000 USDT)
  코인별 배분: 균등 (총 자본 / 코인 수)
  동시 포지션: 제한 없음 (각 코인 독립, 5개 동시 가능)
  Kelly 사이징: 코인 배분 자본 기준 (50% / 30% / 15%)
  숏: 금지

핵심 출력:
  1. 전체 포트폴리오 누적 수익 곡선
  2. 시간별 동시 포지션 수 / 자본 배포율
  3. 코인별 기여도 분석
  4. 실전 운용 가이드 (연간 기대 거래수, 자본 효율, 리스크)

실행: py run_portfolio_live.py
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

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TOTAL_CAPITAL = 10_000   # USDT

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
COIN_LABELS = {
    "BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
    "ADAUSDT": "ADA", "DOTUSDT": "DOT",
}
COIN_COLORS = {
    "BTCUSDT": "#f0c040", "SOLUSDT": "#56d364", "AVAXUSDT": "#d2a8ff",
    "ADAUSDT": "#ffab70", "DOTUSDT": "#58a6ff",
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
# 포트폴리오 시뮬레이터
# ─────────────────────────────────────────────────────────────────────────────

def run_live_portfolio(coin_data: dict, total_capital: float = TOTAL_CAPITAL) -> dict:
    """
    5코인 동시 운용 시뮬레이터

    coin_data: {symbol: {"df": df, "mpe": mpe, "h_onchain": h_onchain}}

    각 코인은 균등 배분 자본(total_capital / n_coins)을 독립적으로 운용.
    신호 발생 시 동시 진입 가능 (코인 간 간섭 없음).
    총 포트폴리오 자본 = 각 코인 자본의 합.
    """
    n_coins      = len(coin_data)
    coin_capital = total_capital / n_coins   # 코인당 배분 자본

    # 코인별 전략 데이터 사전 계산
    prepared = {}
    common_idx = None
    for sym, data in coin_data.items():
        df    = data["df"]
        mpe   = data["mpe"]
        h_oc  = data.get("h_onchain")
        rsi   = compute_rsi(df["close"])
        ma200 = df["close"].rolling(MA_PERIOD).mean()
        sig   = generate_signals(rsi)
        thresh = np.percentile(mpe.dropna(), ENTROPY_PCT)
        oc_thresh = np.percentile(h_oc.dropna(), 40) if h_oc is not None else None

        prepared[sym] = {
            "df": df, "mpe": mpe, "h_oc": h_oc,
            "rsi": rsi, "ma200": ma200, "sig": sig,
            "thresh": thresh, "oc_thresh": oc_thresh,
        }
        if common_idx is None:
            common_idx = df.index

    # 코인별 상태 초기화
    coin_equity   = {sym: coin_capital for sym in coin_data}
    positions     = {}   # {sym: {"entry_price": , "entry_hour": , "pos_size_usdt": , "kelly_frac": }}
    all_trades    = []

    portfolio_curve   = []   # 시간별 총 자산
    deployed_curve    = []   # 시간별 배포된 자산
    concurrent_curve  = []   # 시간별 동시 포지션 수
    coin_equity_curve = {sym: [] for sym in coin_data}

    for i, idx in enumerate(common_idx):

        deployed_usdt   = 0.0
        n_concurrent    = len(positions)

        for sym, prep in prepared.items():
            df    = prep["df"]
            if idx not in df.index:
                coin_equity_curve[sym].append(coin_equity[sym])
                continue

            price     = df["close"].loc[idx]
            rsi_val   = prep["rsi"].loc[idx]   if idx in prep["rsi"].index   else 50.0
            ma_val    = prep["ma200"].loc[idx]  if idx in prep["ma200"].index else np.nan
            mpe_val   = prep["mpe"].loc[idx]    if idx in prep["mpe"].index   else np.nan
            sig_val   = prep["sig"].loc[idx]    if idx in prep["sig"].index   else 0
            oc_ok     = True
            if prep["h_oc"] is not None and prep["oc_thresh"] is not None:
                oc_val = prep["h_oc"].loc[idx] if idx in prep["h_oc"].index else 0
                oc_ok  = oc_val <= prep["oc_thresh"]

            # ── 청산 ─────────────────────────────────────────────────────
            if sym in positions:
                pos   = positions[sym]
                held  = i - pos["entry_hour"]
                ep    = pos["entry_price"]
                kf    = pos["kelly_frac"]
                if rsi_val > 50 or held >= MAX_HOLD_H:
                    pnl_pct = (price - ep) / ep
                    pnl_usdt = pnl_pct * pos["pos_size_usdt"]
                    fee      = pos["pos_size_usdt"] * FEE_RATE
                    coin_equity[sym] += pnl_usdt - fee
                    all_trades.append({
                        "symbol":     sym,
                        "entry_time": pos["entry_time"],
                        "exit_time":  idx,
                        "entry_price": ep,
                        "exit_price": price,
                        "pnl_pct":    pnl_pct,
                        "pnl_usdt":   pnl_usdt - fee,
                        "kelly_frac": kf,
                        "held_h":     held,
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
                            "entry_price": price * (1 + FEE_RATE),
                            "entry_time":  idx,
                            "entry_hour":  i,
                            "pos_size_usdt": pos_usdt,
                            "kelly_frac":  k_frac,
                        }

            # 미실현 평가
            eq = coin_equity[sym]
            if sym in positions:
                pos = positions[sym]
                unreal = (price - pos["entry_price"]) / pos["entry_price"] * pos["pos_size_usdt"]
                eq += unreal
                deployed_usdt += pos["pos_size_usdt"]

            coin_equity_curve[sym].append(eq)

        total_eq = sum(
            coin_equity[s] +
            (((prepared[s]["df"]["close"].loc[idx] - positions[s]["entry_price"])
              / positions[s]["entry_price"] * positions[s]["pos_size_usdt"])
             if s in positions and idx in prepared[s]["df"].index else 0)
            for s in coin_data
        )
        portfolio_curve.append(total_eq)
        deployed_curve.append(deployed_usdt)
        concurrent_curve.append(len(positions))

    trades_df  = pd.DataFrame(all_trades)
    port_equity = pd.Series(portfolio_curve, index=common_idx)
    deploy_ser  = pd.Series(deployed_curve,  index=common_idx)
    conc_ser    = pd.Series(concurrent_curve, index=common_idx)
    coin_eq_ser = {sym: pd.Series(coin_equity_curve[sym], index=common_idx[:len(coin_equity_curve[sym])])
                   for sym in coin_data}

    return {
        "portfolio_equity": port_equity,
        "deployed":         deploy_ser,
        "concurrent":       conc_ser,
        "trades":           trades_df,
        "coin_equity":      coin_eq_ser,
        "final_capital":    port_equity.iloc[-1],
        "coin_capital":     coin_capital,
    }


def print_report(result: dict):
    trades   = result["trades"]
    port_eq  = result["portfolio_equity"]
    metrics  = compute_metrics(port_eq / TOTAL_CAPITAL)
    coin_cap = result["coin_capital"]

    print("\n" + "=" * 70)
    print(f"  실전 운용 포트폴리오 리포트")
    print(f"  총 자본: {TOTAL_CAPITAL:,} USDT  |  코인당: {coin_cap:,.0f} USDT × {len(COINS)}개")
    print("=" * 70)

    print(f"\n[포트폴리오 성과]")
    print(f"  총 수익률  : {metrics['총 수익률']}")
    print(f"  Sharpe     : {metrics['Sharpe']}")
    print(f"  최대 낙폭  : {metrics['최대 낙폭']}")
    print(f"  최종 자산  : {result['final_capital']:,.0f} USDT  "
          f"(+{result['final_capital']-TOTAL_CAPITAL:,.0f} USDT)")

    print(f"\n[거래 통계]")
    if len(trades):
        closed  = trades.dropna(subset=["pnl_usdt"])
        n_total = len(closed)
        wr      = (closed["pnl_pct"] > 0).mean() * 100
        avg_pnl_pct  = closed["pnl_pct"].mean() * 100
        avg_pnl_usdt = closed["pnl_usdt"].mean()
        total_pnl    = closed["pnl_usdt"].sum()
        avg_held     = closed["held_h"].mean()
        annual_n     = n_total / 4   # 4년 데이터

        print(f"  총 거래 수   : {n_total}건 (4년)")
        print(f"  연간 거래 수 : {annual_n:.1f}건/년")
        print(f"  승률         : {wr:.1f}%")
        print(f"  평균 수익    : {avg_pnl_pct:+.3f}% / {avg_pnl_usdt:+.2f} USDT/건")
        print(f"  총 수익      : {total_pnl:+,.2f} USDT")
        print(f"  평균 보유    : {avg_held:.1f}H")

        # Kelly 분포
        k_dist = closed["kelly_frac"].value_counts().sort_index(ascending=False)
        print(f"\n[Kelly 사이징 분포]")
        for kf, cnt in k_dist.items():
            usdt = coin_cap * kf
            print(f"  {int(kf*100):>3}% ({usdt:>6,.0f} USDT/건) : {cnt:>4}건  "
                  f"({cnt/n_total*100:.1f}%)")

        # 코인별 기여도
        print(f"\n[코인별 기여도]")
        print(f"  {'코인':<6} {'거래':>6} {'승률':>8} {'평균PnL(USDT)':>14} {'총PnL(USDT)':>14} "
              f"{'평균Kelly':>10}")
        print(f"  {'-'*65}")
        for sym in COINS:
            sub = closed[closed["symbol"] == sym]
            if not len(sub):
                print(f"  {COIN_LABELS[sym]:<6} {'0':>6}")
                continue
            wr_c  = (sub["pnl_pct"] > 0).mean() * 100
            avg_c = sub["pnl_usdt"].mean()
            tot_c = sub["pnl_usdt"].sum()
            kf_c  = sub["kelly_frac"].mean() * 100
            print(f"  {COIN_LABELS[sym]:<6} {len(sub):>6}건 {wr_c:>7.1f}% "
                  f"{avg_c:>+14.2f} {tot_c:>+14.2f} {kf_c:>9.1f}%")

        # 동시 포지션 통계
        conc = result["concurrent"]
        print(f"\n[동시 포지션 통계]")
        print(f"  0개 동시: {(conc==0).mean()*100:.1f}% (대기)")
        for k in range(1, 6):
            pct = (conc==k).mean()*100
            if pct > 0.1:
                print(f"  {k}개 동시: {pct:.1f}%")
        print(f"  최대 동시 포지션: {conc.max()}개")

        # 자본 배포율
        dep_pct = result["deployed"] / TOTAL_CAPITAL * 100
        print(f"\n[자본 배포율]")
        print(f"  평균 배포율: {dep_pct.mean():.1f}%  (나머지는 현금 대기)")
        print(f"  최대 배포율: {dep_pct.max():.1f}%")

    # 실전 운용 가이드
    print(f"\n{'=' * 70}")
    print(f"  실전 운용 가이드")
    print(f"{'=' * 70}")
    print(f"  권장 최소 자본   : 5,000 USDT (코인당 1,000 USDT, Kelly 15%→150 USDT)")
    print(f"  권장 운용 자본   : 10,000 USDT (코인당 2,000 USDT, Kelly 50%→1,000 USDT)")
    print(f"  코인 배분        : 균등 (코인 추가/제거 시 균등 재조정)")
    print(f"  신호 점검 주기   : 1시간 (1h 봉 기준)")
    print(f"  최대 동시 포지션 : {int(result['concurrent'].max())}개 (실제 발생 기준)")
    print(f"  거래당 최대 리스크: Kelly 50% × 코인 배분 = {coin_cap*0.5:,.0f} USDT/건")
    print(f"  연간 기대 거래   : ~{annual_n:.0f}건")


def plot_portfolio(result: dict):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    port_eq  = result["portfolio_equity"]
    trades   = result["trades"]
    conc     = result["concurrent"]
    deploy   = result["deployed"] / TOTAL_CAPITAL * 100
    coin_eq  = result["coin_equity"]
    metrics  = compute_metrics(port_eq / TOTAL_CAPITAL)

    fig = plt.figure(figsize=(22, 24), facecolor="#0d1117")
    fig.suptitle(
        f"실전 운용 포트폴리오 시뮬레이터  |  총 자본 {TOTAL_CAPITAL:,} USDT\n"
        f"BTC · SOL · AVAX · ADA · DOT  |  Sharpe {metrics['Sharpe']}  "
        f"수익률 {metrics['총 수익률']}  최대낙폭 {metrics['최대 낙폭']}",
        color="#e6edf3", fontsize=13, y=0.99
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2.5, 1.2, 1.2, 1.2])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--")

    def _xfmt(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    # ── 1. 포트폴리오 누적 자산 곡선 ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    ax1.plot(port_eq.index, port_eq.values,
             color="#56d364", linewidth=2.2, alpha=0.95, label="포트폴리오 총 자산 (USDT)")
    ax1.axhline(TOTAL_CAPITAL, color="#8b949e", linewidth=0.8, linestyle=":", label="원금")
    # 코인별 자산 (얇게)
    for sym, eq in coin_eq.items():
        ax1.plot(eq.index, eq.values,
                 color=COIN_COLORS[sym], linewidth=0.8, alpha=0.5,
                 label=f"{COIN_LABELS[sym]} ({result['coin_capital']:,.0f}→{eq.iloc[-1]:,.0f})")
    ax1.set_ylabel("자산 (USDT)", color="#8b949e")
    ax1.set_title("전체 포트폴리오 누적 자산 (초록=합산 / 얇은선=코인별)", color="#e6edf3")
    ax1.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left", ncol=3)
    _xfmt(ax1)

    # ── 2. 자본 배포율 ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)
    ax2.fill_between(deploy.index, deploy.values, 0,
                     color="#79c0ff", alpha=0.5, label="배포 자본 (%)")
    ax2.axhline(deploy.mean(), color="#f7931a", linewidth=1.2, linestyle="--",
                label=f"평균 {deploy.mean():.1f}%")
    ax2.set_ylabel("자본 배포율 (%)", color="#8b949e")
    ax2.set_ylim(0, min(100, deploy.max() * 1.3))
    ax2.set_title("시간별 자본 배포율 (낮을수록 현금 대기 중)", color="#e6edf3")
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    _xfmt(ax2)

    # ── 3. 코인별 Sharpe 및 기여도 ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    if len(trades):
        closed = trades.dropna(subset=["pnl_usdt"])
        syms   = list(COINS)
        totals = [closed[closed["symbol"]==s]["pnl_usdt"].sum() for s in syms]
        colors_bar = ["#56d364" if t >= 0 else "#ff7b72" for t in totals]
        bars = ax3.bar([COIN_LABELS[s] for s in syms], totals,
                       color=colors_bar, alpha=0.85)
        for bar, val in zip(bars, totals):
            ax3.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height() + (10 if val >= 0 else -40),
                     f"{val:+,.0f}", ha="center", color="#e6edf3", fontsize=9)
        ax3.axhline(0, color="#8b949e", linewidth=0.8)
        ax3.set_title("코인별 총 PnL 기여도 (USDT)", color="#e6edf3")
        ax3.set_ylabel("PnL (USDT)", color="#8b949e")

    # ── 4. Kelly 사이징 분포 ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    if len(trades):
        closed = trades.dropna(subset=["pnl_usdt"])
        k_dist = closed["kelly_frac"].value_counts().sort_index(ascending=False)
        kelly_labels = {0.50: "50% (pct1)", 0.30: "30% (pct5)", 0.15: "15% (pct10)"}
        kelly_colors = {0.50: "#f0c040", 0.30: "#56d364", 0.15: "#79c0ff"}
        ax4.bar([kelly_labels.get(k, f"{int(k*100)}%") for k in k_dist.index],
                k_dist.values,
                color=[kelly_colors.get(k, "#8b949e") for k in k_dist.index],
                alpha=0.85)
        for i, (k, cnt) in enumerate(k_dist.items()):
            usdt = result["coin_capital"] * k
            ax4.text(i, cnt + 0.3, f"{cnt}건\n({usdt:,.0f}U)", ha="center",
                     color="#e6edf3", fontsize=8.5)
        ax4.set_title("Kelly 사이징 분포 (신호 강도별)", color="#e6edf3")
        ax4.set_ylabel("거래 수", color="#8b949e")

    # ── 5. 월별 수익 히스토그램 ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    _style(ax5)
    monthly = port_eq.resample("ME").last().pct_change().dropna() * 100
    colors_m = ["#56d364" if v >= 0 else "#ff7b72" for v in monthly.values]
    ax5.bar(range(len(monthly)), monthly.values, color=colors_m, alpha=0.85)
    ax5.axhline(0, color="#8b949e", linewidth=0.8)
    ax5.axhline(monthly.mean(), color="#f7931a", linewidth=1.2, linestyle="--",
                label=f"월평균 {monthly.mean():+.2f}%")
    step = max(1, len(monthly) // 10)
    ax5.set_xticks(range(0, len(monthly), step))
    ax5.set_xticklabels(monthly.index[::step].strftime("%y-%m"),
                        rotation=45, ha="right", color="#8b949e", fontsize=7)
    ax5.set_title("월별 포트폴리오 수익률 (%)", color="#e6edf3")
    ax5.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 6. 요약 테이블 ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    _style(ax6); ax6.axis("off")
    closed = trades.dropna(subset=["pnl_usdt"]) if len(trades) else pd.DataFrame()
    annual_n = len(closed) / 4 if len(closed) else 0
    rows = [
        ["총 자본",         f"{TOTAL_CAPITAL:,} USDT"],
        ["최종 자산",       f"{result['final_capital']:,.0f} USDT"],
        ["총 수익",         f"{result['final_capital']-TOTAL_CAPITAL:+,.0f} USDT"],
        ["Sharpe",          metrics["Sharpe"]],
        ["최대 낙폭",       metrics["최대 낙폭"]],
        ["총 거래",         f"{len(closed)}건 (4년)"],
        ["연간 거래",       f"~{annual_n:.0f}건/년"],
        ["승률",            f"{(closed['pnl_pct']>0).mean()*100:.1f}%" if len(closed) else "—"],
        ["평균 배포율",     f"{deploy.mean():.1f}%"],
        ["최대 동시 포지션",f"{int(conc.max())}개"],
    ]
    tbl = ax6.table(cellText=rows, colLabels=["지표", "값"],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if r == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax6.set_title("성과 요약", color="#e6edf3", fontsize=10, pad=8)

    plt.savefig(RESULTS_DIR / "portfolio_live.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("\n저장: results/portfolio_live.png")


def main():
    print("=" * 70)
    print("실험 12: 실전 운용 포트폴리오 시뮬레이터")
    print(f"  총 자본  : {TOTAL_CAPITAL:,} USDT")
    print(f"  코인     : {' · '.join(COIN_LABELS[s] for s in COINS)}")
    print(f"  기간     : {START} ~ {END}")
    print(f"  전략     : 저엔트로피 롱 + Kelly + MA200 + 온체인")
    print(f"  동시진입 : 제한 없음 (최대 5코인 동시)")
    print("=" * 70)

    print("\n[온체인 데이터 수집...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 수집 + MPE 계산...]")
    coin_data = {}
    for sym in COINS:
        label = COIN_LABELS[sym]
        print(f"  [{label}] 수집 중...")
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168)
        h_onchain = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        coin_data[sym] = {"df": df, "mpe": mpe, "h_onchain": h_onchain}

    print("\n[포트폴리오 시뮬레이션 실행...]")
    result = run_live_portfolio(coin_data, TOTAL_CAPITAL)

    print_report(result)
    plot_portfolio(result)


if __name__ == "__main__":
    main()
