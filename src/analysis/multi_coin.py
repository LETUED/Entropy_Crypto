"""
멀티 코인 분석 + 포트폴리오 전략

전략:
- 각 코인별로 Phase 2 전략 독립 실행
- 포트폴리오: 엔트로피가 낮은 코인에만 자본 배분 (동시 최대 2개)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path


def _setup_font():
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()

from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

COIN_COLORS = {
    "BTCUSDT": "#f7931a",
    "ETHUSDT": "#627eea",
    "BNBUSDT": "#f3ba2f",
    "SOLUSDT": "#9945ff",
    "PORTFOLIO": "#3fb950",
}

COIN_LABELS = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "BNBUSDT": "BNB",
    "SOLUSDT": "SOL",
}


# ── 단일 코인 Phase 2 전략 ────────────────────────────────────────────────────
def run_single_coin(
    df: pd.DataFrame,
    mpe: pd.Series,
    h_onchain: pd.Series = None,
    capital: float = 1.0,
) -> pd.Series:
    """
    켈리 + 추세 + 엔트로피 + 온체인 전략
    capital: 투입 자본 비율 (포트폴리오에서 분할 시 사용)
    """
    from src.analysis.h4_backtest import kelly_size

    rsi    = compute_rsi(df["close"])
    signal = generate_signals(rsi)
    ma200  = df["close"].rolling(MA_PERIOD).mean()

    threshold = np.percentile(mpe.dropna(), ENTROPY_PCT)

    onchain_ok = pd.Series(True, index=df.index)
    if h_onchain is not None:
        oc_thresh = np.percentile(h_onchain.dropna(), 40)
        onchain_ok = h_onchain <= oc_thresh

    equity     = capital
    position   = 0
    entry_price = 0.0
    entry_hour  = -999
    pos_size    = 0.3
    curve       = []

    for i, idx in enumerate(df.index):
        price   = df["close"].loc[idx]
        sig     = signal.loc[idx]   if idx in signal.index   else 0
        mpe_val = mpe.loc[idx]      if idx in mpe.index      else np.nan
        ma_val  = ma200.loc[idx]    if idx in ma200.index     else np.nan
        is_low  = (not np.isnan(mpe_val)) and (mpe_val <= threshold)
        oc_ok   = onchain_ok.loc[idx] if idx in onchain_ok.index else True

        # 청산: RSI 회복(>50) 또는 최대 168H
        if position == 1:
            rsi_val = rsi.loc[idx] if idx in rsi.index else 50.0
            if rsi_val > 50 or (i - entry_hour) >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * pos_size) * (1 - FEE_RATE)
                position = 0

        # 진입
        if position == 0:
            mpe_val_safe = mpe_val if not np.isnan(mpe_val) else 1.0
            pos_size = kelly_size(mpe_val_safe, mpe)

            long_ok = (sig == 1 and is_low
                       and not np.isnan(ma_val) and price > ma_val
                       and pos_size > 0 and oc_ok)

            if long_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i

        curve.append(equity)

    return pd.Series(curve, index=df.index, name="equity")


# ── 포트폴리오 전략 ───────────────────────────────────────────────────────────
def run_portfolio(
    coin_data: dict,   # {symbol: {"df": df, "mpe": mpe, "h_onchain": h_onchain}}
    max_slots: int = 2,   # 동시 최대 진입 코인 수
) -> pd.Series:
    """
    매 시간봉마다 엔트로피가 가장 낮은 max_slots개 코인에 균등 배분
    """
    from src.analysis.h4_backtest import kelly_size

    # 공통 인덱스 + 코인별 RSI 사전 계산
    common_idx = coin_data[list(coin_data.keys())[0]]["df"].index
    rsi_map    = {sym: compute_rsi(data["df"]["close"]) for sym, data in coin_data.items()}

    equity    = 1.0
    positions = {}   # {symbol: (entry_price, entry_hour, pos_size)}
    curve     = []

    for i, idx in enumerate(common_idx):

        # ── 기존 포지션 청산: RSI 회복(>50) 또는 최대 168H ──────────────
        to_close = []
        for sym, (ep, eh, ps) in positions.items():
            if idx not in coin_data[sym]["df"].index:
                continue
            price   = coin_data[sym]["df"]["close"].loc[idx]
            rsi_val = rsi_map[sym].loc[idx] if idx in rsi_map[sym].index else 50.0
            if rsi_val > 50 or (i - eh) >= MAX_HOLD_H:
                pnl = (price - ep) / ep
                equity *= (1 + pnl * ps / max_slots) * (1 - FEE_RATE)
                to_close.append(sym)
        for sym in to_close:
            del positions[sym]

        # ── 새 진입 후보 탐색 ─────────────────────────────────────────────
        if len(positions) < max_slots:
            candidates = []

            for sym, data in coin_data.items():
                if sym in positions:
                    continue
                df      = data["df"]
                mpe     = data["mpe"]
                h_oc    = data.get("h_onchain")

                if idx not in df.index:
                    continue

                price   = df["close"].loc[idx]
                mpe_val = mpe.loc[idx] if idx in mpe.index else np.nan
                if np.isnan(mpe_val):
                    continue

                threshold = np.percentile(mpe.dropna(), ENTROPY_PCT)
                if mpe_val > threshold:
                    continue

                ma200 = df["close"].rolling(MA_PERIOD).mean()
                ma_val = ma200.loc[idx] if idx in ma200.index else np.nan
                if np.isnan(ma_val) or price < ma_val:
                    continue

                rsi    = compute_rsi(df["close"])
                signal = generate_signals(rsi)
                sig    = signal.loc[idx] if idx in signal.index else 0
                if sig != 1:
                    continue

                if h_oc is not None:
                    oc_thresh = np.percentile(h_oc.dropna(), 40)
                    if h_oc.loc[idx] > oc_thresh if idx in h_oc.index else False:
                        continue

                ps = kelly_size(mpe_val, mpe)
                if ps == 0:
                    continue

                candidates.append((mpe_val, sym, price, ps))

            # 엔트로피 낮은 순으로 정렬 → 남은 슬롯만큼 진입
            candidates.sort(key=lambda x: x[0])
            for _, sym, price, ps in candidates[:max_slots - len(positions)]:
                positions[sym] = (price * (1 + FEE_RATE), i, ps)

        curve.append(equity)

    return pd.Series(curve, index=common_idx, name="portfolio")


# ── 메인 실행 ────────────────────────────────────────────────────────────────
def run_multi_coin(coin_data: dict) -> dict:
    print("=" * 60)
    print("멀티 코인 분석")
    print(f"코인: {list(coin_data.keys())}")
    print("=" * 60)

    results = {}

    # 개별 코인 전략
    for sym, data in coin_data.items():
        label = COIN_LABELS.get(sym, sym)
        print(f"  {label} 실행 중...")
        eq = run_single_coin(data["df"], data["mpe"], data.get("h_onchain"))
        results[sym] = {
            "label":   label,
            "equity":  eq,
            "metrics": compute_metrics(eq),
        }

    # 포트폴리오 전략
    print("  포트폴리오 (동적 배분) 실행 중...")
    port_eq = run_portfolio(coin_data, max_slots=2)
    results["PORTFOLIO"] = {
        "label":   "포트폴리오 (동적 배분)",
        "equity":  port_eq,
        "metrics": compute_metrics(port_eq),
    }

    _print_results(results)
    return results


def _print_results(results: dict):
    print("\n" + "=" * 75)
    print(f"{'전략':<25} {'수익률':>10} {'Sharpe':>8} {'최대낙폭':>10} {'최종자산':>10}")
    print("-" * 75)
    for key, r in results.items():
        m = r["metrics"]
        print(f"{r['label']:<25} {m['총 수익률']:>10} {m['Sharpe']:>8} "
              f"{m['최대 낙폭']:>10} {m['최종 자산']:>10}")
    print("=" * 75)


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_multi_coin(results: dict, bah_data: dict = None, save: bool = True):
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 16), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                            height_ratios=[2.5, 1.2, 1])

    # ── 1. 누적 수익 곡선 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    # 바이앤홀드 (회색 점선)
    if bah_data:
        for sym, df in bah_data.items():
            bah = df["close"] / df["close"].iloc[0]
            ax1.plot(bah.index, bah.values,
                     color=COIN_COLORS.get(sym, "#8b949e"),
                     linewidth=0.8, alpha=0.25, linestyle="--")

    # 전략 곡선
    for key, r in results.items():
        lw    = 2.5 if key == "PORTFOLIO" else 1.5
        alpha = 1.0 if key == "PORTFOLIO" else 0.8
        ax1.plot(r["equity"].index, r["equity"].values,
                 color=COIN_COLORS.get(key, "#8b949e"),
                 linewidth=lw, alpha=alpha,
                 label=f"{r['label']}  ({r['metrics']['총 수익률']}  Sharpe {r['metrics']['Sharpe']})")

    ax1.axhline(1.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("누적 수익 배율", color="#8b949e")
    ax1.set_title("멀티 코인 전략 비교  (점선 = 해당 코인 바이앤홀드)",
                  color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")
    _xfmt(ax1)

    # ── 2. 낙폭 비교 ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2)

    for key, r in results.items():
        eq = r["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        lw = 2.0 if key == "PORTFOLIO" else 1.0
        ax2.fill_between(dd.index, dd, 0,
                         color=COIN_COLORS.get(key, "#8b949e"),
                         alpha=0.35 if key != "PORTFOLIO" else 0.6,
                         label=f"{r['label']} (최대 {r['metrics']['최대 낙폭']})")

    ax2.set_ylabel("낙폭 (%)", color="#8b949e")
    ax2.set_title("최대 낙폭 비교  (포트폴리오가 낮을수록 우수)",
                  color="#e6edf3", fontsize=11)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=3)
    _xfmt(ax2)

    # ── 3. 코인별 Sharpe 막대 ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)

    keys   = list(results.keys())
    sharpe = [float(results[k]["metrics"]["Sharpe"]) for k in keys]
    colors = [COIN_COLORS.get(k, "#8b949e") for k in keys]
    labels = [results[k]["label"] for k in keys]

    bars = ax3.bar(range(len(keys)), sharpe, color=colors, alpha=0.85)
    for bar, v in zip(bars, sharpe):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=9, fontweight="bold")

    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(range(len(keys)))
    ax3.set_xticklabels(labels, color="#e6edf3", fontsize=8, rotation=15, ha="right")
    ax3.set_ylabel("Sharpe Ratio", color="#8b949e")
    ax3.set_title("코인별 Sharpe 비교", color="#e6edf3", fontsize=10)
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. 성과 요약 테이블 ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    ax4.axis("off")

    rows = []
    for key, r in results.items():
        m = r["metrics"]
        rows.append([r["label"], m["총 수익률"], m["Sharpe"], m["최대 낙폭"]])

    tbl = ax4.table(
        cellText=rows,
        colLabels=["전략", "수익률", "Sharpe", "최대낙폭"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
        if row > 0 and col == 0 and rows[row - 1][0] == "포트폴리오 (동적 배분)":
            cell.set_facecolor("#1a2d1a")
    ax4.set_title("성과 요약", color="#e6edf3", fontsize=10, pad=8)

    fig.suptitle("멀티 코인 엔트로피 전략  —  포트폴리오 동적 배분",
                 color="#e6edf3", fontsize=14, fontweight="bold", y=1.01)

    if save:
        path = RESULTS_DIR / "multi_coin.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def _style(ax):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.yaxis.grid(True, color="#21262d", linestyle="--")
    ax.xaxis.grid(True, color="#21262d", linestyle=":", linewidth=0.4)


def _xfmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")
