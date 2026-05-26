"""
실험 14: 카테고리별 전략 유효성 + OOS 검증
목적: 어떤 자산 유형에서 엔트로피 전략이 작동하는가?
     전략이 2025년 이후에도 유효한가?

설계:
  코인: 33개 / 8개 카테고리
  IS  (개발 구간): 2021-01-01 ~ 2025-01-01
  OOS (진짜 미래): 2025-01-01 ~ 2026-05-26
  OOS 임계값: IS에서 계산 (진짜 OOS 조건)
  전략: 저엔트로피 롱 + Kelly + MA200 + 온체인

핵심 출력:
  1. 카테고리 x IS/OOS Sharpe 히트맵
  2. 코인별 IS vs OOS 산점도
  3. 카테고리별 승률/거래수 비교
  4. OOS에서 살아남은 코인/카테고리 목록

실행: py exp14_category_oos.py
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
import matplotlib.colors as mcolors

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import (
    kelly_size, compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H
)

RESULTS_DIR = Path("results")
START       = "2021-01-01"
IS_END      = "2025-01-01"   # IS/OOS 분리 기준
OOS_END     = "2026-05-26"

CATEGORIES = {
    "가치저장":        ["BTCUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT"],
    "스마트컨트랙트L1": ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
                        "ATOMUSDT", "NEARUSDT", "FTMUSDT", "MATICUSDT"],
    "DeFi":           ["LINKUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "1INCHUSDT"],
    "거래소토큰":      ["BNBUSDT", "CROUSDT"],
    "밈":             ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"],
    "게임메타버스":    ["AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT"],
    "인프라":         ["FILUSDT", "GRTUSDT", "RNDRUSDT"],
    "결제크로스체인":  ["XRPUSDT", "XLMUSDT", "ALGOUSDT"],
}

# 심볼 -> 카테고리 역매핑
SYM_TO_CAT = {sym: cat for cat, syms in CATEGORIES.items() for sym in syms}

LABEL = {
    "BTCUSDT":"BTC", "LTCUSDT":"LTC", "BCHUSDT":"BCH", "ETCUSDT":"ETC",
    "ETHUSDT":"ETH", "SOLUSDT":"SOL", "AVAXUSDT":"AVAX","ADAUSDT":"ADA",
    "DOTUSDT":"DOT", "ATOMUSDT":"ATOM","NEARUSDT":"NEAR","FTMUSDT":"FTM",
    "MATICUSDT":"MATIC","LINKUSDT":"LINK","UNIUSDT":"UNI","AAVEUSDT":"AAVE",
    "SUSHIUSDT":"SUSHI","1INCHUSDT":"1INCH","BNBUSDT":"BNB","CROUSDT":"CRO",
    "DOGEUSDT":"DOGE","SHIBUSDT":"SHIB","PEPEUSDT":"PEPE",
    "AXSUSDT":"AXS","SANDUSDT":"SAND","MANAUSDT":"MANA","GALAUSDT":"GALA",
    "FILUSDT":"FIL","GRTUSDT":"GRT","RNDRUSDT":"RNDR",
    "XRPUSDT":"XRP","XLMUSDT":"XLM","ALGOUSDT":"ALGO",
}


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────────────────
# 단일 코인 전략 실행 (특정 기간, IS 임계값 사용 가능)
# ─────────────────────────────────────────────────────────────────────────────

def run_coin_period(df, mpe, h_onchain, start, end,
                    mpe_thresh=None, oc_thresh=None):
    """
    df/mpe/h_onchain 전체 데이터에서 start~end 구간만 추출해 전략 실행.
    mpe_thresh/oc_thresh 미지정 시 해당 구간 데이터로 계산 (IS용).
    지정 시 그 값 사용 (OOS용 — IS 임계값 적용).
    """
    df_p  = df[start:end]
    mpe_p = mpe[start:end]
    h_p   = h_onchain[start:end] if h_onchain is not None else None

    if len(df_p) < 200:
        return None   # 데이터 부족

    rsi   = compute_rsi(df_p["close"])
    ma200 = df_p["close"].rolling(MA_PERIOD).mean()
    sig   = generate_signals(rsi)

    # 임계값 계산 (IS용) 또는 적용 (OOS용)
    if mpe_thresh is None:
        valid = mpe_p.dropna()
        if len(valid) < 50:
            return None
        mpe_thresh = np.percentile(valid, ENTROPY_PCT)

    if oc_thresh is None and h_p is not None:
        valid_oc = h_p.dropna()
        oc_thresh = np.percentile(valid_oc, 40) if len(valid_oc) > 10 else None

    # Kelly 임계값 사전 계산 (O(n) 최적화 — 루프 내부 반복 계산 제거)
    mpe_clean = mpe_p.dropna()
    k_pct1  = np.percentile(mpe_clean, 1)
    k_pct5  = np.percentile(mpe_clean, 5)
    k_pct10 = np.percentile(mpe_clean, 10)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity   = 1.0
    position = 0
    entry_price = 0.0
    entry_hour  = 0
    entry_kfrac = 0.0
    trades  = []
    equity_curve = []

    idx_list = df_p.index.tolist()

    for i, idx in enumerate(idx_list):
        price   = df_p["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        mpe_val = mpe_p.loc[idx] if idx in mpe_p.index else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        oc_ok = True
        if h_p is not None and oc_thresh is not None and idx in h_p.index:
            oc_ok = h_p.loc[idx] <= oc_thresh

        # 청산
        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl, "held_h": held, "kfrac": entry_kfrac})
                position = 0

        # 진입
        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            is_low   = mpe_val <= mpe_thresh
            above_ma = price > ma_val
            k_frac   = _kelly(mpe_val)
            if sig_val == 1 and is_low and above_ma and k_frac > 0 and oc_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)  # 수수료는 슬리피지로만
                entry_hour  = i
                entry_kfrac = k_frac

        equity_curve.append(equity)

    # 미청산 포지션 강제 청산
    if position == 1 and len(idx_list) > 0:
        last_price = df_p["close"].iloc[-1]
        pnl = (last_price - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    eq_series = pd.Series(equity_curve, index=idx_list)
    metrics   = compute_metrics(eq_series)

    try:
        sharpe = float(metrics["Sharpe"])
    except Exception:
        sharpe = 0.0

    n_trades = len(trades)
    wr = np.mean([t["pnl_pct"] > 0 for t in trades]) if trades else 0.0
    avg_pnl  = np.mean([t["pnl_pct"] for t in trades]) * 100 if trades else 0.0
    n_months = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days / 30)

    return {
        "sharpe":    sharpe,
        "n_trades":  n_trades,
        "wr":        wr * 100,
        "avg_pnl":   avg_pnl,
        "annual_n":  n_trades / (n_months / 12),
        "mpe_thresh": mpe_thresh,
        "oc_thresh":  oc_thresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 전체 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_all(coin_data: dict) -> pd.DataFrame:
    rows = []
    for sym, data in coin_data.items():
        cat = SYM_TO_CAT.get(sym, "기타")
        lbl = LABEL.get(sym, sym)
        df  = data["df"]
        mpe = data["mpe"]
        h   = data["h_onchain"]

        print(f"  [{lbl:6s}] ", end="", flush=True)

        # IS 실행 (임계값도 IS에서 계산)
        is_res = run_coin_period(df, mpe, h, START, IS_END)
        if is_res is None:
            print("IS 데이터 부족 — 스킵")
            continue

        # OOS 실행 (IS 임계값 적용 — 진짜 OOS)
        oos_res = run_coin_period(df, mpe, h, IS_END, OOS_END,
                                  mpe_thresh=is_res["mpe_thresh"],
                                  oc_thresh=is_res["oc_thresh"])

        print(f"IS Sharpe {is_res['sharpe']:+.3f} ({is_res['n_trades']}건)  "
              f"OOS Sharpe {oos_res['sharpe']:+.3f} ({oos_res['n_trades']}건)"
              if oos_res else f"IS Sharpe {is_res['sharpe']:+.3f}  OOS 데이터 없음")

        rows.append({
            "sym":       sym,
            "label":     lbl,
            "category":  cat,
            "is_sharpe": is_res["sharpe"],
            "is_trades": is_res["n_trades"],
            "is_wr":     is_res["wr"],
            "is_pnl":    is_res["avg_pnl"],
            "oos_sharpe": oos_res["sharpe"]  if oos_res else np.nan,
            "oos_trades": oos_res["n_trades"] if oos_res else 0,
            "oos_wr":     oos_res["wr"]       if oos_res else np.nan,
            "oos_pnl":    oos_res["avg_pnl"]  if oos_res else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  실험 14: 카테고리별 IS vs OOS Sharpe")
    print("  IS: 2021~2025  |  OOS: 2025~2026")
    print("=" * 80)

    cat_order = list(CATEGORIES.keys())

    print(f"\n{'카테고리':<14} {'코인수':>4} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
          f"{'IS 거래/년':>10} {'OOS 거래/년':>11} {'IS->OOS':>8}")
    print("-" * 75)

    for cat in cat_order:
        sub = df[df["category"] == cat]
        if not len(sub):
            continue
        is_avg   = sub["is_sharpe"].mean()
        oos_avg  = sub["oos_sharpe"].mean()
        is_n     = sub["is_trades"].sum() / 4
        oos_n    = sub["oos_trades"].sum() / (17/12)
        direction = "UP" if oos_avg > is_avg else "DOWN"
        flag = " [+]" if oos_avg > 0 else " [-]"
        print(f"  {cat:<12} {len(sub):>4}개 {is_avg:>+10.3f} {oos_avg:>+11.3f}"
              f" {is_n:>10.1f} {oos_n:>11.1f} {direction:>8}{flag}")

    print("\n[코인별 상세]")
    print(f"  {'코인':<6} {'카테고리':<14} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
          f"{'IS 거래':>8} {'OOS 거래':>8} {'IS WR':>7} {'OOS WR':>7}")
    print("  " + "-" * 75)
    for cat in cat_order:
        sub = df[df["category"] == cat].sort_values("is_sharpe", ascending=False)
        for _, row in sub.iterrows():
            oos_s = f"{row['oos_sharpe']:+.3f}" if not np.isnan(row["oos_sharpe"]) else "  N/A"
            oos_w = f"{row['oos_wr']:.0f}%" if not np.isnan(row["oos_wr"]) else " N/A"
            survival = " [OOS+]" if row["oos_sharpe"] > 0 else ""
            print(f"  {row['label']:<6} {row['category']:<14} "
                  f"{row['is_sharpe']:>+10.3f} {oos_s:>11} "
                  f"{row['is_trades']:>8} {row['oos_trades']:>8} "
                  f"{row['is_wr']:>6.0f}% {oos_w:>7}{survival}")


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    cat_order  = list(CATEGORIES.keys())
    cat_colors = {
        "가치저장":        "#f0c040",
        "스마트컨트랙트L1": "#56d364",
        "DeFi":           "#79c0ff",
        "거래소토큰":      "#ff7b72",
        "밈":             "#f78166",
        "게임메타버스":    "#d2a8ff",
        "인프라":         "#58a6ff",
        "결제크로스체인":  "#ffab70",
    }

    fig = plt.figure(figsize=(26, 30), facecolor="#0d1117")
    fig.suptitle(
        "실험 14: 카테고리별 엔트로피 전략 유효성  |  IS(2021~2025) vs OOS(2025~2026)\n"
        "33개 코인 / 8개 카테고리",
        color="#e6edf3", fontsize=14, y=0.99
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2, 2, 2])

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", alpha=0.5)

    # ── 1. IS vs OOS Sharpe 산점도 (코인별) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    ax1.axhline(0, color="#8b949e", linewidth=1.0, linestyle="--", alpha=0.6)
    ax1.axvline(0, color="#8b949e", linewidth=1.0, linestyle="--", alpha=0.6)
    # 대각선 (IS=OOS 라인)
    lim = max(df["is_sharpe"].abs().max(), df["oos_sharpe"].abs().max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], color="#30363d", linewidth=1.0, linestyle=":")

    for cat in cat_order:
        sub = df[df["category"] == cat].dropna(subset=["oos_sharpe"])
        if not len(sub): continue
        c = cat_colors.get(cat, "#8b949e")
        ax1.scatter(sub["is_sharpe"], sub["oos_sharpe"],
                    color=c, s=120, alpha=0.9, label=cat, zorder=5)
        for _, row in sub.iterrows():
            ax1.annotate(row["label"],
                         (row["is_sharpe"], row["oos_sharpe"]),
                         textcoords="offset points", xytext=(5, 5),
                         color=c, fontsize=7.5, alpha=0.9)

    ax1.set_xlabel("IS Sharpe (2021~2025)", color="#8b949e")
    ax1.set_ylabel("OOS Sharpe (2025~2026)", color="#8b949e")
    ax1.set_title("IS vs OOS Sharpe 산점도 (우상단 = IS/OOS 모두 유효)", color="#e6edf3")
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=4, loc="upper left")
    # 사분면 레이블
    ax1.text(lim*0.7,  lim*0.85,  "IS+/OOS+ (타임리스)", color="#56d364", fontsize=9, alpha=0.7)
    ax1.text(-lim*0.9, lim*0.85,  "IS-/OOS+ (회복)", color="#79c0ff", fontsize=9, alpha=0.7)
    ax1.text(lim*0.7, -lim*0.9,   "IS+/OOS- (과적합)", color="#ff7b72", fontsize=9, alpha=0.7)
    ax1.text(-lim*0.9, -lim*0.9,  "IS-/OOS- (비적합)", color="#8b949e", fontsize=9, alpha=0.7)

    # ── 2. 카테고리별 IS Sharpe (막대) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    cat_is  = [df[df["category"]==c]["is_sharpe"].mean()  for c in cat_order]
    cat_oos = [df[df["category"]==c]["oos_sharpe"].mean() for c in cat_order]
    x = np.arange(len(cat_order))
    w = 0.35
    bars_is  = ax2.bar(x - w/2, cat_is,  w, label="IS  (2021~2025)",
                       color=[cat_colors.get(c,"#8b949e") for c in cat_order], alpha=0.85)
    bars_oos = ax2.bar(x + w/2, cat_oos, w, label="OOS (2025~2026)",
                       color=[cat_colors.get(c,"#8b949e") for c in cat_order],
                       alpha=0.45, hatch="//")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    for bar, val in zip(bars_is, cat_is):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.02 if val>=0 else -0.06),
                 f"{val:+.2f}", ha="center", color="#e6edf3", fontsize=7.5)
    for bar, val in zip(bars_oos, cat_oos):
        if not np.isnan(val):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height() + (0.02 if val>=0 else -0.06),
                     f"{val:+.2f}", ha="center", color="#e6edf3", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_order, rotation=30, ha="right", color="#8b949e", fontsize=8)
    ax2.set_title("카테고리별 평균 Sharpe (IS vs OOS)", color="#e6edf3")
    ax2.set_ylabel("Sharpe", color="#8b949e")
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 3. OOS 생존률 (카테고리별 OOS Sharpe>0 비율) ─────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    survival = []
    for cat in cat_order:
        sub = df[df["category"]==cat].dropna(subset=["oos_sharpe"])
        if not len(sub):
            survival.append(0)
            continue
        survival.append((sub["oos_sharpe"] > 0).mean() * 100)
    colors_s = ["#56d364" if s >= 50 else ("#ffab70" if s >= 30 else "#ff7b72")
                for s in survival]
    bars3 = ax3.bar(range(len(cat_order)), survival, color=colors_s, alpha=0.85)
    for bar, val in zip(bars3, survival):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val:.0f}%", ha="center", color="#e6edf3", fontsize=9)
    ax3.axhline(50, color="#f7931a", linewidth=1.2, linestyle="--", label="50% 기준선")
    ax3.set_xticks(range(len(cat_order)))
    ax3.set_xticklabels(cat_order, rotation=30, ha="right", color="#8b949e", fontsize=8)
    ax3.set_ylim(0, 115)
    ax3.set_title("카테고리별 OOS 생존률 (OOS Sharpe>0 코인 비율)", color="#e6edf3")
    ax3.set_ylabel("OOS 생존률 (%)", color="#8b949e")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    # ── 4. 코인별 Sharpe 히트맵 ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    _style(ax4)
    ax4.axis("off")

    # 카테고리순 정렬
    df_sorted = pd.concat([
        df[df["category"]==c].sort_values("is_sharpe", ascending=False)
        for c in cat_order
    ])

    col_labels = ["카테고리", "코인", "IS Sharpe", "IS 거래(4년)", "IS WR",
                  "OOS Sharpe", "OOS 거래(1.4년)", "OOS WR", "판정"]
    table_data = []
    for _, row in df_sorted.iterrows():
        oos_s = f"{row['oos_sharpe']:+.3f}" if not np.isnan(row["oos_sharpe"]) else "N/A"
        oos_w = f"{row['oos_wr']:.0f}%" if not np.isnan(row["oos_wr"]) else "N/A"
        if not np.isnan(row["oos_sharpe"]) and row["is_sharpe"] > 0 and row["oos_sharpe"] > 0:
            verdict = "타임리스"
        elif not np.isnan(row["oos_sharpe"]) and row["is_sharpe"] > 0 and row["oos_sharpe"] <= 0:
            verdict = "과적합"
        elif not np.isnan(row["oos_sharpe"]) and row["is_sharpe"] <= 0:
            verdict = "비적합"
        else:
            verdict = "데이터부족"
        table_data.append([
            row["category"], row["label"],
            f"{row['is_sharpe']:+.3f}", str(row["is_trades"]),
            f"{row['is_wr']:.0f}%",
            oos_s, str(row["oos_trades"]), oos_w,
            verdict
        ])

    tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    verdict_colors = {
        "타임리스": "#1a3a1a",
        "과적합":   "#3a1a1a",
        "비적합":   "#2a2a2a",
        "데이터부족": "#1a1a2a",
    }
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#21262d")
        else:
            v = table_data[r-1][8] if r-1 < len(table_data) else ""
            cell.set_facecolor(verdict_colors.get(v, "#161b22"))
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax4.set_title("전체 코인 결과 (초록=타임리스 / 빨강=과적합)", color="#e6edf3",
                  fontsize=10, pad=8)

    out = RESULTS_DIR / "exp14_category_oos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n저장: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("실험 14: 카테고리별 전략 유효성 + OOS 검증")
    print(f"  IS : {START} ~ {IS_END}")
    print(f"  OOS: {IS_END} ~ {OOS_END}")
    print(f"  코인: {sum(len(v) for v in CATEGORIES.values())}개 / {len(CATEGORIES)}개 카테고리")
    print("=" * 70)

    print("\n[온체인 데이터 수집...]")
    funding  = collect_funding_rate("BTCUSDT", START, OOS_END)
    fg       = collect_fear_greed(START, OOS_END)

    print("\n[코인 데이터 로드...]")
    coin_data = {}
    failed    = []
    for cat, syms in CATEGORIES.items():
        for sym in syms:
            lbl = LABEL.get(sym, sym)
            try:
                df  = collect(sym, "1h", START, OOS_END)
                mpe = rolling_mpe(df["close"], window=168,
                                  cache_key=f"{sym}_1h_{START}_{OOS_END}")
                h   = combined_onchain_entropy(
                    funding_entropy(funding, df.index),
                    fear_greed_entropy(fg, df.index),
                )
                coin_data[sym] = {"df": df, "mpe": mpe, "h_onchain": h}
                print(f"  [{lbl:6s}] 로드 완료 ({len(df):,}봉)")
            except Exception as e:
                print(f"  [{lbl:6s}] 실패: {e}")
                failed.append(sym)

    if failed:
        print(f"\n실패 코인: {[LABEL.get(s,s) for s in failed]}")

    print(f"\n[전략 실행 중... ({len(coin_data)}개 코인)]")
    result_df = run_all(coin_data)

    print_report(result_df)
    plot_results(result_df)

    # 핵심 결론
    timeless = result_df[(result_df["is_sharpe"] > 0) & (result_df["oos_sharpe"] > 0)]
    overfit  = result_df[(result_df["is_sharpe"] > 0) & (result_df["oos_sharpe"] <= 0)]
    unfit    = result_df[result_df["is_sharpe"] <= 0]

    print(f"\n{'=' * 70}")
    print(f"  실험 14 결론")
    print(f"{'=' * 70}")
    print(f"  타임리스 (IS+/OOS+): {len(timeless)}개  {list(timeless['label'])}")
    print(f"  과적합   (IS+/OOS-): {len(overfit)}개   {list(overfit['label'])}")
    print(f"  비적합   (IS-):      {len(unfit)}개     {list(unfit['label'])}")

    # 카테고리별 결론
    print(f"\n  카테고리별 타임리스 비율:")
    for cat in CATEGORIES:
        sub      = result_df[result_df["category"] == cat]
        n_tl     = len(sub[(sub["is_sharpe"]>0) & (sub["oos_sharpe"]>0)])
        n_total  = len(sub.dropna(subset=["oos_sharpe"]))
        if n_total:
            print(f"    {cat:<14}: {n_tl}/{n_total} ({n_tl/n_total*100:.0f}%)")

    result_df.to_csv(RESULTS_DIR / "exp14_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: results/exp14_results.csv")


if __name__ == "__main__":
    main()
