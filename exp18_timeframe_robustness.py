"""
실험 18: 타임프레임 강건성 검증

질문: 전략이 1H 봉에 과적합된 것인가, 아니면 타임프레임 독립적인 엣지인가?

방법: 동일 전략을 4H 봉에 적용
  - 1H 기준: window=168봉 (= 168시간 = 7일)
  - 4H 기준: window=42봉  (= 42 × 4H = 168시간 = 7일) → 동일한 물리적 윈도우
  - 파라미터: RSI window도 비율 유지 (RSI 14봉 → 14봉 그대로)
  - 청산: RSI>50 OR MAX_HOLD (1H: 168봉 → 4H: 42봉, 동일 168시간)
  - MA200: 1H 200봉 (8.3일) → 4H 200봉 (33.3일) — 더 장기 추세 필터

비교:
  1H: 기존 결과 (Sharpe +0.804, 93건, 2021-2025)
  4H: 새로 계산

핵심 인사이트: "타임리스 공식"이라면 어느 시간축에서도 신호가 살아있어야 함.

실행: py exp18_timeframe_robustness.py
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
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import FEE_RATE, ENTROPY_PCT

RESULTS_DIR = Path("results")
START, END   = "2021-01-01", "2025-01-01"

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT":"BTC","SOLUSDT":"SOL","AVAXUSDT":"AVAX","ADAUSDT":"ADA","DOTUSDT":"DOT"}

# ── 타임프레임 설정 ────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "1h": {
        "interval":   "1h",
        "mpe_window": 168,    # 168봉 = 168H
        "ma_period":  200,    # 200봉 = 8.3일
        "max_hold":   168,    # 168봉 = 168H
        "rsi_window": 14,
        "annualize":  np.sqrt(24 * 365),
    },
    "4h": {
        "interval":   "4h",
        "mpe_window": 42,     # 42봉 = 168H (동일 물리적 윈도우)
        "ma_period":  50,     # 50봉 × 4H = 200H ≈ MA200(1H) 동일 물리적 기준
        "max_hold":   42,     # 42봉 = 168H
        "rsi_window": 14,
        "annualize":  np.sqrt(6 * 365),   # 4H → 6봉/일
    },
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


# ── 단일 타임프레임 전략 실행 ────────────────────────────────────────────────
def run_strategy_tf(df, mpe, h_onchain, cfg):
    """
    타임프레임 독립 전략 실행.
    cfg: TIMEFRAMES[tf] 딕셔너리
    """
    ma = df["close"].rolling(cfg["ma_period"]).mean()

    # RSI — 봉 단위 (창 개수는 tf와 무관하게 14봉 고정)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(cfg["rsi_window"]).mean()
    loss  = (-delta.clip(upper=0)).rolling(cfg["rsi_window"]).mean()
    rs    = gain / (loss + 1e-12)
    rsi   = 100 - 100 / (1 + rs)

    mpe_thresh = np.percentile(mpe.dropna(), ENTROPY_PCT)
    oc_thresh  = np.percentile(h_onchain.dropna(), 40) if h_onchain is not None else None

    mpe_clean = mpe.dropna()
    k_pct1  = np.percentile(mpe_clean, 1)
    k_pct5  = np.percentile(mpe_clean, 5)
    k_pct10 = np.percentile(mpe_clean, 10)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity      = 1.0
    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    entry_kfrac = 0.0
    trades      = []
    curve       = []

    idx_list = df.index.tolist()
    n = len(idx_list)

    for i, idx in enumerate(idx_list):
        price   = df["close"].iloc[i]
        rsi_val = rsi.iloc[i]   if i < n else 50.0
        ma_val  = ma.iloc[i]    if i < n else np.nan
        mpe_val = mpe.loc[idx]  if idx in mpe.index else np.nan

        oc_ok = True
        if h_onchain is not None and oc_thresh is not None and idx in h_onchain.index:
            oc_ok = h_onchain.loc[idx] <= oc_thresh

        if position == 1:
            held = i - entry_bar
            if rsi_val > 50 or held >= cfg["max_hold"]:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl": pnl, "kfrac": entry_kfrac, "held_bars": held})
                position = 0

        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            is_low   = mpe_val <= mpe_thresh
            above_ma = price > ma_val
            k_frac   = _kelly(mpe_val)
            # RSI<30 진입
            if rsi_val < 30 and is_low and above_ma and k_frac > 0 and oc_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_bar   = i
                entry_kfrac = k_frac

        curve.append(equity)

    if position == 1:
        pnl = (df["close"].iloc[-1] - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    eq = pd.Series(curve, index=idx_list)
    return eq, trades


# ── 포트폴리오 Sharpe 계산 ────────────────────────────────────────────────────
def portfolio_sharpe(eq_list, annualize_factor):
    port = sum(e / len(eq_list) for e in eq_list)
    ret  = port.pct_change().dropna()
    return float(ret.mean() / (ret.std() + 1e-12) * annualize_factor), port


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Exp18: 타임프레임 강건성 검증 (1H vs 4H)")
    print("=" * 65)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    results = {}   # tf → {sym: (eq, trades)}

    for tf, cfg in TIMEFRAMES.items():
        print(f"\n{'─'*50}")
        print(f"[{tf.upper()} 데이터 로드 및 전략 실행...]")
        results[tf] = {}

        for sym in COINS:
            df  = collect(sym, cfg["interval"], START, END)
            mpe = rolling_mpe(df["close"],
                              window=cfg["mpe_window"],
                              cache_key=f"{sym}_{cfg['interval']}_{START}_{END}")

            h_f  = funding_entropy(funding, df.index)
            h_g  = fear_greed_entropy(fg, df.index)
            h_oc = combined_onchain_entropy(h_f, h_g)

            eq, trades = run_strategy_tf(df, mpe, h_oc, cfg)
            results[tf][sym] = (eq, trades)

            ret = eq.pct_change().dropna()
            s   = float(ret.mean() / (ret.std() + 1e-12) * cfg["annualize"])
            avg_pnl = np.mean([t["pnl"]*100 for t in trades]) if trades else 0.0
            print(f"  [{LABEL[sym]:6s}] Sharpe={s:+.3f}, 거래={len(trades)}, 평균PnL={avg_pnl:+.3f}%")

        eq_list = [results[tf][sym][0] for sym in COINS]
        port_s, port_eq = portfolio_sharpe(eq_list, cfg["annualize"])
        all_trades = []
        for sym in COINS:
            all_trades.extend(results[tf][sym][1])

        results[tf]["_port_sharpe"] = port_s
        results[tf]["_port_eq"]     = port_eq
        results[tf]["_all_trades"]  = all_trades
        print(f"\n  [{tf.upper()} 포트폴리오] Sharpe={port_s:+.4f}, 총거래={len(all_trades)}건")

    # ── 비교 테이블 출력 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("타임프레임 비교 요약")
    print("=" * 65)
    print(f"{'코인':>6}  {'1H Sharpe':>12}  {'1H 거래':>8}  {'4H Sharpe':>12}  {'4H 거래':>8}")
    print("-" * 60)
    for sym in COINS:
        for tf in ["1h","4h"]:
            eq, trades = results[tf][sym]
            ret = eq.pct_change().dropna()
            results[tf][sym + "_sharpe"] = float(ret.mean() / (ret.std() + 1e-12) * TIMEFRAMES[tf]["annualize"])
            results[tf][sym + "_n"]      = len(trades)

        s1 = results["1h"][sym + "_sharpe"]
        n1 = results["1h"][sym + "_n"]
        s4 = results["4h"][sym + "_sharpe"]
        n4 = results["4h"][sym + "_n"]
        consistent = "✅" if s1 > 0 and s4 > 0 else ("⚠️" if s1 > 0 or s4 > 0 else "❌")
        print(f"  {LABEL[sym]:>6}  {s1:>+12.3f}  {n1:>8}  {s4:>+12.3f}  {n4:>8}  {consistent}")

    s1p = results["1h"]["_port_sharpe"]
    s4p = results["4h"]["_port_sharpe"]
    n1p = len(results["1h"]["_all_trades"])
    n4p = len(results["4h"]["_all_trades"])
    print("-" * 60)
    print(f"  {'포트':>6}  {s1p:>+12.4f}  {n1p:>8}  {s4p:>+12.4f}  {n4p:>8}")

    consistent_port = s1p > 0 and s4p > 0
    print(f"\n  결론: {'두 타임프레임 모두 양수 — 타임리스 엣지 확인 ✅' if consistent_port else '타임프레임 일관성 없음 — 1H 과적합 의심 ⚠️'}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for col, (tf, cfg) in enumerate(TIMEFRAMES.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e")

        colors_c = ["#f7931a","#00c896","#e040fb","#40a9ff","#ff6b6b"]
        for j, sym in enumerate(COINS):
            eq, _ = results[tf][sym]
            norm  = eq / eq.iloc[0]
            ax.plot(eq.index, norm, linewidth=0.9, color=colors_c[j],
                    alpha=0.75, label=LABEL[sym])

        port_eq = results[tf]["_port_eq"]
        norm_p  = port_eq / port_eq.iloc[0]
        ax.plot(port_eq.index, norm_p, linewidth=2.0, color="white",
                label=f"Portfolio (Sharpe {results[tf]['_port_sharpe']:+.3f})")

        ax.axhline(1.0, color="#30363d", linewidth=0.8)
        ax.set_xlabel("날짜", color="#8b949e")
        ax.set_ylabel("자산 배수 (정규화)", color="#8b949e")
        ax.set_title(
            f"{tf.upper()} 타임프레임 — Sharpe {results[tf]['_port_sharpe']:+.4f}  |  {len(results[tf]['_all_trades'])}건",
            color="#e6edf3", fontsize=11
        )
        ax.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
        ax.yaxis.grid(True, color="#21262d", linestyle="--")

    consistent_str = "일관성 있음 ✅" if consistent_port else "불일치 ⚠️"
    fig.suptitle(
        f"Exp18: 타임프레임 강건성  |  1H Sharpe {s1p:+.3f}  vs  4H Sharpe {s4p:+.3f}  |  {consistent_str}",
        color="#e6edf3", fontsize=12, fontweight="bold"
    )

    path = RESULTS_DIR / "exp18_timeframe_robustness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장: {path}")
    plt.show()

    # ── 최종 평가 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("타임프레임 강건성 최종 평가")
    print("=" * 65)
    print(f"  1H 포트폴리오 Sharpe: {s1p:+.4f} ({n1p}건)")
    print(f"  4H 포트폴리오 Sharpe: {s4p:+.4f} ({n4p}건)")
    print(f"  비율 (4H/1H): {s4p/s1p:.2f}x")
    print()

    if s1p > 0 and s4p > 0:
        print("  [최종] 두 타임프레임 모두 양수")
        print("  → MPE 신호는 타임프레임 독립적 — '타임리스 공식' 가설 지지")
    elif s1p > 0 and s4p <= 0:
        print("  [최종] 1H는 양수지만 4H는 음수")
        print("  → 1H 봉 특화 패턴 가능성 — 추가 검증 필요")
    elif s1p <= 0 and s4p > 0:
        print("  [최종] 4H 더 좋음 — 데이터 문제 확인 필요")
    else:
        print("  [최종] 두 타임프레임 모두 음수 — 전략 재검토 필요")


if __name__ == "__main__":
    main()
