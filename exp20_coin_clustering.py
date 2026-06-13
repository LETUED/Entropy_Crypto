"""
실험 20: 코인 효율성 클러스터링

목적: 20개 코인을 대상으로 전략 성과 + MPE 효율성 점수 분석
     → 만성 비효율 코인(전략에 최적인 코인) 과학적 선별
     → 현재 5코인 포트폴리오 재검토

방법:
  1. 20개 코인 × 2021-01-01 ~ 2025-01-01 전체 기간 백테스트
  2. 성과 지표: 거래수, 평균PnL%, Sharpe, 승률, MDD
  3. 효율성 점수: 정규화 MPE 평균 (낮을수록 비효율 = 우리에게 유리)
     - 이론 최대 MPE (m=3): ln(6) ≈ 1.7918
     - 정규화 = 실제MPE / ln(6)  → 1에 가까울수록 효율적(랜덤워크)
  4. 랭킹 후 최적 코인셋 선별

이론 근거:
  - Zunino et al. (2009): 낮은 PE = 시장 비효율 = 알파 기회
  - PMC 2019 (75개 암호화폐): 만성 비효율 코인 20%에서 기회 집중

실행: py exp20_coin_clustering.py
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
from tqdm import tqdm

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h4_backtest import kelly_size, compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi, generate_signals

RESULTS_DIR = Path("results")
START, END  = "2021-01-01", "2025-01-01"

MPE_WINDOW  = 168
MPE_SCALES  = [1, 2, 4, 8]
MPE_M       = 3
MPE_MAX     = np.log(6)   # ln(3!) — m=3 이론 최대 MPE

# ── 테스트 코인 20개 (순수 시장 코인, 2021 이전 상장) ─────────────────────────
COINS = {
    "BTCUSDT":  "BTC",   "ETHUSDT":  "ETH",   "SOLUSDT":  "SOL",
    "AVAXUSDT": "AVAX",  "ADAUSDT":  "ADA",   "DOTUSDT":  "DOT",
    "LTCUSDT":  "LTC",   "LINKUSDT": "LINK",  "ATOMUSDT": "ATOM",
    "DOGEUSDT": "DOGE",  "XRPUSDT":  "XRP",   "ALGOUSDT": "ALGO",
    "MATICUSDT":"MATIC", "FILUSDT":  "FIL",   "UNIUSDT":  "UNI",
    "AAVEUSDT": "AAVE",  "NEARUSDT": "NEAR",  "GRTUSDT":  "GRT",
    "TRXUSDT":  "TRX",   "SUSHIUSDT":"SUSHI",
}

ORIGINAL_5 = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def run_coin(sym: str, df: pd.DataFrame, mpe: pd.Series,
             fund_raw: pd.Series, fg_raw: pd.Series) -> dict:
    """단일 코인 전체 기간 백테스트 → 성과 + 효율성 점수 반환."""
    h_fund    = funding_entropy(fund_raw, df.index)
    h_fg      = fear_greed_entropy(fg_raw, df.index)
    h_onchain = combined_onchain_entropy(h_fund, h_fg)

    rsi    = compute_rsi(df["close"])
    ma200  = df["close"].rolling(MA_PERIOD).mean()
    oc_thr = np.percentile(h_onchain.dropna(), 40)
    mpe_thr = np.percentile(mpe.dropna(), ENTROPY_PCT)

    equity      = 1.0
    position    = 0
    entry_price = 0.0
    entry_bar   = -999
    curve       = []
    trades      = []

    for i, idx in enumerate(df.index):
        price    = df["close"].loc[idx]
        rsi_val  = rsi.loc[idx]   if idx in rsi.index   else np.nan
        mpe_val  = mpe.loc[idx]   if idx in mpe.index   else np.nan
        oc_val   = h_onchain.loc[idx] if idx in h_onchain.index else np.nan
        ma_val   = ma200.loc[idx] if idx in ma200.index else np.nan

        # 청산 조건
        if position == 1:
            held = i - entry_bar
            exit_sig = (not np.isnan(rsi_val) and rsi_val > 50) or (held >= MAX_HOLD_H)
            if exit_sig:
                pnl = (price - entry_price) / entry_price - FEE_RATE
                equity *= (1 + pnl)
                trades.append(pnl)
                position = 0

        # 진입 조건
        if position == 0:
            long_ok = (
                not np.isnan(rsi_val) and rsi_val < 30 and
                not np.isnan(mpe_val) and mpe_val <= mpe_thr and
                not np.isnan(ma_val)  and price > ma_val and
                not np.isnan(oc_val)  and oc_val <= oc_thr
            )
            if long_ok:
                entry_price = price * (1 + FEE_RATE)
                entry_bar   = i
                position    = 1

        curve.append(equity)

    # 미청산 포지션 강제 마감
    if position == 1:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - entry_price) / entry_price - FEE_RATE
        equity *= (1 + pnl)
        trades.append(pnl)

    trades = np.array(trades)
    n = len(trades)

    if n == 0:
        return {
            "sym": sym, "n_trades": 0, "mean_pnl": 0.0, "sharpe": 0.0,
            "win_rate": 0.0, "mdd": 0.0, "efficiency": 1.0,
            "total_return": 0.0,
        }

    mean_pnl   = trades.mean() * 100
    sharpe     = trades.mean() / (trades.std() + 1e-9) * np.sqrt(252 * 24 / MAX_HOLD_H)
    win_rate   = (trades > 0).mean() * 100
    eq_arr     = np.array(curve)
    peak       = np.maximum.accumulate(eq_arr)
    mdd        = ((eq_arr - peak) / peak).min() * 100
    total_ret  = (equity - 1) * 100

    # 효율성 점수: 평균 정규화 MPE (낮을수록 비효율 = 우리에게 유리)
    eff_score  = float((mpe.dropna() / MPE_MAX).mean())

    return {
        "sym": sym, "n_trades": n, "mean_pnl": mean_pnl,
        "sharpe": sharpe, "win_rate": win_rate, "mdd": mdd,
        "efficiency": eff_score, "total_return": total_ret,
    }


def main():
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 온체인 데이터 (공통 원본 — 각 코인 price_index로 재샘플은 run_coin 내부) ──
    print("온체인 데이터 로드...")
    fund_raw = collect_funding_rate("BTCUSDT", START, END)
    fg_raw   = collect_fear_greed(START, END)

    # ── 코인별 백테스트 ─────────────────────────────────────────────────────
    results = []
    print(f"\n{len(COINS)}개 코인 백테스트 시작...")

    for sym, label in tqdm(COINS.items(), desc="코인"):
        try:
            df = collect(sym, "1h", START, END)
            if df is None or len(df) < MPE_WINDOW + MA_PERIOD + 50:
                print(f"  {label}: 데이터 부족 — 건너뜀")
                continue

            cache_key = f"{sym}_1h_{START}_{END}"
            mpe = rolling_mpe(df["close"], window=MPE_WINDOW,
                              m=MPE_M, scales=MPE_SCALES,
                              cache_key=cache_key)

            res = run_coin(sym, df, mpe, fund_raw, fg_raw)
            res["label"] = label
            results.append(res)

        except Exception as e:
            print(f"  {sym} 오류: {e}")
            continue

    if not results:
        print("결과 없음")
        return

    df_res = pd.DataFrame(results).set_index("sym")
    df_res = df_res.sort_values("sharpe", ascending=False)

    # ── 결과 출력 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'코인':<8} {'거래수':>6} {'평균PnL%':>9} {'Sharpe':>8} {'승률%':>7} "
          f"{'MDD%':>7} {'효율성':>7} {'총수익%':>8}")
    print("-" * 75)
    for sym, row in df_res.iterrows():
        marker = " ◀" if sym in ORIGINAL_5 else ""
        print(f"{row['label']:<8} {row['n_trades']:>6} {row['mean_pnl']:>+9.3f} "
              f"{row['sharpe']:>8.3f} {row['win_rate']:>7.1f} "
              f"{row['mdd']:>7.2f} {row['efficiency']:>7.3f} "
              f"{row['total_return']:>+8.2f}{marker}")
    print("=" * 75)
    print("◀ = 현재 5코인 포트폴리오")

    # 상위 5코인, 상위 8코인 추천
    top5  = df_res[df_res["n_trades"] >= 5].head(5)
    top8  = df_res[df_res["n_trades"] >= 5].head(8)
    print(f"\n[추천 5코인] {list(top5['label'])}")
    print(f"[추천 8코인] {list(top8['label'])}")

    orig_mean_sharpe = df_res.loc[
        [s for s in ORIGINAL_5 if s in df_res.index], "sharpe"
    ].mean()
    top5_mean_sharpe = top5["sharpe"].mean()
    print(f"\n현재 5코인 평균 Sharpe: {orig_mean_sharpe:.3f}")
    print(f"추천 5코인 평균 Sharpe: {top5_mean_sharpe:.3f}")
    print(f"개선폭: {top5_mean_sharpe - orig_mean_sharpe:+.3f}")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    labels     = df_res["label"].tolist()
    sharpes    = df_res["sharpe"].tolist()
    mean_pnls  = df_res["mean_pnl"].tolist()
    eff_scores = df_res["efficiency"].tolist()
    n_trades   = df_res["n_trades"].tolist()
    colors     = ["#f78166" if s in ORIGINAL_5 else "#58a6ff"
                  for s in df_res.index]

    # 1) Sharpe 랭킹
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(labels[::-1], sharpes[::-1], color=colors[::-1])
    ax1.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_title("코인별 Sharpe 랭킹")
    ax1.tick_params(labelsize=9)

    # 2) 효율성 점수 vs Sharpe 산점도
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (sym, row) in enumerate(df_res.iterrows()):
        c = "#f78166" if sym in ORIGINAL_5 else "#58a6ff"
        ax2.scatter(row["efficiency"], row["sharpe"], s=row["n_trades"] * 8 + 30,
                    color=c, alpha=0.8, zorder=3)
        ax2.annotate(row["label"], (row["efficiency"], row["sharpe"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("효율성 점수 (낮을수록 비효율 = 유리)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("효율성 vs 성과 (원 크기 = 거래수)")

    # 3) 평균 PnL% 랭킹
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(labels[::-1], mean_pnls[::-1], color=colors[::-1])
    ax3.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("평균 PnL per Trade (%)")
    ax3.set_title("코인별 평균 PnL 랭킹")
    ax3.tick_params(labelsize=9)

    # 4) 거래수 vs Sharpe
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (sym, row) in enumerate(df_res.iterrows()):
        c = "#f78166" if sym in ORIGINAL_5 else "#58a6ff"
        ax4.scatter(row["n_trades"], row["sharpe"], s=80, color=c, alpha=0.8, zorder=3)
        ax4.annotate(row["label"], (row["n_trades"], row["sharpe"]),
                     textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax4.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax4.set_xlabel("거래수 (2021-2025)")
    ax4.set_ylabel("Sharpe Ratio")
    ax4.set_title("거래 빈도 vs 성과")

    # 범례
    from matplotlib.patches import Patch
    legend = [Patch(color="#f78166", label="현재 5코인"),
              Patch(color="#58a6ff", label="신규 후보")]
    fig.legend(handles=legend, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), fontsize=10)

    fig.patch.set_facecolor("#0d1117")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.suptitle("Exp20: 코인 효율성 클러스터링 (20개 코인, 2021-2025)",
                 color="white", fontsize=13, y=1.01)

    out = RESULTS_DIR / "exp20_coin_clustering.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {out}")

    # ── CSV 저장 ────────────────────────────────────────────────────────────
    csv_out = RESULTS_DIR / "exp20_coin_clustering.csv"
    df_res.to_csv(csv_out)
    print(f"데이터 저장: {csv_out}")


if __name__ == "__main__":
    main()
