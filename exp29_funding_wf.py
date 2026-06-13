"""
실험 29: Funding Rate 역발상 신호 Walk-forward

기반: Exp28-D (진입 Exp23-D + 청산 TP+2% OR RSI>50 OR 168H, WF Sharpe +1.391, 37건)

목표: BTC 극단 음수 펀딩비를 역발상 신호로 활용해 품질 개선 or 거래 확장
  - 극단 음수 펀딩 = 숏 포지션 과열 (Crowded Short) → 반등 확률 ↑
  - 펀딩 정산(UTC 00:00/08:00/16:00) 후 2시간 대기 후 신호 유효

이론 근거:
  Kim et al. (2019) https://www.sciencedirect.com/article/pii/S037843711930216X
    → 단기 역발상 효과 IR 20+ — 극단 음수 펀딩 = 숏 혼잡 = 반등 가능성
  Temporal Dynamics (2026) https://www.mdpi.com/2227-7072/14/5/103
    → 8시간 정산 후 2시간 대기가 최적 진입 타이밍
  Crypto Return Prediction (2024) https://arxiv.org/abs/2410.14532
    → 펀딩 단독 신호 약함, MPE + 펀딩 조합 시 예측력 대폭 향상
  Funding Rate Arbitrage (2025) https://www.sciencedirect.com/article/pii/S2096720925000818
    → 2025년 이후 CEX 펀딩 수익성 악화 → 극단값(±0.05% 이상)만 유효 신호

비교 조건:
  A (기준):  Exp28-D 그대로 (37건, +1.391)
  B (필터):  Exp28-D + BTC_funding < -0.05%/8h AND 정산 후 2H 대기
             → 거래 감소 예상, 승률/Sharpe 개선 여부 측정
  C (완화):  BTC_funding < -0.05% 구간에서 MPE 임계 10%ile → 15%ile 완화
             → 동적 MPE 임계, 거래 소폭 추가 가능
  D (확장):  BTC_funding < -0.05% 구간에서 MPE<15%ile + vol>=30%ile + RSI<35
             → 최대 확장: 극단 펀딩 시 가장 넓은 진입 조건

청산 (모든 모드 동일): TP+2.0% OR RSI>50 OR 168H (Exp28-D 결과)

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp29_funding_wf.py
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
from src.analysis.h4_backtest import FEE_RATE, MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
         "ADAUSDT": "ADA", "DOTUSDT": "DOT"}

MPE_WINDOW   = 168
MPE_M        = 3
MPE_SCALES   = [1, 2, 4, 8]

# ── 진입 상수 ─────────────────────────────────────────────────────────────────
ENTROPY_PCT   = 10    # 기본 MPE 임계 (하위 10%ile)
ENTROPY_RELAX = 15    # 완화 MPE 임계 (하위 15%ile, 모드 C/D)
VOL_PCT       = 50    # 기본 볼륨 MPE >= 50%ile
VOL_RELAX_PCT = 30    # 완화 볼륨 MPE >= 30%ile (모드 D)
RSI_OVERSOLD  = 30    # 기본 RSI 임계
RSI_RELAX     = 35    # 완화 RSI 임계 (모드 D)

# ── 펀딩비 상수 ───────────────────────────────────────────────────────────────
# BTC 실제 분포: 음수 12%, -0.005%이하 4.6%(200건), -0.01%이하 1.8%(78건)
# 논문 기준 -0.05%는 현실에서 5건(0.1%)뿐 → 실용 임계로 완화
FUNDING_THRESH = -0.0001   # -0.01%/8h (극단 음수 기준, 78건/4383건 = 1.8%)
FUNDING_WAIT_H = 2         # 정산 후 대기 시간 (UTC 시 % 8 >= 2)

# ── 청산 상수 ─────────────────────────────────────────────────────────────────
TP_PCT = 2.0   # 목표 수익률 (Exp28-D 승리 조건)


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, price_mpe_test, vol_mpe_test,
                h_test, price_mpe_train, vol_mpe_train,
                oc_threshold, k_pct1, k_pct5, k_pct10,
                btc_funding_raw,
                mode="A"):
    """
    진입 (mode별 차이):
      A — Exp28-D 기준 (MPE<10% + vol>=50% + RSI<30 + MA200 + 온체인<40%)
      B — A + BTC_funding < -0.05% AND 정산 후 2H 대기
      C — 펀딩<-0.05% 시 MPE<15%ile 완화, 아니면 10%ile (나머지 동일)
      D — 펀딩<-0.05% 시 MPE<15% + vol>=30% + RSI<35, 아니면 기본 A 조건

    청산 (모든 모드): TP+2% OR RSI>50 OR 168H (Exp28-D)
    """
    rsi   = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    # MPE 임계값 (train 기간 기반)
    p_train_clean  = price_mpe_train.dropna()
    price_thr      = np.percentile(p_train_clean, ENTROPY_PCT)    # 10%ile
    price_thr_15   = np.percentile(p_train_clean, ENTROPY_RELAX)  # 15%ile

    vol_train_c = vol_mpe_train.dropna()
    vol_thr_50  = np.percentile(vol_train_c, VOL_PCT)      if len(vol_train_c) > 10 else np.nan
    vol_thr_30  = np.percentile(vol_train_c, VOL_RELAX_PCT) if len(vol_train_c) > 10 else np.nan

    # BTC 펀딩비를 1H 인덱스로 ffill
    funding_1h = btc_funding_raw.reindex(df_test.index, method="ffill")

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity      = 1.0
    position    = 0
    entry_price = entry_hour = 0
    entry_kfrac = 0.0
    trades      = []
    curve       = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe   = price_mpe_test.loc[idx] if idx in price_mpe_test.index else np.nan
        v_mpe   = vol_mpe_test.loc[idx]   if idx in vol_mpe_test.index   else np.nan

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        vol_ok_50 = (not np.isnan(v_mpe)
                     and not np.isnan(vol_thr_50)
                     and v_mpe >= vol_thr_50)

        vol_ok_30 = (not np.isnan(v_mpe)
                     and not np.isnan(vol_thr_30)
                     and v_mpe >= vol_thr_30)

        # BTC 펀딩비 상태
        fund_val = funding_1h.loc[idx] if idx in funding_1h.index else np.nan
        wait_ok  = (idx.hour % 8 >= FUNDING_WAIT_H)  # 정산(UTC 0/8/16시) 후 2H 경과
        extreme_funding = (not np.isnan(fund_val)
                           and fund_val < FUNDING_THRESH
                           and wait_ok)

        # 청산 (모든 모드 동일: TP+2% OR RSI>50 OR 168H)
        if position == 1:
            held = i - entry_hour
            current_pnl_pct = (price - entry_price) / entry_price * 100
            exit_ok = (current_pnl_pct >= TP_PCT or rsi_val > 50 or held >= MAX_HOLD_H)

            if exit_ok:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        # 진입 (mode별 분기)
        if position == 0:
            if mode == "A":
                # 기준 (Exp28-D)
                entry_ok = (rsi_val < RSI_OVERSOLD
                            and not np.isnan(p_mpe)
                            and not np.isnan(ma_val)
                            and p_mpe <= price_thr
                            and price > ma_val
                            and oc_ok
                            and vol_ok_50)

            elif mode == "B":
                # A 조건 + 극단 펀딩 필터
                entry_ok = (extreme_funding
                            and rsi_val < RSI_OVERSOLD
                            and not np.isnan(p_mpe)
                            and not np.isnan(ma_val)
                            and p_mpe <= price_thr
                            and price > ma_val
                            and oc_ok
                            and vol_ok_50)

            elif mode == "C":
                # 동적 MPE 임계: 펀딩 극단이면 15%ile, 아니면 10%ile
                eff_thr = price_thr_15 if extreme_funding else price_thr
                entry_ok = (rsi_val < RSI_OVERSOLD
                            and not np.isnan(p_mpe)
                            and not np.isnan(ma_val)
                            and p_mpe <= eff_thr
                            and price > ma_val
                            and oc_ok
                            and vol_ok_50)

            else:  # D
                # 동적 MPE+볼륨+RSI: 펀딩 극단이면 3가지 완화
                if extreme_funding:
                    eff_thr     = price_thr_15
                    eff_vol_ok  = vol_ok_30
                    eff_rsi_thr = RSI_RELAX
                else:
                    eff_thr     = price_thr
                    eff_vol_ok  = vol_ok_50
                    eff_rsi_thr = RSI_OVERSOLD

                entry_ok = (rsi_val < eff_rsi_thr
                            and not np.isnan(p_mpe)
                            and not np.isnan(ma_val)
                            and p_mpe <= eff_thr
                            and price > ma_val
                            and oc_ok
                            and eff_vol_ok)

            if entry_ok:
                k_frac = _kelly(p_mpe)
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
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, h_onchain, btc_funding_raw):
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

        df_train = df[train_start:train_end]
        df_test  = df[test_start:test_end]
        if len(df_train) < 500 or len(df_test) < 100:
            cursor += test_delta
            continue

        price_mpe_train = price_mpe[train_start:train_end]
        price_mpe_test  = price_mpe[test_start:test_end]
        vol_mpe_train   = vol_mpe[train_start:train_end]
        vol_mpe_test    = vol_mpe[test_start:test_end]

        if h_onchain is not None:
            h_test = h_onchain[test_start:test_end]
            oc_thr = np.percentile(h_onchain[train_start:train_end].dropna(), 40)
        else:
            h_test = None
            oc_thr = None

        p_clean = price_mpe_train.dropna()
        if len(p_clean) < 20:
            cursor += test_delta
            continue

        k1  = np.percentile(p_clean, 1)
        k5  = np.percentile(p_clean, 5)
        k10 = np.percentile(p_clean, 10)

        for mode in ["A", "B", "C", "D"]:
            curve, trades = _run_window(
                df_test, price_mpe_test, vol_mpe_test,
                h_test, price_mpe_train, vol_mpe_train,
                oc_thr, k1, k5, k10,
                btc_funding_raw,
                mode=mode
            )
            ret = curve.pct_change().dropna()
            sharpe  = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(24 * 365))
            ret_pct = float((curve.iloc[-1] / curve.iloc[0] - 1) * 100)
            avg_held = (np.mean([t["held_h"] for t in trades])
                        if trades else 0.0)
            results_by_mode[mode].append({
                "period": f"{test_start.strftime('%Y-%m')} ~ {(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}",
                "sharpe": sharpe,
                "ret_pct": ret_pct,
                "n_trades": len(trades),
                "win_rate": (sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades)
                             if trades else 0.0),
                "avg_held_h": avg_held,
                "curve": curve,
            })

        cursor += test_delta

    label = LABEL.get(sym, sym[:4])
    for mode in ["A", "B", "C", "D"]:
        rows = results_by_mode[mode]
        sharpes = [r["sharpe"] for r in rows]
        trades  = sum(r["n_trades"] for r in rows)
        avg_h   = np.mean([r["avg_held_h"] for r in rows]) if rows else 0
        wf_sharpe = np.mean(sharpes) if sharpes else 0.0
        print(f"  [{label:6s} {mode}] Sharpe={wf_sharpe:+.3f}  "
              f"양수:{sum(1 for s in sharpes if s>0)}/{len(sharpes)}  "
              f"거래:{trades}  평균보유:{avg_h:.1f}H")

    return results_by_mode


# ── 포트폴리오 집계 ───────────────────────────────────────────────────────────
def aggregate_portfolio(all_results):
    modes   = ["A", "B", "C", "D"]
    n_coins = len(COINS)
    combined = {m: [] for m in modes}

    n_windows = len(next(iter(all_results.values()))["A"])
    for wi in range(n_windows):
        for mode in modes:
            port_curve = None
            for sym in COINS:
                c = all_results[sym][mode][wi]["curve"]
                port_curve = c if port_curve is None else port_curve + c
            if port_curve is not None:
                port_curve = port_curve / n_coins
                ret = port_curve.pct_change().dropna()
                sharpe  = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(24 * 365))
                ret_pct = float((port_curve.iloc[-1] / port_curve.iloc[0] - 1) * 100)
                ref_row = all_results[COINS[0]][mode][wi]
                combined[mode].append({
                    "period":    ref_row["period"],
                    "sharpe":    sharpe,
                    "ret_pct":   ret_pct,
                    "n_trades":  sum(all_results[sym][mode][wi]["n_trades"]  for sym in COINS),
                    "win_rate":  np.mean([all_results[sym][mode][wi]["win_rate"] for sym in COINS]),
                    "avg_held_h":np.mean([all_results[sym][mode][wi]["avg_held_h"] for sym in COINS]),
                    "curve":     port_curve,
                })
    return combined


def print_portfolio_results(combined):
    modes = ["A", "B", "C", "D"]
    MODE_DESC = {
        "A": "기준 (Exp28-D)",
        "B": "펀딩<-0.05% 필터",
        "C": "펀딩<-0.05%→MPE 15%ile 완화",
        "D": "펀딩<-0.05%→MPE+vol+RSI 완화",
    }

    print("\n" + "="*70)
    print("  포트폴리오 결과 (O5 균등 배분)")
    print("="*70)

    summary = {}
    for mode in modes:
        rows = combined[mode]
        print(f"\n{'='*70}")
        print(f"  모드 {mode}: {MODE_DESC[mode]}")
        print(f"{'='*70}")
        print(f"  {'기간':20s} {'Sharpe':>8} {'수익률':>8} {'거래수':>6} {'승률':>7} {'평균보유':>8}")
        print("  " + "-"*62)
        for r in rows:
            sign = "[+]" if r["sharpe"] > 0 else "[-]"
            print(f"  {sign} {r['period']:20s} {r['sharpe']:>8.3f} "
                  f"{r['ret_pct']:>8.2f}% {r['n_trades']:>6} "
                  f"{r['win_rate']:>7.1%} {r['avg_held_h']:>7.1f}H")

        sharpes   = [r["sharpe"] for r in rows]
        avg_sharp = np.mean(sharpes)
        n_pos     = sum(1 for s in sharpes if s > 0)
        tot_trade = sum(r["n_trades"] for r in rows)
        avg_h     = np.mean([r["avg_held_h"] for r in rows])
        print("  " + "-"*62)
        print(f"  평균 Sharpe: {avg_sharp:+.3f}  |  "
              f"양수: {n_pos}/{len(sharpes)}  |  "
              f"총 거래: {tot_trade}  |  평균 보유: {avg_h:.1f}H")
        summary[mode] = {"sharpe": avg_sharp, "n_pos": n_pos,
                         "trades": tot_trade, "avg_h": avg_h}

    print("\n" + "="*70)
    print(f"  요약 비교 (기준: A = Exp28-D +1.391, 37건)")
    base_sharpe = summary["A"]["sharpe"]
    print(f"  {'조건':5s} | {'평균 Sharpe':>13} | {'개선폭':>10} | {'양수구간':>10} | {'거래수':>6} | {'평균보유':>8}")
    print("  " + "-"*68)
    for mode in modes:
        s    = summary[mode]
        diff = s["sharpe"] - base_sharpe
        star = " ★" if s["sharpe"] == max(v["sharpe"] for v in summary.values()) else ""
        print(f"  {mode}{star:2s}   | {s['sharpe']:>13.3f} | {diff:>+10.3f} | "
              f"{s['n_pos']}/{len(combined[mode]):>6}  | {s['trades']:>6} | {s['avg_h']:>6.1f}H")
    print("="*70)

    best_mode = max(summary, key=lambda m: summary[m]["sharpe"])
    b = summary[best_mode]
    print(f"\n  최적 모드: {best_mode} "
          f"(Sharpe={b['sharpe']:+.3f}, 거래={b['trades']}건, "
          f"평균보유={b['avg_h']:.1f}H)")
    return summary


def plot_results(combined, summary):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    modes     = ["A", "B", "C", "D"]
    n_windows = len(combined["A"])
    periods   = [r["period"] for r in combined["A"]]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors  = {"A": "#4878cf", "B": "#6acc65", "C": "#d65f5f", "D": "#ee854a"}
    markers = {"A": "o", "B": "s", "C": "D", "D": "^"}

    # 상단 2: 구간별 Sharpe
    ax_top = fig.add_subplot(gs[0, :])
    x = np.arange(n_windows)
    w = 0.18
    for j, mode in enumerate(modes):
        sharpes = [r["sharpe"] for r in combined[mode]]
        offset  = (j - 1.5) * w
        bars = ax_top.bar(x + offset, sharpes, width=w,
                          color=colors[mode], alpha=0.8, label=mode)
    ax_top.axhline(0, color="black", lw=0.8, ls="--")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(periods, rotation=15, fontsize=9)
    ax_top.set_ylabel("Sharpe")
    ax_top.set_title("Exp29: 구간별 Sharpe (모드 A~D)")
    ax_top.legend(loc="upper right")
    ax_top.grid(axis="y", alpha=0.3)

    # 하단 좌: 평균 Sharpe 비교
    ax_bl = fig.add_subplot(gs[1, 0])
    avg_sharpes = [summary[m]["sharpe"] for m in modes]
    bar_colors  = [colors[m] for m in modes]
    bars = ax_bl.bar(modes, avg_sharpes, color=bar_colors, alpha=0.85)
    ax_bl.axhline(0, color="black", lw=0.8, ls="--")
    ax_bl.axhline(1.391, color="gray", lw=1.0, ls=":", label="Exp28-D 기준 (+1.391)")
    for bar, v in zip(bars, avg_sharpes):
        ax_bl.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f"{v:+.3f}", ha="center", va="bottom", fontsize=10)
    ax_bl.set_ylabel("평균 WF Sharpe")
    ax_bl.set_title("모드별 평균 Sharpe")
    ax_bl.legend(fontsize=8)
    ax_bl.grid(axis="y", alpha=0.3)

    # 하단 우: 거래수 vs Sharpe 산점도
    ax_br = fig.add_subplot(gs[1, 1])
    for mode in modes:
        ax_br.scatter(summary[mode]["trades"],
                      summary[mode]["sharpe"],
                      color=colors[mode], marker=markers[mode],
                      s=120, label=mode, zorder=5)
        ax_br.annotate(f"{mode} ({summary[mode]['trades']}건)",
                       (summary[mode]["trades"], summary[mode]["sharpe"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax_br.axhline(1.391, color="gray", lw=1.0, ls=":", label="Exp28-D 기준")
    ax_br.axhline(0, color="black", lw=0.6, ls="--")
    ax_br.set_xlabel("총 거래수")
    ax_br.set_ylabel("평균 WF Sharpe")
    ax_br.set_title("거래수 vs Sharpe (트레이드오프)")
    ax_br.legend(fontsize=8)
    ax_br.grid(alpha=0.3)

    plt.suptitle("Exp29: Funding Rate 역발상 신호 Walk-forward (2022~2024)",
                 fontsize=13, y=1.01)

    out = RESULTS_DIR / "exp29_funding_wf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp29: Funding Rate 역발상 신호 Walk-forward")
    print(f"  코인: BTC, SOL, AVAX, ADA, DOT")
    print(f"  기간: {START} ~ {END}")
    print(f"  기반: Exp28-D (RSI<30 + MPE<10% + vol>=50% + MA200 + 온체인<40%)")
    print(f"  청산: TP+{TP_PCT:.1f}% OR RSI>50 OR 168H (모든 모드 동일)")
    print(f"  펀딩 기준: BTC funding < {FUNDING_THRESH*100:.3f}%/8h, "
          f"정산 후 {FUNDING_WAIT_H}H 대기")
    print(f"  모드 A: 기준 (Exp28-D)")
    print(f"  모드 B: A + 펀딩 필터 추가")
    print(f"  모드 C: 펀딩 극단 시 MPE<15%ile 완화")
    print(f"  모드 D: 펀딩 극단 시 MPE<15% + vol>=30% + RSI<35 완화")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding_raw = collect_funding_rate("BTCUSDT", START, END)  # raw rate (8H)
    fear_greed  = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE 계산 (캐시 사용)...]")
    coin_data = {}
    for sym in COINS:
        df = collect(sym, "1h", START, END)
        price_mpe = rolling_mpe(df["close"].values,
                                m=MPE_M, scales=MPE_SCALES, window=MPE_WINDOW,
                                cache_key=f"{sym}_1h_{START}_{END}")
        price_mpe = pd.Series(price_mpe, index=df.index)
        vol_mpe   = rolling_mpe(df["volume"].values,
                                m=MPE_M, scales=MPE_SCALES, window=MPE_WINDOW,
                                cache_key=f"{sym}_1h_volume_{START}_{END}")
        vol_mpe   = pd.Series(vol_mpe, index=df.index)
        h_fund    = funding_entropy(funding_raw, df.index)
        h_fg      = fear_greed_entropy(fear_greed, df.index)
        h_onchain = combined_onchain_entropy(h_fund, h_fg)
        coin_data[sym] = {
            "df": df, "price_mpe": price_mpe,
            "vol_mpe": vol_mpe, "h_onchain": h_onchain,
        }
        print(f"  [{LABEL[sym]:6s}] {len(df):,}봉 로드 완료")

    print("\n[Walk-forward 실행 중 (4개 모드 × 5코인)...]")
    all_results = {}
    for sym in COINS:
        d = coin_data[sym]
        all_results[sym] = run_coin_wf_all(
            sym, d["df"], d["price_mpe"],
            d["vol_mpe"], d["h_onchain"],
            funding_raw
        )

    combined = aggregate_portfolio(all_results)
    summary  = print_portfolio_results(combined)

    print("\n[시각화 생성...]")
    plot_results(combined, summary)

    print("\nExp29 완료!")


if __name__ == "__main__":
    main()
