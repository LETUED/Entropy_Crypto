"""
실험 30: 거래비용·슬리피지 민감도 + 표본 견고성 정면 검증

배경: 감사(Exp01~29 다관점)에서 드러난 치명 이슈 3건을 한 번에 해결.
  (1) 희소곡선 Sharpe 왜곡  → 거래 단위 통계(per-trade)로 측정 교체
  (2) 다중가설/소표본       → 블록 부트스트랩 95% CI + leave-one-period-out
  (3) 거래비용 미검증        → 비용(수수료+슬리피지) 스윕으로 우위 소멸점 도출

※ 대상은 1H MPE 스윙 전략(Exp10/23-D/28-D). HFT 틱 타커 트랙(run_hft_taker.py,
  이미 '구조적 사망' 확정)과는 별개.

핵심 설계 (look-ahead 없음, 비용 해석적 분리):
  - WF(train 12M→test 6M, 6구간, 5코인)를 RAW 가격으로 1회 실행 → gross 거래 기록 수집.
    gross PnL은 비용과 독립이므로, 비용 c는 사후에 해석적으로 적용(정확·고속).
  - 비용 적용: 진입 P_entry*(1+c), 청산 시 (1-c). 왕복 ~2c.
  - S2(TP+2%)의 청산 판정은 RAW 가격 이동 기준 → 거래 집합이 비용과 무관해져 스윕이 정확.
    (원본 exp28은 수수료 포함가로 TP 판정했으나 2% 임계에서 0.1% 차이는 무시 가능, 명시함.)

비교 전략 (진입 우주가 다름):
  S0 (베이스라인, ~Exp10): MPE<10% AND RSI<30 AND MA200 AND 온체인<40%       / 청산 RSI>50 OR 168H
  S1 (Exp23-D):           S0 + 볼륨MPE>=50%                                  / 청산 RSI>50 OR 168H
  S2 (Exp28-D):           S1 진입                                            / 청산 TP+2% OR RSI>50 OR 168H

측정 지표 (거래 단위, 풀링):
  - 평균 순수익/거래(bps), t값, 승률
  - 거래단위 연율 Sharpe = (mean/std)*sqrt(거래수/년)
  - 블록 부트스트랩(코인×구간 블록, B=5000, seed=42) 95% CI
  - leave-one-period-out 민감도(특정 구간 제거 시 Sharpe)

실행: py exp30_cost_robustness_wf.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-", "") not in ("utf8", "utf-8"):
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
from src.analysis.h4_backtest import MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
         "ADAUSDT": "ADA", "DOTUSDT": "DOT"}

MPE_WINDOW = 168
MPE_M      = 3
MPE_SCALES = [1, 2, 4, 8]

ENTROPY_PCT  = 10     # 진입 MPE 하위 10%ile
VOL_PCT      = 50     # 볼륨 MPE >= 50%ile
RSI_OVERSOLD = 30
TP_PCT       = 2.0    # S2 목표 수익률(%)

# 비용 스윕: 한쪽(one-way) 비용 = 수수료 + 슬리피지. 왕복 ~2배.
#   0.0010 = 현재 가정(수수료 0.1%, 슬리피지 0)
#   0.0030 = 수수료 0.1% + 슬리피지 ~0.2% (시장가 taker 현실)
COST_GRID = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030]
BASE_COST = 0.0010

STRATEGIES = ["S0", "S1", "S2"]
STRAT_DESC = {
    "S0": "베이스라인(Exp10): RSI>50/168H 청산",
    "S1": "Exp23-D: +볼륨MPE>=50%",
    "S2": "Exp28-D: +TP2% 청산",
}

BOOT_B    = 5000
BOOT_SEED = 42
YEARS     = 4.0


def _setup_font():
    try:
        cands = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if cands:
            plt.rcParams["font.family"] = fm.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우: RAW 거래 수집 (비용 미적용) ──────────────────────────────────
def collect_trades_window(strategy, df_test, p_test, v_test, h_test,
                          p_train, v_train, oc_threshold,
                          k1, k5, k10):
    """
    진입/청산을 RAW 가격으로 시뮬레이션해 거래 리스트 반환.
    각 거래: dict(p_entry, p_exit, kfrac, held_h, gross_pct)
      gross_pct = (p_exit/p_entry - 1)*100  (비용·Kelly 미반영, 사후 적용)
    """
    rsi   = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    price_thr = np.percentile(p_train, ENTROPY_PCT)
    vt        = v_train.dropna()
    vol_thr   = np.percentile(vt, VOL_PCT) if len(vt) > 10 else np.nan

    use_vol = (strategy in ("S1", "S2"))
    use_tp  = (strategy == "S2")

    def _kelly(v):
        if v <= k1:  return 0.5
        if v <= k5:  return 0.3
        if v <= k10: return 0.15
        return 0.0

    position = 0
    p_entry = entry_i = 0
    kfrac = 0.0
    trades = []

    closes = df_test["close"]
    for i, idx in enumerate(df_test.index):
        price   = closes.loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe   = p_test.loc[idx] if idx in p_test.index else np.nan
        v_mpe   = v_test.loc[idx] if idx in v_test.index else np.nan

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        vol_ok = True
        if use_vol:
            vol_ok = (not np.isnan(v_mpe) and not np.isnan(vol_thr) and v_mpe >= vol_thr)

        # 청산 (RAW 기준)
        if position == 1:
            held = i - entry_i
            gross_pct = (price - p_entry) / p_entry * 100.0
            if use_tp:
                exit_ok = (gross_pct >= TP_PCT or rsi_val > 50 or held >= MAX_HOLD_H)
            else:
                exit_ok = (rsi_val > 50 or held >= MAX_HOLD_H)
            if exit_ok:
                trades.append({"p_entry": p_entry, "p_exit": price,
                               "kfrac": kfrac, "held_h": held,
                               "gross_pct": gross_pct})
                position = 0

        # 진입
        if (position == 0
                and rsi_val < RSI_OVERSOLD
                and not np.isnan(p_mpe)
                and not np.isnan(ma_val)
                and p_mpe <= price_thr
                and price > ma_val
                and oc_ok
                and vol_ok):
            kf = _kelly(p_mpe)
            if kf > 0:
                position = 1
                p_entry  = price
                entry_i  = i
                kfrac    = kf

    # 미청산 포지션은 마지막 가격으로 청산
    if position == 1:
        price = closes.iloc[-1]
        trades.append({"p_entry": p_entry, "p_exit": price,
                       "kfrac": kfrac, "held_h": len(df_test) - entry_i,
                       "gross_pct": (price - p_entry) / p_entry * 100.0})
    return trades


# ── 코인별 WF: 모든 전략 거래 수집 ────────────────────────────────────────────
def run_coin_wf(sym, df, price_mpe, vol_mpe, h_onchain):
    out = {s: [] for s in STRATEGIES}

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

        p_train = price_mpe[train_start:train_end].dropna()
        if len(p_train) < 200:
            cursor += test_delta
            continue

        k1 = np.percentile(p_train, 1)
        k5 = np.percentile(p_train, 5)
        k10 = np.percentile(p_train, 10)
        v_train = vol_mpe[train_start:train_end]

        oc_threshold = None
        if h_onchain is not None:
            hoc = h_onchain[train_start:train_end].dropna()
            if len(hoc) > 0:
                oc_threshold = np.percentile(hoc, 40)

        df_test = df[test_start:test_end]
        p_test  = price_mpe[test_start:test_end]
        v_test  = vol_mpe[test_start:test_end]
        h_test  = h_onchain[test_start:test_end] if h_onchain is not None else None
        if len(df_test) < 168:
            cursor += test_delta
            continue

        period = test_start.strftime("%Y-%m")
        for s in STRATEGIES:
            tr = collect_trades_window(s, df_test, p_test, v_test, h_test,
                                       p_train, v_train, oc_threshold, k1, k5, k10)
            for t in tr:
                t["coin"]   = LABEL[sym]
                t["period"] = period
                t["block"]  = f"{LABEL[sym]}|{period}"
            out[s].extend(tr)
        cursor += test_delta
    return out


# ── 비용 적용 + 거래 단위 통계 ────────────────────────────────────────────────
def net_return(t, c):
    """Kelly·비용 반영 거래당 순수익(자기자본 대비, 소수)."""
    g = t["p_exit"] / (t["p_entry"] * (1.0 + c)) - 1.0
    mult = (1.0 + t["kfrac"] * g) * (1.0 - c)
    return mult - 1.0


def pooled_stats(trades, c):
    if not trades:
        return None
    r = np.array([net_return(t, c) for t in trades])
    n = len(r)
    mean = r.mean()
    std  = r.std(ddof=1) if n > 1 else 0.0
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    winrate = float((r > 0).mean())
    tpy = n / YEARS
    sharpe = (mean / std) * np.sqrt(tpy) if std > 0 else float("nan")
    return {"n": n, "mean_bps": mean * 1e4, "t": t_stat,
            "winrate": winrate, "sharpe": sharpe}


def block_bootstrap(trades, c, B=BOOT_B, seed=BOOT_SEED):
    """코인×구간 블록 리샘플 → 평균 순수익/Sharpe의 95% CI."""
    if len(trades) < 2:
        return None
    blocks = {}
    for t in trades:
        blocks.setdefault(t["block"], []).append(t)
    keys = list(blocks.keys())
    rng = np.random.default_rng(seed)

    means, sharpes = [], []
    for _ in range(B):
        pick = rng.integers(0, len(keys), size=len(keys))
        pooled = []
        for j in pick:
            pooled.extend(blocks[keys[j]])
        r = np.array([net_return(t, c) for t in pooled])
        if len(r) < 2:
            continue
        m = r.mean()
        sd = r.std(ddof=1)
        means.append(m * 1e4)
        if sd > 0:
            sharpes.append((m / sd) * np.sqrt(len(r) / YEARS))
    if not means:
        return None
    return {
        "mean_lo": np.percentile(means, 2.5),
        "mean_hi": np.percentile(means, 97.5),
        "sharpe_lo": np.percentile(sharpes, 2.5) if sharpes else float("nan"),
        "sharpe_hi": np.percentile(sharpes, 97.5) if sharpes else float("nan"),
    }


def leave_one_period_out(trades, c):
    periods = sorted(set(t["period"] for t in trades))
    rows = []
    full = pooled_stats(trades, c)
    for p in periods:
        sub = [t for t in trades if t["period"] != p]
        st = pooled_stats(sub, c)
        rows.append({"dropped": p, "sharpe": st["sharpe"], "n": st["n"],
                     "delta": st["sharpe"] - full["sharpe"]})
    return full, rows


# ── 출력 ──────────────────────────────────────────────────────────────────────
def print_report(all_trades):
    print("\n" + "=" * 78)
    print("  거래 단위 풀링 통계 (look-ahead 없는 WF OOS 거래)")
    print("=" * 78)
    for s in STRATEGIES:
        n = len(all_trades[s])
        warn = "  ⚠ 표본<30" if n < 30 else ""
        print(f"\n  [{s}] {STRAT_DESC[s]}  (총 {n}거래{warn})")

    print("\n" + "=" * 78)
    print("  비용 스윕: 거래단위 연율 Sharpe (왕복 ~2×one-way)")
    print("=" * 78)
    header = "  one-way비용 |" + "".join(f"{s:>10}" for s in STRATEGIES)
    print(header)
    print("  " + "-" * 74)
    sharpe_table = {s: [] for s in STRATEGIES}
    for c in COST_GRID:
        line = f"  {c*100:>6.2f}%     |"
        for s in STRATEGIES:
            st = pooled_stats(all_trades[s], c)
            sharpe_table[s].append(st["sharpe"] if st else float("nan"))
            line += f"{(st['sharpe'] if st else float('nan')):>10.3f}"
        print(line)
    print("  " + "-" * 74)
    print("  (참고: 이 Sharpe는 거래단위 연율값 — 기존 시간당-곡선 +1.391과 직접 비교 불가)")

    print("\n" + "=" * 78)
    print("  평균 순수익/거래(bps) + t값 + 승률  @ one-way 비용별")
    print("=" * 78)
    for s in STRATEGIES:
        print(f"\n  [{s}] {STRAT_DESC[s]}")
        print(f"  {'비용':>8} {'평균bps':>10} {'t값':>8} {'승률':>8} {'거래수':>7}")
        print("  " + "-" * 50)
        for c in COST_GRID:
            st = pooled_stats(all_trades[s], c)
            if st:
                sig = " *" if (not np.isnan(st["t"]) and abs(st["t"]) > 1.96) else ""
                print(f"  {c*100:>7.2f}% {st['mean_bps']:>10.1f} "
                      f"{st['t']:>8.2f} {st['winrate']:>7.1%} {st['n']:>7}{sig}")

    # 부트스트랩 CI (기준 비용)
    print("\n" + "=" * 78)
    print(f"  블록 부트스트랩 95% CI  @ one-way 비용 {BASE_COST*100:.2f}% (B={BOOT_B}, seed={BOOT_SEED})")
    print("=" * 78)
    print(f"  {'전략':>5} {'평균bps':>10} {'평균CI':>22} {'Sharpe':>9} {'Sharpe CI':>22}")
    print("  " + "-" * 72)
    for s in STRATEGIES:
        st = pooled_stats(all_trades[s], BASE_COST)
        ci = block_bootstrap(all_trades[s], BASE_COST)
        if st and ci:
            mean_ci = f"[{ci['mean_lo']:+.1f}, {ci['mean_hi']:+.1f}]"
            shp_ci  = f"[{ci['sharpe_lo']:+.2f}, {ci['sharpe_hi']:+.2f}]"
            flag = "  CI하한>0 ✅" if ci["mean_lo"] > 0 else "  CI 0포함 ⚠"
            print(f"  {s:>5} {st['mean_bps']:>10.1f} {mean_ci:>22} "
                  f"{st['sharpe']:>9.3f} {shp_ci:>22}{flag}")

    # leave-one-period-out (S2 기준)
    print("\n" + "=" * 78)
    print(f"  Leave-one-period-out 민감도: S2(Exp28-D) @ 비용 {BASE_COST*100:.2f}%")
    print("=" * 78)
    full, rows = leave_one_period_out(all_trades["S2"], BASE_COST)
    print(f"  전체: Sharpe {full['sharpe']:+.3f}, {full['n']}거래")
    print(f"  {'제거구간':>12} {'남은거래':>8} {'Sharpe':>9} {'변화':>9}")
    print("  " + "-" * 44)
    for r in sorted(rows, key=lambda x: x["delta"]):
        mark = "  ← 핵심의존" if r["delta"] < -0.3 else ("  ← 손실원" if r["delta"] > 0.3 else "")
        print(f"  {r['dropped']:>12} {r['n']:>8} {r['sharpe']:>9.3f} {r['delta']:>+9.3f}{mark}")

    return sharpe_table


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_report(all_trades, sharpe_table):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26)
    colors = {"S0": "#888888", "S1": "#4878cf", "S2": "#ee854a"}
    xs = [c * 100 for c in COST_GRID]

    # (1) 비용 vs 거래단위 Sharpe
    ax1 = fig.add_subplot(gs[0, 0])
    for s in STRATEGIES:
        ax1.plot(xs, sharpe_table[s], "o-", color=colors[s], label=s, lw=2)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xlabel("one-way 비용 (%)")
    ax1.set_ylabel("거래단위 연율 Sharpe")
    ax1.set_title("비용 민감도: Sharpe 소멸점")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) 비용 vs 평균 순수익/거래(bps)
    ax2 = fig.add_subplot(gs[0, 1])
    for s in STRATEGIES:
        means = [pooled_stats(all_trades[s], c)["mean_bps"] for c in COST_GRID]
        ax2.plot(xs, means, "s-", color=colors[s], label=s, lw=2)
    ax2.axhline(0, color="red", lw=1.0, ls="--", label="손익분기")
    ax2.set_xlabel("one-way 비용 (%)")
    ax2.set_ylabel("평균 순수익/거래 (bps)")
    ax2.set_title("비용 vs 거래당 순엣지")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # (3) 부트스트랩 평균 CI (기준 비용)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, s in enumerate(STRATEGIES):
        st = pooled_stats(all_trades[s], BASE_COST)
        ci = block_bootstrap(all_trades[s], BASE_COST)
        if st and ci:
            ax3.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=colors[s], capsize=6, ms=9, lw=2)
    ax3.axhline(0, color="red", lw=1.0, ls="--")
    ax3.set_xticks(range(len(STRATEGIES)))
    ax3.set_xticklabels(STRATEGIES)
    ax3.set_ylabel("평균 순수익/거래 (bps)")
    ax3.set_title(f"블록 부트스트랩 95% CI @ 비용 {BASE_COST*100:.2f}%")
    ax3.grid(axis="y", alpha=0.3)

    # (4) leave-one-period-out (S2)
    ax4 = fig.add_subplot(gs[1, 1])
    full, rows = leave_one_period_out(all_trades["S2"], BASE_COST)
    labels = ["전체"] + [r["dropped"] for r in rows]
    vals   = [full["sharpe"]] + [r["sharpe"] for r in rows]
    bcolors = ["#2a2a2a"] + ["#d65f5f" if v < full["sharpe"] else "#6acc65" for v in vals[1:]]
    ax4.bar(range(len(labels)), vals, color=bcolors, alpha=0.85)
    ax4.axhline(full["sharpe"], color="gray", ls=":", lw=1.0, label=f"전체 {full['sharpe']:+.2f}")
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=30, fontsize=8, ha="right")
    ax4.set_ylabel("거래단위 연율 Sharpe")
    ax4.set_title("S2 구간 제거 민감도 (집중도)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp30: 거래비용·슬리피지 민감도 + 표본 견고성 (1H MPE 스윙 전략)",
                 fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp30_cost_robustness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("Exp30: 거래비용·슬리피지 민감도 + 표본 견고성 정면 검증")
    print(f"  코인: BTC, SOL, AVAX, ADA, DOT  |  기간: {START} ~ {END}")
    print(f"  비용 스윕(one-way): {[f'{c*100:.2f}%' for c in COST_GRID]}")
    print(f"  측정: 거래단위 통계 + 블록 부트스트랩 95% CI (시간당-곡선 Sharpe 폐기)")
    print("=" * 78)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE 계산 (캐시 사용)...]")
    coin_data = {}
    for sym in COINS:
        df = collect(sym, "1h", START, END)
        price_mpe = pd.Series(
            rolling_mpe(df["close"].values, m=MPE_M, scales=MPE_SCALES, window=MPE_WINDOW,
                        cache_key=f"{sym}_1h_{START}_{END}"), index=df.index)
        vol_mpe = pd.Series(
            rolling_mpe(df["volume"].values, m=MPE_M, scales=MPE_SCALES, window=MPE_WINDOW,
                        cache_key=f"{sym}_1h_volume_{START}_{END}"), index=df.index)
        h_fund = funding_entropy(funding, df.index)
        h_fg   = fear_greed_entropy(fg, df.index)
        h_oc   = combined_onchain_entropy(h_fund, h_fg)
        coin_data[sym] = (df, price_mpe, vol_mpe, h_oc)
        print(f"  [{LABEL[sym]:5s}] {len(df):,}봉")

    print("\n[Walk-forward RAW 거래 수집 (3전략 × 5코인)...]")
    all_trades = {s: [] for s in STRATEGIES}
    for sym in COINS:
        df, pm, vm, ho = coin_data[sym]
        per_coin = run_coin_wf(sym, df, pm, vm, ho)
        for s in STRATEGIES:
            all_trades[s].extend(per_coin[s])
        cnts = " ".join(f"{s}:{len(per_coin[s])}" for s in STRATEGIES)
        print(f"  [{LABEL[sym]:5s}] {cnts}")

    sharpe_table = print_report(all_trades)

    print("\n[시각화 생성...]")
    plot_report(all_trades, sharpe_table)

    print("\nExp30 완료!")


if __name__ == "__main__":
    main()
