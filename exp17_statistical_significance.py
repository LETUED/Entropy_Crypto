"""
실험 17: 통계적 유의성 검정 (최적화 버전)

질문: Sharpe +0.529가 실력인가, 운인가?

방법: 사전 계산 기반 Permutation Test
  1. 모든 "가능한 진입점" 에서 실제 PnL을 미리 계산 (한 번만)
  2. 실제 전략은 이 중 MPE+RSI+온체인 조건을 만족하는 N개를 선택
  3. 랜덤 기준선: 동일 우주에서 N개를 무작위 선택 → 10,000회
  4. 실제 평균 PnL이 랜덤 분포의 몇 %인지 → p-value

질문 본질: "MPE+RSI+온체인 조건이 실제로 더 좋은 거래를 골라내는가?"

실행: py exp17_statistical_significance.py
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
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import (
    compute_metrics, FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H
)

RESULTS_DIR = Path("results")
START, END   = "2021-01-01", "2025-01-01"
N_PERM       = 10_000
N_BOOT       = 5_000

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT":"BTC","SOLUSDT":"SOL","AVAXUSDT":"AVAX","ADAUSDT":"ADA","DOTUSDT":"DOT"}


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 전체 진입 가능 우주 사전 계산 ────────────────────────────────────────────
def precompute_trade_universe(df, rsi):
    """
    MA200 위에 있고 유효한 MPE가 있는 모든 봉에서:
    '만약 여기서 진입했다면 어떤 PnL이 났을까?' 를 미리 계산.

    청산 조건: RSI>50 OR MAX_HOLD_H
    반환: List[dict] — 가능한 모든 거래의 PnL 정보
    """
    ma200 = df["close"].rolling(MA_PERIOD).mean()
    idx_list = df.index.tolist()
    n = len(idx_list)
    prices = df["close"].values
    rsi_vals = rsi.reindex(df.index).values

    universe = []

    for i in range(MA_PERIOD, n - 1):
        # MA200 위에 있어야 함
        if np.isnan(ma200.iloc[i]) or prices[i] <= ma200.iloc[i]:
            continue

        entry_price = prices[i] * (1 + FEE_RATE)

        # 청산 시점 찾기
        exit_i = n - 1
        for j in range(i + 1, min(i + MAX_HOLD_H + 1, n)):
            if rsi_vals[j] > 50 or (j - i) >= MAX_HOLD_H:
                exit_i = j
                break

        exit_price = prices[exit_i]
        pnl = (exit_price - entry_price) / entry_price
        held = exit_i - i

        universe.append({
            "entry_i": i,
            "pnl":     pnl,
            "held_h":  held,
        })

    return universe


# ── 실제 전략 실행 (거래 리스트 반환) ─────────────────────────────────────────
def run_actual_strategy(df, mpe, h_onchain):
    rsi   = compute_rsi(df["close"])
    sig   = generate_signals(rsi)
    ma200 = df["close"].rolling(MA_PERIOD).mean()

    mpe_thresh = np.percentile(mpe.dropna(), ENTROPY_PCT)
    oc_thresh  = np.percentile(h_onchain.dropna(), 40) if h_onchain is not None else None

    k_pct1  = np.percentile(mpe.dropna(), 1)
    k_pct5  = np.percentile(mpe.dropna(), 5)
    k_pct10 = np.percentile(mpe.dropna(), 10)

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
    curve   = []

    idx_list = df.index.tolist()

    for i, idx in enumerate(idx_list):
        price   = df["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        mpe_val = mpe.loc[idx]   if idx in mpe.index   else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        oc_ok = True
        if h_onchain is not None and oc_thresh is not None and idx in h_onchain.index:
            oc_ok = h_onchain.loc[idx] <= oc_thresh

        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl": pnl, "kfrac": entry_kfrac, "held_h": held})
                position = 0

        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            is_low   = mpe_val <= mpe_thresh
            above_ma = price > ma_val
            k_frac   = _kelly(mpe_val)
            if sig_val == 1 and is_low and above_ma and k_frac > 0 and oc_ok:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
                entry_kfrac = k_frac

        curve.append(equity)

    if position == 1:
        pnl = (df["close"].iloc[-1] - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    eq = pd.Series(curve, index=idx_list)
    return eq, trades


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Exp17: 통계적 유의성 검정 (사전계산 최적화)")
    print(f"순열 검정: {N_PERM:,}회 / 부트스트랩: {N_BOOT:,}회")
    print("=" * 65)

    rng = np.random.default_rng(seed=42)

    print("\n[데이터 로드...]")
    funding  = collect_funding_rate("BTCUSDT", START, END)
    fg       = collect_fear_greed(START, END)

    coin_data = {}
    for sym in COINS:
        df  = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], window=168,
                          cache_key=f"{sym}_1h_{START}_{END}")
        h_f = funding_entropy(funding, df.index)
        h_g = fear_greed_entropy(fg, df.index)
        h_oc = combined_onchain_entropy(h_f, h_g)
        coin_data[sym] = {"df": df, "mpe": mpe, "h": h_oc}
        print(f"  [{LABEL[sym]:6s}] 로드 완료")

    # ── 1. 실제 전략 실행 ─────────────────────────────────────────────────
    print("\n[실제 전략 실행...]")
    all_actual_trades = []
    port_curves = []
    coin_sharpes = {}

    for sym in COINS:
        d = coin_data[sym]
        eq, trades = run_actual_strategy(d["df"], d["mpe"], d["h"])
        all_actual_trades.extend(trades)
        port_curves.append(eq)

        ret = eq.pct_change().dropna()
        s   = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(24 * 365))
        coin_sharpes[sym] = s
        print(f"  [{LABEL[sym]:6s}] Sharpe={s:+.3f}, 거래={len(trades)}")

    port_eq  = sum(eq / len(COINS) for eq in port_curves)
    port_ret = port_eq.pct_change().dropna()
    actual_sharpe = float(port_ret.mean() / (port_ret.std() + 1e-12) * np.sqrt(24 * 365))
    actual_mean_pnl = np.mean([t["pnl"] * 100 for t in all_actual_trades])

    print(f"\n  포트폴리오 Sharpe: {actual_sharpe:+.4f}")
    print(f"  총 거래: {len(all_actual_trades)}건  |  평균 PnL: {actual_mean_pnl:+.4f}%")

    # ── 2. 진입 가능 우주 사전 계산 ──────────────────────────────────────────
    print("\n[진입 가능 우주 사전 계산 (전략 조건 없이 MA200 위 전체)...]")
    all_universes = {}
    for sym in COINS:
        d   = coin_data[sym]
        rsi = compute_rsi(d["df"]["close"])
        universe = precompute_trade_universe(d["df"], rsi)
        all_universes[sym] = np.array([t["pnl"] for t in universe])
        print(f"  [{LABEL[sym]:6s}] 가능한 진입점: {len(universe):,}개")

    n_per_coin = len(all_actual_trades) // len(COINS)  # 코인당 평균 거래수

    # ── 3. Permutation Test (빠른 버전: PnL 샘플링) ──────────────────────
    print(f"\n[순열 검정 실행 중... {N_PERM:,}회 (빠른 PnL 샘플링)]")
    print(f"  각 코인에서 {n_per_coin}건씩 랜덤 선택")

    perm_mean_pnls = np.zeros(N_PERM)

    for k in range(N_PERM):
        pnls = []
        for sym in COINS:
            universe_pnls = all_universes[sym]
            n_sample = len([t for t in all_actual_trades]) // len(COINS)
            n_sample = max(1, min(n_sample, len(universe_pnls)))
            sampled  = rng.choice(universe_pnls, size=n_sample, replace=False)
            pnls.extend(sampled.tolist())
        perm_mean_pnls[k] = np.mean(pnls) * 100  # %

    p_value_pnl  = np.mean(perm_mean_pnls >= actual_mean_pnl)
    percentile   = np.mean(perm_mean_pnls < actual_mean_pnl) * 100

    print(f"\n  실제 평균 PnL:     {actual_mean_pnl:+.4f}%")
    print(f"  랜덤 평균 PnL:     {perm_mean_pnls.mean():+.4f}%")
    print(f"  랜덤 표준편차:     {perm_mean_pnls.std():.4f}%")
    print(f"  p-value (PnL):    {p_value_pnl:.4f}")
    print(f"  백분위:            {percentile:.1f}%")

    if p_value_pnl < 0.01:
        print("  [결론] p < 0.01 — 신호가 랜덤 대비 통계적으로 유의하게 좋은 진입점 선택")
    elif p_value_pnl < 0.05:
        print("  [결론] p < 0.05 — 5% 유의수준에서 통계적으로 유의")
    elif p_value_pnl < 0.10:
        print("  [결론] p < 0.10 — 10% 유의수준에서 경계선")
    else:
        print(f"  [결론] p = {p_value_pnl:.3f} — 통계적으로 유의하지 않음")

    # ── 4. Bootstrap 신뢰구간 (거래 PnL 재샘플링) ────────────────────────
    print(f"\n[부트스트랩 신뢰구간... {N_BOOT:,}회]")
    actual_pnls = np.array([t["pnl"] * 100 for t in all_actual_trades])
    n_trades = len(actual_pnls)
    avg_held = np.mean([t["held_h"] for t in all_actual_trades])
    trades_per_year = 8760 / max(avg_held, 1)

    boot_mean_pnls  = np.zeros(N_BOOT)
    boot_sharpes    = np.zeros(N_BOOT)

    for k in range(N_BOOT):
        sample = rng.choice(actual_pnls, size=n_trades, replace=True)
        boot_mean_pnls[k] = np.mean(sample)
        boot_sharpes[k]   = np.mean(sample) / (np.std(sample) + 1e-12) * np.sqrt(trades_per_year)

    ci_lo_pnl = np.percentile(boot_mean_pnls, 2.5)
    ci_hi_pnl = np.percentile(boot_mean_pnls, 97.5)
    ci_lo_s   = np.percentile(boot_sharpes, 2.5)
    ci_hi_s   = np.percentile(boot_sharpes, 97.5)

    print(f"\n  평균 PnL 95% CI: [{ci_lo_pnl:+.4f}%, {ci_hi_pnl:+.4f}%]  "
          f"{'(양수!)' if ci_lo_pnl > 0 else '(0 포함)'}")
    print(f"  Sharpe 95% CI:   [{ci_lo_s:+.3f}, {ci_hi_s:+.3f}]  "
          f"{'(양수!)' if ci_lo_s > 0 else '(0 포함)'}")

    # ── 5. 코인별 유의성 ──────────────────────────────────────────────────
    print(f"\n코인별 평균 PnL 비교")
    print(f"{'코인':>6} {'실제 PnL':>10} {'랜덤 PnL':>10} {'p-value':>10} {'결론':>10}")
    print("-" * 55)
    for sym in COINS:
        coin_actual = [t["pnl"] * 100 for t in all_actual_trades
                       if t.get("sym") == sym]
        # 코인별 집계 (전체 all_actual_trades에서 분리 불가 → 코인별 재실행)
        d   = coin_data[sym]
        _, coin_trades = run_actual_strategy(d["df"], d["mpe"], d["h"])
        coin_pnl = np.mean([t["pnl"] * 100 for t in coin_trades]) if coin_trades else 0.0
        universe_pnls = all_universes[sym]

        # 코인별 p-value
        n_sample = max(1, len(coin_trades))
        perm_pnls_coin = np.array([
            np.mean(rng.choice(universe_pnls, size=n_sample, replace=False)) * 100
            for _ in range(1000)
        ])
        p_coin = np.mean(perm_pnls_coin >= coin_pnl)

        mark = "(*)" if p_coin < 0.05 else "   "
        print(f"  {LABEL[sym]:>6} {coin_pnl:>+10.4f}% {perm_pnls_coin.mean():>+10.4f}% {p_coin:>10.4f} {mark}")

    # ── 6. 시각화 ─────────────────────────────────────────────────────────
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # 순열 검정 분포
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22"); ax1.spines[:].set_color("#30363d")
    ax1.tick_params(colors="#8b949e")

    ax1.hist(perm_mean_pnls, bins=80, color="#58a6ff", alpha=0.75,
             edgecolor="none", label=f"랜덤 진입 PnL 분포 ({N_PERM:,}회)")
    ax1.axvline(actual_mean_pnl, color="#f7931a", linewidth=2.5,
                label=f"실제 전략 PnL = {actual_mean_pnl:+.3f}%")
    ax1.axvline(np.percentile(perm_mean_pnls, 95), color="#3fb950",
                linewidth=1.5, linestyle="--",
                label=f"95th pct = {np.percentile(perm_mean_pnls, 95):+.3f}%")
    ax1.axvline(np.percentile(perm_mean_pnls, 99), color="#ff7b72",
                linewidth=1.5, linestyle=":",
                label=f"99th pct = {np.percentile(perm_mean_pnls, 99):+.3f}%")
    ax1.axvline(0, color="#8b949e", linewidth=0.8)

    ax1.set_xlabel("평균 PnL (%)", color="#8b949e")
    ax1.set_ylabel("빈도", color="#8b949e")
    ax1.set_title(
        f"Permutation Test: 신호 vs 랜덤\np-value = {p_value_pnl:.4f}  |  {percentile:.1f}th 백분위",
        color="#e6edf3", fontsize=11
    )
    ax1.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # 부트스트랩 CI
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22"); ax2.spines[:].set_color("#30363d")
    ax2.tick_params(colors="#8b949e")

    ax2.hist(boot_mean_pnls, bins=80, color="#d2a8ff", alpha=0.75,
             edgecolor="none", label=f"Bootstrap PnL ({N_BOOT:,}회)")
    ax2.axvline(np.mean(actual_pnls), color="#f7931a", linewidth=2.0,
                label=f"실제 평균 = {np.mean(actual_pnls):+.3f}%")
    ax2.axvline(ci_lo_pnl, color="#3fb950", linewidth=1.5, linestyle="--",
                label=f"95% CI = [{ci_lo_pnl:+.3f}%, {ci_hi_pnl:+.3f}%]")
    ax2.axvline(ci_hi_pnl, color="#3fb950", linewidth=1.5, linestyle="--")
    ax2.axvline(0, color="#ff7b72", linewidth=1.2, linestyle=":",
                label="PnL = 0 기준선")

    ax2.set_xlabel("평균 PnL (%)", color="#8b949e")
    ax2.set_ylabel("빈도", color="#8b949e")
    ax2.set_title(
        f"Bootstrap 신뢰구간\n95% CI = [{ci_lo_pnl:+.3f}%, {ci_hi_pnl:+.3f}%]",
        color="#e6edf3", fontsize=11
    )
    ax2.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    fig.suptitle(
        f"Exp17: 통계적 유의성  |  실제 PnL {actual_mean_pnl:+.3f}%  |  "
        f"p={p_value_pnl:.4f}  |  {len(all_actual_trades)}건",
        color="#e6edf3", fontsize=12, fontweight="bold"
    )

    path = RESULTS_DIR / "exp17_statistical_significance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장: {path}")
    plt.show()

    # ── 최종 요약 ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("최종 통계적 유의성 요약")
    print("=" * 65)
    print(f"  포트폴리오 Sharpe:    {actual_sharpe:+.4f}")
    print(f"  실제 평균 PnL:        {actual_mean_pnl:+.4f}%")
    print(f"  랜덤 평균 PnL:        {perm_mean_pnls.mean():+.4f}%")
    print(f"  p-value (순열):       {p_value_pnl:.4f}  {'(유의)' if p_value_pnl < 0.05 else '(비유의)'}")
    print(f"  PnL 95% CI:           [{ci_lo_pnl:+.3f}%, {ci_hi_pnl:+.3f}%]"
          f"  {'(양수 확정)' if ci_lo_pnl > 0 else '(0 포함)'}")
    print(f"  총 거래수:            {len(all_actual_trades)}건")
    print()

    if p_value_pnl < 0.05 and ci_lo_pnl > 0:
        print("  [최종] 통계적으로 유의 + 부트스트랩 양수 — 실력 기반 엣지 확인")
    elif p_value_pnl < 0.05:
        print("  [최종] 순열 검정 통과, 부트스트랩 CI 0 포함 — 엣지 존재하나 불확실")
    elif ci_lo_pnl > 0:
        print("  [최종] CI 양수, 순열 미통과 — 더 많은 데이터 필요")
    else:
        print("  [최종] 통계적 증거 불충분 — 더 많은 거래 또는 긴 기간 필요")


if __name__ == "__main__":
    main()
