"""
실험 15: BTC IS Sharpe 불일치 원인 규명

Exp01: BTC IS Sharpe +0.436  (run_strategy  in h4_backtest.py)
Exp14: BTC IS Sharpe -0.626  (run_coin_period in exp14_category_oos.py)
동일 기간(2021-2025), 동일 코인 — 왜 다른가?

진단 방법:
  A) 두 구현을 같은 데이터로 실행 → 결과 비교
  B) 임계값(mpe_thresh, oc_thresh) 비교
  C) 개별 거래 리스트 비교
  D) 수수료 구조 차이 정량화
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-","") not in ("utf8","utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import (
    FEE_RATE, MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H,
    kelly_size, compute_metrics, run_strategy
)

START  = "2021-01-01"
IS_END = "2025-01-01"
SYM    = "BTCUSDT"

print("=" * 65)
print("Exp15: BTC IS Sharpe 불일치 원인 규명")
print(f"기간: {START} ~ {IS_END}")
print("=" * 65)


# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
print("\n[1] 데이터 로드 중...")

# IS 전용 데이터 (Exp01 조건)
df_is      = collect(SYM, "1h", START, IS_END)
funding_is = collect_funding_rate(SYM, START, IS_END)
fg_is      = collect_fear_greed(START, IS_END)

# 전체 범위 데이터 (Exp14 조건) - 캐시에서 로드 후 슬라이싱
df_full      = collect(SYM, "1h", START, "2026-05-26")
funding_full = collect_funding_rate(SYM, START, "2026-05-26")
fg_full      = collect_fear_greed(START, "2026-05-26")

print(f"  IS 데이터:   {len(df_is):,}행 ({df_is.index[0].date()} ~ {df_is.index[-1].date()})")
print(f"  Full 데이터: {len(df_full):,}행 ({df_full.index[0].date()} ~ {df_full.index[-1].date()})")


# ── 2. MPE 계산 ─────────────────────────────────────────────────────────────
print("\n[2] MPE 계산 중...")

mpe_is   = rolling_mpe(df_is["close"],   window=168, cache_key=f"BTCUSDT_1h_{START}_{IS_END}")
mpe_full = rolling_mpe(df_full["close"], window=168, cache_key=f"BTCUSDT_1h_{START}_2026-05-26")
mpe_full_is = mpe_full[START:IS_END]

# MPE 값 비교
common_idx = mpe_is.dropna().index.intersection(mpe_full_is.dropna().index)
if len(common_idx) > 0:
    diff = (mpe_is[common_idx] - mpe_full_is[common_idx]).abs()
    print(f"  MPE 값 차이 (IS vs Full-sliced): max={diff.max():.6f}, mean={diff.mean():.6f}")
    if diff.max() < 1e-9:
        print("  -> MPE 값 동일 (데이터 일치)")
    else:
        print("  -> MPE 값 상이! (데이터 불일치 가능)")


# ── 3. 온체인 엔트로피 계산 ─────────────────────────────────────────────────
print("\n[3] 온체인 엔트로피 계산...")

h_fund_is = funding_entropy(funding_is, df_is.index)
h_fg_is   = fear_greed_entropy(fg_is, df_is.index)
h_onchain_is = combined_onchain_entropy(h_fund_is, h_fg_is)

h_fund_full = funding_entropy(funding_full, df_full.index)
h_fg_full   = fear_greed_entropy(fg_full, df_full.index)
h_onchain_full = combined_onchain_entropy(h_fund_full, h_fg_full)
h_onchain_full_is = h_onchain_full[START:IS_END]

# 온체인 엔트로피 비교
common_oc = h_onchain_is.dropna().index.intersection(h_onchain_full_is.dropna().index)
if len(common_oc) > 0:
    diff_oc = (h_onchain_is[common_oc] - h_onchain_full_is[common_oc]).abs()
    print(f"  H_onchain 값 차이: max={diff_oc.max():.6f}, mean={diff_oc.mean():.6f}")
    if diff_oc.max() < 1e-9:
        print("  -> H_onchain 값 동일")
    else:
        print("  -> H_onchain 값 상이!")

# 임계값 비교
oc_thresh_is = np.percentile(h_onchain_is.dropna(), 40)
oc_thresh_full_is = np.percentile(h_onchain_full_is.dropna(), 40)
print(f"\n  OC 임계값 (IS-only 데이터):      {oc_thresh_is:.6f}")
print(f"  OC 임계값 (Full->IS 슬라이스):   {oc_thresh_full_is:.6f}")

mpe_thresh_is = np.percentile(mpe_is.dropna(), ENTROPY_PCT)
mpe_thresh_full_is = np.percentile(mpe_full_is.dropna(), ENTROPY_PCT)
print(f"  MPE 임계값 (IS-only 데이터):     {mpe_thresh_is:.6f}")
print(f"  MPE 임계값 (Full->IS 슬라이스):  {mpe_thresh_full_is:.6f}")


# ── 4. 구현 A: run_strategy() — Exp01 방식 ─────────────────────────────────
print("\n" + "=" * 65)
print("[4] 구현 A: run_strategy() — Exp01 방식")
print("    (수수료: 진입가격 슬리피지 + 청산시 equity×(1-fee))")

eq_A = run_strategy(
    df_is, mpe_is,
    use_entropy_filter=True, use_trend_filter=True,
    use_kelly=True, allow_short=False,
    h_onchain=h_onchain_is,
    label="Exp01-style"
)
metrics_A = compute_metrics(eq_A)
print(f"  Sharpe:    {metrics_A['Sharpe']}")
print(f"  총수익률:  {metrics_A['총 수익률']}")
print(f"  최대낙폭:  {metrics_A['최대 낙폭']}")


# ── 5. 구현 B: run_coin_period() — Exp14 방식 (버그 포함) ─────────────────
print("\n" + "=" * 65)
print("[5] 구현 B: run_coin_period() — Exp14 방식 (수수료 이중 청구)")
print("    (수수료: 진입 equity×(1-fee) + 진입가격 슬리피지 + 청산 equity×(1-fee))")

def run_exp14_style(df, mpe, h_onchain, mpe_thresh=None, oc_thresh=None):
    """Exp14의 run_coin_period 로직 그대로"""
    rsi   = compute_rsi(df["close"])
    ma200 = df["close"].rolling(MA_PERIOD).mean()
    sig   = generate_signals(rsi)

    if mpe_thresh is None:
        valid = mpe.dropna()
        mpe_thresh = np.percentile(valid, ENTROPY_PCT)

    if oc_thresh is None and h_onchain is not None:
        valid_oc = h_onchain.dropna()
        oc_thresh = np.percentile(valid_oc, 40) if len(valid_oc) > 10 else None

    equity   = 1.0
    position = 0
    entry_price = 0.0
    entry_hour  = 0
    entry_kfrac = 0.0
    trades  = []
    equity_curve = []

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

        # 청산
        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({
                    "entry_idx": entry_hour,
                    "exit_idx":  i,
                    "entry_dt":  idx_list[entry_hour],
                    "exit_dt":   idx,
                    "pnl_pct":   pnl * 100,
                    "kfrac":     entry_kfrac,
                    "held_h":    held,
                })
                position = 0

        # 진입
        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            is_low   = mpe_val <= mpe_thresh
            above_ma = price > ma_val
            k_frac   = kelly_size(mpe_val, mpe)
            if sig_val == 1 and is_low and above_ma and k_frac > 0 and oc_ok:
                equity *= (1 - FEE_RATE)   # <-- 수수료 이중 청구 (Exp14 버그)
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
                entry_kfrac = k_frac

        equity_curve.append(equity)

    if position == 1 and len(idx_list) > 0:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    eq_series = pd.Series(equity_curve, index=idx_list)
    return eq_series, trades, mpe_thresh, oc_thresh


eq_B, trades_B, mt_B, ot_B = run_exp14_style(df_is, mpe_is, h_onchain_is)
metrics_B = compute_metrics(eq_B)
print(f"  Sharpe:    {metrics_B['Sharpe']}")
print(f"  총수익률:  {metrics_B['총 수익률']}")
print(f"  최대낙폭:  {metrics_B['최대 낙폭']}")
print(f"  거래수:    {len(trades_B)}")


# ── 6. 구현 C: 수수료 이중청구 수정 ────────────────────────────────────────
print("\n" + "=" * 65)
print("[6] 구현 C: 수수료 이중청구 수정 (진입 equity 감소 제거)")

def run_fixed_style(df, mpe, h_onchain, mpe_thresh=None, oc_thresh=None):
    """수수료 이중청구 제거한 버전"""
    rsi   = compute_rsi(df["close"])
    ma200 = df["close"].rolling(MA_PERIOD).mean()
    sig   = generate_signals(rsi)

    if mpe_thresh is None:
        valid = mpe.dropna()
        mpe_thresh = np.percentile(valid, ENTROPY_PCT)

    if oc_thresh is None and h_onchain is not None:
        valid_oc = h_onchain.dropna()
        oc_thresh = np.percentile(valid_oc, 40) if len(valid_oc) > 10 else None

    equity   = 1.0
    position = 0
    entry_price = 0.0
    entry_hour  = 0
    entry_kfrac = 0.0
    trades  = []
    equity_curve = []

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
                trades.append({
                    "entry_dt": idx_list[entry_hour],
                    "exit_dt":  idx,
                    "pnl_pct":  pnl * 100,
                    "kfrac":    entry_kfrac,
                    "held_h":   held,
                })
                position = 0

        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            is_low   = mpe_val <= mpe_thresh
            above_ma = price > ma_val
            k_frac   = kelly_size(mpe_val, mpe)
            if sig_val == 1 and is_low and above_ma and k_frac > 0 and oc_ok:
                # 수수료 이중청구 제거 — 진입가격 슬리피지만 사용
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
                entry_kfrac = k_frac

        equity_curve.append(equity)

    if position == 1 and len(idx_list) > 0:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    eq_series = pd.Series(equity_curve, index=idx_list)
    return eq_series, trades, mpe_thresh, oc_thresh


eq_C, trades_C, mt_C, ot_C = run_fixed_style(df_is, mpe_is, h_onchain_is)
metrics_C = compute_metrics(eq_C)
print(f"  Sharpe:    {metrics_C['Sharpe']}")
print(f"  총수익률:  {metrics_C['총 수익률']}")
print(f"  최대낙폭:  {metrics_C['최대 낙폭']}")
print(f"  거래수:    {len(trades_C)}")


# ── 7. 거래 리스트 비교 ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("[7] 구현 B 거래 리스트 (Exp14 방식)")
print(f"{'#':>3} {'진입일':>12} {'청산일':>12} {'보유(H)':>8} {'PnL%':>8} {'Kelly':>6}")
print("-" * 55)
for j, t in enumerate(trades_B, 1):
    ed = str(t['entry_dt'])[:10]
    xd = str(t['exit_dt'])[:10]
    print(f"{j:>3} {ed:>12} {xd:>12} {t['held_h']:>8} {t['pnl_pct']:>8.3f} {t['kfrac']:>6.2f}")

print(f"\n구현 A (Exp01 방식) 거래수: 결과에서 역산 불가 — run_strategy 내부")
print(f"구현 B (Exp14 방식) 거래수: {len(trades_B)}")
print(f"구현 C (수수료 수정) 거래수: {len(trades_C)}")


# ── 8. 최종 비교표 ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("최종 비교 요약")
print("=" * 65)
print(f"{'구현':30} {'Sharpe':>8} {'총수익률':>10} {'거래수':>6}")
print("-" * 60)
print(f"{'A. run_strategy() [Exp01]':30} {float(metrics_A['Sharpe']):>8.3f} {metrics_A['총 수익률']:>10} {'?':>6}")
print(f"{'B. run_coin_period() [Exp14]':30} {float(metrics_B['Sharpe']):>8.3f} {metrics_B['총 수익률']:>10} {len(trades_B):>6}")
print(f"{'C. 수수료 수정 버전':30} {float(metrics_C['Sharpe']):>8.3f} {metrics_C['총 수익률']:>10} {len(trades_C):>6}")
print()

sharpe_A = float(metrics_A['Sharpe'])
sharpe_B = float(metrics_B['Sharpe'])
sharpe_C = float(metrics_C['Sharpe'])

print(f"A vs B 차이: {sharpe_A - sharpe_B:+.3f} Sharpe")
print(f"A vs C 차이: {sharpe_A - sharpe_C:+.3f} Sharpe")
print(f"B vs C 차이 (수수료 이중청구 효과): {sharpe_B - sharpe_C:+.3f} Sharpe")
print()
print(f"MPE 임계값: {mpe_thresh_is:.6f}")
print(f"OC  임계값: {oc_thresh_is:.6f}")
print()

if abs(sharpe_A - sharpe_C) < 0.05:
    print("[결론] 수수료 수정만으로 A=C — Exp14의 수수료 이중청구가 유일한 원인")
elif abs(sharpe_A - sharpe_B) > 0.5:
    if abs(sharpe_A - sharpe_C) < abs(sharpe_A - sharpe_B) * 0.5:
        print("[결론] 수수료 이중청구가 주요 원인, 잔차는 구현 차이")
    else:
        print("[결론] 수수료 이중청구 외에 다른 구현 차이 존재 — 추가 조사 필요")
else:
    print("[결론] 차이 소폭 — 수수료 이중청구 부분적 원인, 데이터/로직 차이 추가 확인 필요")
