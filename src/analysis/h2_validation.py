"""
H2 가설 검증:
"엔트로피 하위 5% 구간에서 이후 N시간 변동폭이 평균의 2.89배 이상이다"

검증 방법:
1. 롤링 MPE 계산
2. 엔트로피 하위 5% 구간 식별
3. 이후 24H 변동폭 측정
4. Mann-Whitney U test
5. 경쟁 지표(BBW, ATR)와 동일 실험 → 엔트로피 고유 효과 입증
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.entropy.calculators import rolling_mpe, bollinger_band_width, average_true_range


FORWARD_HOURS = 24      # 이후 몇 시간 변동폭을 볼 것인가
ENTROPY_THRESHOLD = 5   # 하위 몇 % 구간을 "저엔트로피 구간"으로 볼 것인가


def compute_forward_volatility(df: pd.DataFrame, hours: int = FORWARD_HOURS) -> pd.Series:
    """이후 N시간 동안의 최대 변동폭 (%) 계산"""
    high_fwd = df["high"].rolling(hours).max().shift(-hours)
    low_fwd = df["low"].rolling(hours).min().shift(-hours)
    volatility = (high_fwd - low_fwd) / df["close"] * 100
    return volatility.rename("fwd_volatility")


def run_indicator_test(
    signal: pd.Series,
    fwd_vol: pd.Series,
    label: str,
    low_is_low_entropy: bool = True,
    threshold_pct: float = ENTROPY_THRESHOLD,
) -> dict:
    """
    단일 지표에 대한 H2 검증 실행
    low_is_low_entropy: True면 하위 threshold_pct%, False면 상위 threshold_pct%
    """
    combined = pd.DataFrame({"signal": signal, "fwd_vol": fwd_vol}).dropna()

    if low_is_low_entropy:
        cutoff = np.percentile(combined["signal"], threshold_pct)
        mask_low = combined["signal"] <= cutoff
    else:
        cutoff = np.percentile(combined["signal"], 100 - threshold_pct)
        mask_low = combined["signal"] >= cutoff

    low_vol = combined.loc[mask_low, "fwd_vol"].values
    other_vol = combined.loc[~mask_low, "fwd_vol"].values

    if len(low_vol) < 10 or len(other_vol) < 10:
        return {"label": label, "error": "샘플 부족"}

    ratio = np.median(low_vol) / np.median(other_vol)
    stat, p_value = stats.mannwhitneyu(low_vol, other_vol, alternative="greater")

    return {
        "label": label,
        "n_low_entropy": len(low_vol),
        "n_other": len(other_vol),
        "median_low_entropy_vol": round(np.median(low_vol), 4),
        "median_other_vol": round(np.median(other_vol), 4),
        "ratio": round(ratio, 4),
        "mean_low_entropy_vol": round(np.mean(low_vol), 4),
        "mean_other_vol": round(np.mean(other_vol), 4),
        "mean_ratio": round(np.mean(low_vol) / np.mean(other_vol), 4),
        "mannwhitney_stat": round(stat, 2),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "cutoff_value": round(cutoff, 6),
    }


def validate_h2(df: pd.DataFrame, mpe_window: int = 168, mpe=None) -> dict:
    """
    H2 검증 메인 함수

    Returns: 각 지표별 검증 결과 딕셔너리
    """
    print("=" * 60)
    print("H2 가설 검증 시작")
    print(f"데이터: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,}개 캔들)")
    print(f"설정: 이후 {FORWARD_HOURS}H 변동폭, 하위 {ENTROPY_THRESHOLD}% 구간")
    print("=" * 60)

    # 1. 이후 변동폭 계산
    print("\n[1/4] 이후 변동폭 계산...")
    fwd_vol = compute_forward_volatility(df)

    # 2. 각 지표 계산 (외부에서 받은 MPE 재사용, 없으면 새로 계산)
    if mpe is None:
        print("[2/4] 멀티스케일 순열 엔트로피 계산 중... (수 분 소요)")
        mpe = rolling_mpe(df["close"], window=mpe_window)
    else:
        print("[2/4] MPE 외부 입력 사용")

    print("[3/4] 경쟁 지표 계산 (BBW, ATR)...")
    bbw = bollinger_band_width(df)
    atr = average_true_range(df)

    # 3. 각 지표별 H2 검증
    print("[4/4] 통계 검정 실행...")
    results = {
        "mpe": run_indicator_test(mpe, fwd_vol, label="멀티스케일 순열 엔트로피 (MPE)", low_is_low_entropy=True),
        "bbw": run_indicator_test(bbw, fwd_vol, label="볼린저 밴드 수축 (BBW)", low_is_low_entropy=True),
        "atr": run_indicator_test(atr, fwd_vol, label="ATR (변동성)", low_is_low_entropy=True),
    }

    # 4. 결과 출력
    _print_results(results)

    # 5. 원시 데이터 반환 (시각화용)
    results["_data"] = pd.DataFrame({
        "close": df["close"],
        "mpe": mpe,
        "bbw": bbw,
        "atr": atr,
        "fwd_vol": fwd_vol,
    })

    return results


def _print_results(results: dict):
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)

    for key, r in results.items():
        if key == "_data":
            continue
        print(f"\n▶ {r['label']}")
        if "error" in r:
            print(f"  오류: {r['error']}")
            continue
        print(f"  저엔트로피 구간 샘플 수 : {r['n_low_entropy']:,}개")
        print(f"  기타 구간 샘플 수       : {r['n_other']:,}개")
        print(f"  저엔트로피 중앙값 변동폭 : {r['median_low_entropy_vol']:.4f}%")
        print(f"  기타 구간 중앙값 변동폭  : {r['median_other_vol']:.4f}%")
        sig_vol = "[O]" if r['ratio'] >= 2.0 else "[X]"
        sig_p = "[O] 유의" if r['significant'] else "[X] 비유의"
        print(f"  변동폭 배율 (중앙값)     : {r['ratio']:.2f}x  {sig_vol}")
        print(f"  변동폭 배율 (평균)       : {r['mean_ratio']:.2f}x")
        print(f"  Mann-Whitney p-value    : {r['p_value']:.6f}  {sig_p}")

    print("\n" + "=" * 60)
    print("H2 판정")
    mpe_r = results.get("mpe", {})
    if mpe_r.get("significant") and mpe_r.get("ratio", 0) >= 2.0:
        print("[H2 채택] MPE 저엔트로피 구간에서 변동폭이 유의미하게 크다")
    elif mpe_r.get("significant"):
        print("[부분 채택] 통계적으로 유의하나 배율이 2x 미만")
    else:
        print("[H2 기각] 통계적으로 유의미한 차이 없음")
    print("=" * 60)
