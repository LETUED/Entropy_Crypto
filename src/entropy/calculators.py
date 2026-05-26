"""
엔트로피 및 경쟁 지표 계산

파라미터는 선행 논문 기반으로 고정 (과적합 방지):
- m=3, tau=1, scales=[1,2,4,8]
출처: Multiscale Permutation Entropy (preprints.org/202511.1980)
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


# ── 순열 엔트로피 (Permutation Entropy) ──────────────────────────────────────

def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    정규화된 순열 엔트로피 [0, 1]
    0 = 완전한 질서, 1 = 완전한 무질서
    """
    n = len(x)
    patterns: dict = {}

    for i in range(n - (m - 1) * tau):
        pattern = tuple(np.argsort(x[i : i + m * tau : tau], kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    total = sum(patterns.values())
    probs = np.array(list(patterns.values())) / total
    pe = -np.sum(probs * np.log2(probs + 1e-12))

    max_pe = math.log2(math.factorial(m))
    return pe / max_pe if max_pe > 0 else 0.0


def multiscale_permutation_entropy(
    x: np.ndarray,
    m: int = 3,
    tau: int = 1,
    scales: list = None,
) -> float:
    """
    멀티스케일 순열 엔트로피: 여러 시간 스케일에서 PE 평균
    """
    if scales is None:
        scales = [1, 2, 4, 8]

    pe_values = []
    for scale in scales:
        # 코스-그레인(coarse-grain): scale 단위로 평균 내기
        n = len(x) // scale
        if n < m + 1:
            continue
        coarse = x[: n * scale].reshape(n, scale).mean(axis=1)
        pe_values.append(permutation_entropy(coarse, m=m, tau=tau))

    return float(np.mean(pe_values)) if pe_values else float("nan")


# ── 롤링 엔트로피 계산 ───────────────────────────────────────────────────────

def rolling_mpe(
    series: pd.Series,
    window: int = 168,   # 7일 (1H 기준)
    m: int = 3,
    tau: int = 1,
    scales: list = None,
    cache_key: str = None,   # 캐시 식별자 (예: "BTCUSDT_1h_2021-01-01_2025-01-01")
) -> pd.Series:
    """
    시계열 전체에 대해 롤링 윈도우로 MPE 계산.
    cache_key 지정 시 결과를 data/cache/에 parquet로 저장 — 재실행 시 즉시 로드.
    """
    if scales is None:
        scales = [1, 2, 4, 8]

    # 캐시 조회 — MPE 파라미터 고정(m=3,tau=1,scales=[1,2,4,8],window=168)이므로
    # 데이터 식별자만으로 키 구성
    if cache_key:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _CACHE_DIR / f"mpe_{cache_key}.parquet"
        if cache_path.exists():
            print(f"캐시 로드: {cache_path.name}")
            return pd.read_parquet(cache_path)["mpe"]

    values = series.values
    result = np.full(len(values), np.nan)

    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1 : i + 1]
        result[i] = multiscale_permutation_entropy(window_data, m=m, tau=tau, scales=scales)

    mpe_series = pd.Series(result, index=series.index, name="mpe")

    if cache_key:
        mpe_series.to_frame().to_parquet(cache_path)
        print(f"MPE 캐시 저장: {cache_path.name}")

    return mpe_series


# ── Delta MPE (엔트로피 변화율) ──────────────────────────────────────────────

def rolling_delta_mpe(
    mpe_series: pd.Series,
    smooth_window: int = 20,
) -> pd.Series:
    """
    MPE 변화율 (기울기)
    - 스무딩 후 1차 차분
    - 음수 극값 = MPE가 빠르게 하락 중 = 고→저 레짐 전환 중
    근거: Thermodynamic Analysis (2023) — Delta Entropy가 분산의 41~57% 설명
    """
    smoothed = mpe_series.rolling(smooth_window).mean()
    delta    = smoothed.diff()
    return delta.rename("delta_mpe")


# ── 경쟁 지표 ────────────────────────────────────────────────────────────────

def bollinger_band_width(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    볼린저 밴드 수축 지표 (BBW)
    낮을수록 변동성 압축 = 엔트로피 낮음과 유사한 효과
    """
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    bbw = (std * 2) / ma  # 정규화된 밴드폭
    return bbw.rename("bbw")


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR: 절대 변동성 측정
    낮을수록 조용한 시장 = 엔트로피 낮음과 유사
    """
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1
    ).max(axis=1)

    atr = tr.rolling(window).mean()
    # 가격 대비 정규화 (%) — 데이터셋 간 비교 가능하게
    atr_pct = atr / df["close"] * 100
    return atr_pct.rename("atr_pct")
