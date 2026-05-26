"""
온체인/심리 데이터 → 엔트로피 변환

H_funding : 펀딩비 롤링 순열 엔트로피
             낮음 = 한쪽으로 지속 쏠림 = 방향성 있음
H_fg      : 공포탐욕 롤링 순열 엔트로피
             낮음 = 지속적 공포 or 지속적 탐욕 = 전환점 근처
H_onchain : 두 엔트로피를 합산한 통합 지표
"""

import numpy as np
import pandas as pd
from src.entropy.calculators import permutation_entropy


def funding_entropy(
    funding: pd.Series,
    price_index: pd.DatetimeIndex,
    window: int = 21,        # 21개 × 8H = 7일
    m: int = 3,
    tau: int = 1,
) -> pd.Series:
    """
    펀딩비 롤링 PE → 1H 가격 인덱스로 리샘플
    """
    # 롤링 PE
    values = funding.values
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = permutation_entropy(values[i - window + 1: i + 1], m=m, tau=tau)

    pe_series = pd.Series(result, index=funding.index, name="h_funding")

    # 1H 가격 인덱스로 forward-fill (펀딩비는 8H마다 갱신)
    reindexed = pe_series.reindex(price_index, method="ffill")
    return reindexed


def fear_greed_entropy(
    fg: pd.Series,
    price_index: pd.DatetimeIndex,
    window: int = 14,        # 14일 롤링
    m: int = 3,
    tau: int = 1,
) -> pd.Series:
    """
    공포탐욕지수 롤링 PE → 1H 가격 인덱스로 리샘플
    """
    values = fg.values
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = permutation_entropy(values[i - window + 1: i + 1], m=m, tau=tau)

    pe_series = pd.Series(result, index=fg.index, name="h_fg")

    # 일별 데이터를 1H로 forward-fill
    reindexed = pe_series.reindex(price_index, method="ffill")
    return reindexed


def combined_onchain_entropy(
    h_funding: pd.Series,
    h_fg: pd.Series,
    w_funding: float = 0.6,
    w_fg: float = 0.4,
) -> pd.Series:
    """
    H_onchain = w_funding * H_funding + w_fg * H_fg
    가중치: 펀딩비가 더 즉각적 신호이므로 더 높게
    """
    combined = w_funding * h_funding + w_fg * h_fg
    return combined.rename("h_onchain")
