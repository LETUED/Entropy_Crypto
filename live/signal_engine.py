"""
신호 계산 엔진.

백테스트와 동일한 로직. 마지막 완성된 캔들 기준으로 신호 판단.
- MPE < 10th percentile (of full series)
- RSI < 30
- Price > MA200
- Onchain entropy < 40th percentile
반환: (signal: bool, kelly_frac: float, debug_dict)
"""

import numpy as np
import pandas as pd

from live.config import (
    MPE_WINDOW, MA_PERIOD, RSI_WINDOW, ENTROPY_PCT, ONCHAIN_PCT,
    KELLY_PCT1, KELLY_PCT5, KELLY_PCT10,
)
from live.logger_setup import get_logger

log = get_logger()

# 백테스트 MPE 계산 재사용
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import (
    funding_entropy, fear_greed_entropy, combined_onchain_entropy,
)


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def check_signal(
    df: pd.DataFrame,
    funding_s: pd.Series,
    fg_s: pd.Series,
    sym: str,
) -> tuple[bool, float, dict]:
    """
    마지막 완성된 캔들(iloc[-2]) 기준으로 신호 판단.
    iloc[-1]은 현재 미완성 캔들이므로 제외.

    반환: (진입 신호 여부, Kelly 분수, 디버그 정보)
    """
    # 미완성 캔들 제외
    df = df.iloc[:-1].copy()
    if len(df) < MA_PERIOD + MPE_WINDOW:
        log.warning(f"{sym}: 데이터 부족 ({len(df)}봉)")
        return False, 0.0, {}

    close = df["close"]
    price = float(close.iloc[-1])

    # MA200
    ma200 = float(close.rolling(MA_PERIOD).mean().iloc[-1])
    above_ma = price > ma200

    # RSI
    rsi_val = float(compute_rsi(close).iloc[-1])
    rsi_ok  = rsi_val < 30

    # MPE (no cache — live 계산)
    mpe = rolling_mpe(close, window=MPE_WINDOW)
    mpe_clean   = mpe.dropna()
    mpe_val     = float(mpe.iloc[-1]) if not np.isnan(mpe.iloc[-1]) else np.nan
    mpe_thresh  = float(np.percentile(mpe_clean, ENTROPY_PCT))
    mpe_ok      = (not np.isnan(mpe_val)) and (mpe_val <= mpe_thresh)
    mpe_pct     = float(np.mean(mpe_clean <= mpe_val) * 100) if not np.isnan(mpe_val) else np.nan

    # Kelly 분수
    k_pct1  = np.percentile(mpe_clean, 1)
    k_pct5  = np.percentile(mpe_clean, 5)
    k_pct10 = np.percentile(mpe_clean, 10)

    if   not np.isnan(mpe_val) and mpe_val <= k_pct1:  kelly = KELLY_PCT1
    elif not np.isnan(mpe_val) and mpe_val <= k_pct5:  kelly = KELLY_PCT5
    elif not np.isnan(mpe_val) and mpe_val <= k_pct10: kelly = KELLY_PCT10
    else:                                               kelly = 0.0

    # 온체인 엔트로피
    h_f   = funding_entropy(funding_s, df.index)
    h_g   = fear_greed_entropy(fg_s, df.index)
    h_oc  = combined_onchain_entropy(h_f, h_g)

    oc_ok = True
    oc_val = np.nan
    if h_oc is not None and not h_oc.empty:
        oc_clean = h_oc.dropna()
        if len(oc_clean) > 0:
            idx = df.index[-1]
            if idx in h_oc.index:
                oc_val    = float(h_oc.loc[idx])
                oc_thresh = float(np.percentile(oc_clean, ONCHAIN_PCT))
                oc_ok     = oc_val <= oc_thresh

    signal = above_ma and rsi_ok and mpe_ok and oc_ok and kelly > 0

    debug = {
        "price":      price,
        "ma200":      round(ma200, 4),
        "above_ma":   above_ma,
        "rsi":        round(rsi_val, 2),
        "rsi_ok":     rsi_ok,
        "mpe":        round(mpe_val, 6) if not np.isnan(mpe_val) else None,
        "mpe_pct":    round(mpe_pct, 1) if not np.isnan(mpe_pct) else None,
        "mpe_ok":     mpe_ok,
        "oc_val":     round(oc_val, 4) if not np.isnan(oc_val) else None,
        "oc_ok":      oc_ok,
        "kelly":      kelly,
        "signal":     signal,
    }

    return signal, kelly, debug


def check_exit(
    position: dict,
    df: pd.DataFrame,
    current_bar: int,
) -> tuple[bool, str]:
    """
    열린 포지션의 청산 조건 확인.
    position: state.json의 개별 포지션 dict
    current_bar: 현재 바 인덱스 (진입 후 몇 시간 경과)

    반환: (청산 여부, 이유)
    """
    df = df.iloc[:-1]  # 미완성 캔들 제외
    close  = df["close"]
    rsi    = compute_rsi(close)
    rsi_v  = float(rsi.iloc[-1])
    held_h = current_bar

    if rsi_v > 50:
        return True, f"RSI>{rsi_v:.1f}"
    if held_h >= 168:
        return True, f"최대보유{held_h}H"
    return False, ""
