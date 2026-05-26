"""
실시간 거래 흐름 엔트로피
- Binance aggTrade 스트림의 매수/매도 시퀀스로 Shannon entropy 계산
- 낮은 엔트로피 = 단방향 쏠림 = 정보 거래자 활동 = 타이밍 신호

Volume-Adaptive 모드 (target_seconds > 0):
  trades/sec 를 실시간 측정해 window 크기를 자동 조정.
  새벽(0.9/s) → window≈27, 피크(5/s) → window≈150
  → 항상 동일한 30초 time-scale로 entropy 계산
"""

from __future__ import annotations
import time
from collections import deque
import numpy as np


class TradeEntropySignal:
    def __init__(
        self,
        window: int         = 100,   # 고정 모드 윈도우 (Maker 호환)
        threshold: float    = 0.85,
        target_seconds: float = 0.0, # > 0 이면 adaptive 모드
        min_window: int     = 30,
        max_window: int     = 300,
        tps_measure_s: float = 10.0, # TPS 측정 구간(초)
    ):
        self._adaptive      = target_seconds > 0
        self.target_seconds = target_seconds
        self.threshold      = threshold
        self.min_window     = min_window
        self.max_window     = max_window
        self.tps_measure_s  = tps_measure_s

        buf = max_window if self._adaptive else window
        self._trades = deque(maxlen=buf)  # 1=매수, 0=매도
        self._ts     = deque(maxlen=buf)  # 대응 타임스탬프
        self._window = window if not self._adaptive else min_window

    # ── 업데이트 ─────────────────────────────────────────────────────────────

    def update(self, is_buyer_maker: bool) -> float:
        """
        is_buyer_maker=True  → 테이커가 매도 (SELL 거래)
        is_buyer_maker=False → 테이커가 매수 (BUY 거래)
        """
        now = time.time()
        self._trades.append(0 if is_buyer_maker else 1)
        self._ts.append(now)

        if self._adaptive:
            tps = self._calc_tps(now)
            if tps > 0:
                target = int(tps * self.target_seconds)
                self._window = max(self.min_window, min(self.max_window, target))

        return self.entropy

    # ── 속성 ─────────────────────────────────────────────────────────────────

    @property
    def entropy(self) -> float:
        """Shannon entropy [0, 1]. 0=완전 단방향, 1=완전 랜덤"""
        n         = self._window
        available = len(self._trades)
        min_ok    = max(10, n // 3)        # 윈도우의 1/3 이상 채워져야 유효
        if available < min_ok:
            return 1.0
        trades = np.array(list(self._trades)[-n:], dtype=float)
        p_buy  = trades.mean()
        p_sell = 1.0 - p_buy
        if p_buy == 0.0 or p_sell == 0.0:
            return 0.0
        return float(-p_buy * np.log2(p_buy) - p_sell * np.log2(p_sell))

    @property
    def is_signal(self) -> bool:
        return self.entropy < self.threshold

    @property
    def buy_pressure(self) -> float:
        """최근 거래의 매수 비율 [0, 1]"""
        if not self._trades:
            return 0.5
        return float(np.mean(self._trades))

    @property
    def sample_count(self) -> int:
        return len(self._trades)

    @property
    def current_window(self) -> int:
        return self._window

    @property
    def tps(self) -> float:
        """최근 tps_measure_s 초 기준 trades/sec"""
        if not self._ts:
            return 0.0
        return self._calc_tps(self._ts[-1])

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _calc_tps(self, now: float) -> float:
        cutoff = now - self.tps_measure_s
        recent = [t for t in self._ts if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        elapsed = recent[-1] - recent[0]
        return (len(recent) - 1) / elapsed if elapsed > 0 else 0.0
