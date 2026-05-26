"""
개별 포지션 — Taker 방향성 전략
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class Position:
    pos_id:       int
    direction:    str    # "long" | "short"
    entry_price:  float
    size:         float  # base asset 수량
    target_price: float
    stop_price:   float
    fee_pct:      float = 0.0005
    opened_at:    float = field(default_factory=time.time)

    exit_price:  float = None
    exit_reason: str   = None
    closed_at:   float = None

    # ── 상태 ─────────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def hold_seconds(self) -> float:
        return time.time() - self.opened_at

    # ── 손익 ─────────────────────────────────────────────────────────────────

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open:
            return 0.0
        if self.direction == "long":
            return self.size * (current_price - self.entry_price)
        return self.size * (self.entry_price - current_price)

    def unrealized_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.direction == "long":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price

    # ── 청산 조건 체크 ────────────────────────────────────────────────────────

    def check_exit(self, bid: float, ask: float) -> str | None:
        """'target' | 'stop' | None"""
        if self.direction == "long":
            if bid >= self.target_price:
                return "target"
            if bid <= self.stop_price:
                return "stop"
        else:
            if ask <= self.target_price:
                return "target"
            if ask >= self.stop_price:
                return "stop"
        return None

    # ── 청산 실행 ─────────────────────────────────────────────────────────────

    def close(self, exit_price: float, reason: str) -> float:
        """포지션 청산 → 실현 손익 반환"""
        self.exit_price  = exit_price
        self.exit_reason = reason
        self.closed_at   = time.time()

        entry_fee = self.size * self.entry_price * self.fee_pct
        exit_fee  = self.size * exit_price * self.fee_pct

        if self.direction == "long":
            gross = self.size * (exit_price - self.entry_price)
        else:
            gross = self.size * (self.entry_price - exit_price)

        return gross - entry_fee - exit_fee
