"""
5-세그먼트 오케스트레이터
- 엔트로피 신호 → 세그먼트 활성화 / 철수 결정
- 각 거래마다 체결 시뮬레이션 + 스톱로스 체크
"""

from __future__ import annotations
from typing import Optional
from .segment import Segment
from .. import config


class SegmentManager:
    def __init__(self):
        self.segments: list[Segment] = [
            Segment(
                seg_id        = i,
                capital       = config.CAPITAL_PER_SEGMENT,
                stop_loss_pct = config.STOP_LOSS_PCT,
                spread_pct    = config.MAKER_SPREAD_PCT,
                maker_fee     = config.MAKER_FEE,
                taker_fee     = config.TAKER_FEE,
            )
            for i in range(config.N_SEGMENTS)
        ]
        self._stops_triggered = 0

    # ── 세그먼트 선택 ─────────────────────────────────────────────────────────

    def get_idle_segment(self) -> Optional[Segment]:
        """비활성 + 비청산 상태인 첫 번째 세그먼트"""
        for seg in self.segments:
            if not seg.active and not seg.stopped_out:
                return seg
        return None

    def activate(self, seg: Segment, mid: float, obi: float) -> None:
        import time
        seg.active       = True
        seg.stopped_out  = False
        seg.activated_at = time.time()
        seg.refresh_quotes(mid, obi)

    # ── 거래 처리 ─────────────────────────────────────────────────────────────

    def on_trade(self, price: float, side: str) -> list[str]:
        """aggTrade 수신 시 호출 — 체결 및 스톱 체크"""
        events = []
        for seg in self.segments:
            if not seg.active:
                continue
            result = seg.simulate_fill(price, side)
            if result:
                events.append(f"SEG{seg.seg_id} {result.upper()} FILLED @ {price:.2f}")
            if seg.check_stop(price):
                self._stops_triggered += 1
                events.append(f"SEG{seg.seg_id} STOP-LOSS @ {price:.2f} | rPnL={seg.realized_pnl:+.4f}")
        return events

    def refresh_all_quotes(self, mid: float, obi: float) -> None:
        for seg in self.segments:
            if seg.active and not seg.stopped_out:
                seg.refresh_quotes(mid, obi)

    # ── 긴급 철수 ─────────────────────────────────────────────────────────────

    def withdraw_all(self) -> None:
        """엔트로피 위험 신호 → 모든 호가 취소"""
        for seg in self.segments:
            seg.bid_quote = None
            seg.ask_quote = None

    def deactivate_all(self) -> None:
        """전체 세그먼트 비활성화 (비상 정지)"""
        self.withdraw_all()
        for seg in self.segments:
            seg.active = False

    # ── 집계 ─────────────────────────────────────────────────────────────────

    @property
    def total_realized_pnl(self) -> float:
        return sum(s.realized_pnl for s in self.segments)

    def total_unrealized_pnl(self, mid: float) -> float:
        return sum(s.unrealized_pnl(mid) for s in self.segments)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self.segments if s.active)

    @property
    def stopped_count(self) -> int:
        return self._stops_triggered

    def status(self, mid: float) -> list[dict]:
        return [s.status(mid) for s in self.segments]
