"""
Taker 포지션 매니저
- 최대 3개 동시 포지션
- 목표/스톱/엔트로피 정상화/타임아웃 조건으로 자동 청산
"""

from __future__ import annotations
import time
from .position import Position
from .. import config


class TakerManager:
    def __init__(self):
        self._positions: list[Position] = []
        self._next_id    = 0
        self.total_pnl   = 0.0
        self.wins        = 0
        self.losses      = 0

    # ── 진입 ─────────────────────────────────────────────────────────────────

    def open(
        self,
        direction:    str,
        entry_price:  float,
        size:         float,
        target_price: float,
        stop_price:   float,
    ) -> Position | None:
        if len(self.open_positions) >= config.TAKER_MAX_POSITIONS:
            return None

        pos = Position(
            pos_id       = self._next_id,
            direction    = direction,
            entry_price  = entry_price,
            size         = size,
            target_price = target_price,
            stop_price   = stop_price,
            fee_pct      = config.TAKER_FEE,
        )
        self._next_id += 1
        self._positions.append(pos)
        return pos

    # ── 청산 체크 ─────────────────────────────────────────────────────────────

    def check_exits(
        self,
        bid: float,
        ask: float,
        entropy_normalized: bool = False,
    ) -> list[str]:
        events = []
        for pos in self.open_positions:
            reason = pos.check_exit(bid, ask)

            if not reason and entropy_normalized:
                reason = "entropy_exit"
            if not reason and pos.hold_seconds > config.TAKER_MAX_HOLD_S:
                reason = "timeout"

            if reason:
                exit_px = bid if pos.direction == "long" else ask
                pnl     = pos.close(exit_px, reason)
                self.total_pnl += pnl
                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1

                sign = "+" if pnl >= 0 else ""
                events.append(
                    f"POS{pos.pos_id} [{reason.upper():>12}] "
                    f"{pos.direction.upper()} {pos.entry_price:.4f} -> {exit_px:.4f} | "
                    f"PnL: {sign}{pnl:.4f} USDT | "
                    f"hold: {pos.hold_seconds:.1f}s"
                )

        # 청산된 포지션 제거
        self._positions = [p for p in self._positions if p.is_open]
        return events

    # ── 집계 ─────────────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.is_open]

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def trade_count(self) -> int:
        return self.wins + self.losses

    def unrealized_pnl(self, mid: float) -> float:
        return sum(p.unrealized_pnl(mid) for p in self.open_positions)

    def status(self, mid: float) -> list[dict]:
        return [
            {
                "id":        p.pos_id,
                "dir":       p.direction,
                "entry":     p.entry_price,
                "target":    p.target_price,
                "stop":      p.stop_price,
                "size":      p.size,
                "hold_s":    round(p.hold_seconds, 1),
                "upnl":      round(p.unrealized_pnl(mid), 4),
                "upnl_pct":  round(p.unrealized_pct(mid) * 100, 3),
            }
            for p in self.open_positions
        ]
