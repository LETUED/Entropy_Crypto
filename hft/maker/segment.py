"""
마켓메이킹 세그먼트 — 독립 자본 단위
- 5개 세그먼트 중 하나가 청산돼도 나머지는 생존
- 호가 제출 / 체결 시뮬레이션 / 2% 스톱로스
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Quote:
    price:     float
    qty:       float
    side:      str    # "buy" | "sell"
    placed_at: float = field(default_factory=time.time)


class Segment:
    def __init__(
        self,
        seg_id:        int,
        capital:       float,
        stop_loss_pct: float,
        spread_pct:    float,
        maker_fee:     float = 0.0002,
        taker_fee:     float = 0.0005,
    ):
        self.seg_id        = seg_id
        self.initial_cap   = capital
        self.available     = capital
        self.stop_loss_pct = stop_loss_pct
        self.spread_pct    = spread_pct
        self.maker_fee     = maker_fee
        self.taker_fee     = taker_fee

        # 포지션
        self.inventory:      float = 0.0
        self.avg_entry:      float = 0.0
        self.realized_pnl:   float = 0.0

        # 현재 호가
        self.bid_quote: Optional[Quote] = None
        self.ask_quote: Optional[Quote] = None

        # 상태
        self.active      = False
        self.stopped_out = False
        self.activated_at: Optional[float] = None

    # ── 호가 계산 ─────────────────────────────────────────────────────────────

    def make_quotes(self, mid: float, obi: float, capital_ratio: float = 0.5) -> tuple[Quote, Quote]:
        """OBI로 호가를 비대칭으로 설정 (유리한 방향에 더 가깝게)"""
        half = mid * self.spread_pct
        skew = obi * half * 0.3           # 최대 30% 편향
        qty  = (self.available * capital_ratio) / mid

        bid = Quote(price=round(mid - half + skew, 2), qty=qty, side="buy")
        ask = Quote(price=round(mid + half + skew, 2), qty=qty, side="sell")
        return bid, ask

    def refresh_quotes(self, mid: float, obi: float) -> None:
        self.bid_quote, self.ask_quote = self.make_quotes(mid, obi)

    # ── 체결 시뮬레이션 ───────────────────────────────────────────────────────

    def simulate_fill(self, trade_price: float, trade_side: str) -> Optional[str]:
        """
        trade_side: "sell" → 테이커가 팔았음 → 내 bid 체결 가능
                    "buy"  → 테이커가 샀음  → 내 ask 체결 가능
        """
        filled = None

        # bid 체결 체크
        if self.bid_quote and trade_side == "sell":
            if trade_price <= self.bid_quote.price:
                q    = self.bid_quote.qty
                cost = q * self.bid_quote.price * (1.0 + self.maker_fee)
                if cost <= self.available:
                    self.available -= cost
                    prev = self.inventory
                    self.inventory += q
                    if prev == 0:
                        self.avg_entry = self.bid_quote.price
                    else:
                        self.avg_entry = (prev * self.avg_entry + q * self.bid_quote.price) / self.inventory
                    self.bid_quote = None
                    filled = "bid"

        # ask 체결 체크
        if self.ask_quote and trade_side == "buy":
            if trade_price >= self.ask_quote.price:
                q       = min(self.ask_quote.qty, max(self.inventory, 0.0))
                revenue = q * self.ask_quote.price * (1.0 - self.maker_fee)
                pnl     = (self.ask_quote.price - self.avg_entry) * q - q * self.ask_quote.price * self.maker_fee
                self.available   += revenue
                self.realized_pnl += pnl
                self.inventory   -= q
                if self.inventory <= 1e-9:
                    self.inventory = 0.0
                    self.avg_entry = 0.0
                self.ask_quote = None
                filled = "ask"

        return filled

    # ── 스톱로스 ─────────────────────────────────────────────────────────────

    def check_stop(self, current_price: float) -> bool:
        """True = 스톱로스 발동 → 세그먼트 비활성화"""
        if self.inventory <= 1e-9:
            return False
        loss_pct = (current_price - self.avg_entry) / self.avg_entry
        if loss_pct < -self.stop_loss_pct:
            # 시장가 청산 (taker fee)
            proceeds         = self.inventory * current_price * (1.0 - self.taker_fee)
            pnl              = (current_price - self.avg_entry) * self.inventory
            self.realized_pnl += pnl
            self.available   += proceeds
            self.inventory   = 0.0
            self.avg_entry   = 0.0
            self.bid_quote   = None
            self.ask_quote   = None
            self.stopped_out = True
            self.active      = False
            return True
        return False

    # ── 상태 ─────────────────────────────────────────────────────────────────

    def unrealized_pnl(self, mid: float) -> float:
        if self.inventory <= 0 or mid <= 0:
            return 0.0
        return self.inventory * (mid - self.avg_entry)

    def status(self, mid: float) -> dict:
        return {
            "id":            self.seg_id,
            "active":        self.active,
            "stopped":       self.stopped_out,
            "capital":       round(self.available, 4),
            "inventory":     round(self.inventory, 8),
            "avg_entry":     round(self.avg_entry, 2),
            "unrealized":    round(self.unrealized_pnl(mid), 4),
            "realized":      round(self.realized_pnl, 4),
            "bid_price":     round(self.bid_quote.price, 2) if self.bid_quote else None,
            "ask_price":     round(self.ask_quote.price, 2) if self.ask_quote else None,
        }
