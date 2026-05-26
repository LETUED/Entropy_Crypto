"""
오더북 불균형 (Order Book Imbalance)
- 방향 신호: 엔트로피가 "언제"를 잡으면, OBI는 "어느 방향"을 잡음
- OBI > +threshold → 매수 압력
- OBI < -threshold → 매도 압력
"""


class OrderBookImbalance:
    def __init__(self, levels: int = 10, threshold: float = 0.15):
        self.levels    = levels
        self.threshold = threshold
        self._bids: dict[float, float] = {}  # price → qty
        self._asks: dict[float, float] = {}

    # ── 업데이트 ─────────────────────────────────────────────────────────────

    def update_book(self, bids: list, asks: list) -> None:
        """depth20 스냅샷 전체 교체"""
        self._bids = {float(p): float(q) for p, q in bids}
        self._asks = {float(p): float(q) for p, q in asks}

    # ── 속성 ─────────────────────────────────────────────────────────────────

    @property
    def obi(self) -> float:
        """[-1, 1]. 양수=매수 압력, 음수=매도 압력"""
        if not self._bids or not self._asks:
            return 0.0
        top_bids = sorted(self._bids, reverse=True)[: self.levels]
        top_asks = sorted(self._asks)[: self.levels]
        bid_vol  = sum(self._bids[p] for p in top_bids)
        ask_vol  = sum(self._asks[p] for p in top_asks)
        total    = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    @property
    def best_bid(self) -> float:
        return max(self._bids) if self._bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self._asks) if self._asks else 0.0

    @property
    def mid_price(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self._bids and self._asks else 0.0

    @property
    def direction(self) -> str:
        o = self.obi
        if o > self.threshold:
            return "buy"
        if o < -self.threshold:
            return "sell"
        return "neutral"
