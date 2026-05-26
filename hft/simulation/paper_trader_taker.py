"""
Taker 방향성 전략 페이퍼 트레이더

진입 조건:
  1. 엔트로피 < ENTROPY_THRESHOLD (예측 가능 구간)
  2. OBI_THRESHOLD < |OBI| <= OBI_ACTIVATE_MAX (방향성 있되 극단 아님)
  3. 새 신호 첫 발생 시에만 진입

청산 조건:
  - 목표가 도달 (0.2%)
  - 스톱 도달  (0.1%)
  - 엔트로피 정상화
  - 60초 타임아웃
"""

from __future__ import annotations
import asyncio
import json
import sys
import time

import websockets

from ..signals.entropy_rt import TradeEntropySignal
from ..signals.obi import OrderBookImbalance
from ..taker.manager import TakerManager
from .. import config

sys.stdout.reconfigure(encoding="utf-8")


class TakerPaperTrader:
    def __init__(self, symbol: str = config.SYMBOL):
        self.symbol  = symbol.lower()
        self.entropy = TradeEntropySignal(
            threshold      = config.ENTROPY_THRESHOLD,
            target_seconds = config.ADAPTIVE_WINDOW_SECONDS,
            min_window     = config.ADAPTIVE_WINDOW_MIN,
            max_window     = config.ADAPTIVE_WINDOW_MAX,
        )
        self.obi     = OrderBookImbalance(config.OBI_BOOK_LEVELS, config.OBI_THRESHOLD)
        self.manager = TakerManager()

        self._trade_count       = 0
        self._last_display      = 0.0
        self._entropy_signaling = False
        self._events: list[str] = []

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        url = (
            f"{config.WS_BASE}?streams="
            f"{self.symbol}@aggTrade/"
            f"{self.symbol}@depth20@100ms"
        )
        print(f"연결 중... Binance WebSocket ({self.symbol.upper()})")
        async with websockets.connect(url, ping_interval=20) as ws:
            print(f"연결됨. Taker 페이퍼 트레이딩 시작\n")
            async for raw in ws:
                msg    = json.loads(raw)
                stream = msg.get("stream", "")
                data   = msg.get("data", {})
                if "aggTrade" in stream:
                    self._on_trade(data)
                elif "depth" in stream:
                    self._on_depth(data)

    # ── 거래 수신 → 청산 체크 ──────────────────────────────────────────────────

    def _on_trade(self, data: dict) -> None:
        price          = float(data["p"])
        is_buyer_maker = bool(data["m"])
        self.entropy.update(is_buyer_maker)
        self._trade_count += 1

        # 매 거래마다 청산 조건 체크
        bid = self.obi.best_bid or price
        ask = self.obi.best_ask or price
        events = self.manager.check_exits(
            bid                = bid,
            ask                = ask,
            entropy_normalized = not self.entropy.is_signal,
        )
        self._events.extend(events)

        now = time.time()
        if now - self._last_display >= 2.0:
            self._display(price)
            self._last_display = now

    # ── 오더북 수신 → 진입 결정 ───────────────────────────────────────────────

    def _on_depth(self, data: dict) -> None:
        self.obi.update_book(data.get("bids", []), data.get("asks", []))

        mid     = self.obi.mid_price
        bid     = self.obi.best_bid
        ask     = self.obi.best_ask
        if mid <= 0:
            return

        is_signal = self.entropy.is_signal
        direction = self.obi.direction
        obi_abs   = abs(self.obi.obi)
        in_range  = config.OBI_THRESHOLD < obi_abs <= config.OBI_ACTIVATE_MAX
        tps_ok    = self.entropy.tps >= config.VOLUME_GATE_MIN_TPS

        # ── 진입 ──────────────────────────────────────────────────────────────
        if is_signal and direction != "neutral" and in_range and tps_ok:
            if not self._entropy_signaling:                       # 새 신호 첫 발생
                self._entropy_signaling = True
                self._try_open(direction, bid, ask)

        elif not is_signal:
            self._entropy_signaling = False

    def _try_open(self, direction: str, bid: float, ask: float) -> None:
        capital = config.TOTAL_CAPITAL_USDT * config.TAKER_POSITION_PCT

        if direction == "long":
            entry  = ask                                           # 매수 = ask에 체결
            target = round(entry * (1 + config.TAKER_TARGET_PCT), 6)
            stop   = round(entry * (1 - config.TAKER_STOP_PCT),   6)
        else:
            entry  = bid                                           # 매도 = bid에 체결
            target = round(entry * (1 - config.TAKER_TARGET_PCT), 6)
            stop   = round(entry * (1 + config.TAKER_STOP_PCT),   6)

        size = capital / entry
        pos  = self.manager.open(direction, entry, size, target, stop)
        if pos:
            rr = config.TAKER_TARGET_PCT / config.TAKER_STOP_PCT
            self._events.append(
                f"POS{pos.pos_id} [      OPEN] "
                f"{direction.upper()} @ {entry:.4f} | "
                f"target={target:.4f} stop={stop:.4f} | "
                f"R:R=1:{rr:.0f} | "
                f"PE={self.entropy.entropy:.3f} OBI={self.obi.obi:+.3f}"
            )

    # ── 출력 ─────────────────────────────────────────────────────────────────

    def _display(self, price: float) -> None:
        mid       = self.obi.mid_price or price
        entropy   = self.entropy.entropy
        obi_val   = self.obi.obi
        direction = self.obi.direction
        r_pnl     = self.manager.total_pnl
        u_pnl     = self.manager.unrealized_pnl(mid)
        n_open    = len(self.manager.open_positions)
        wr        = self.manager.win_rate * 100
        tps       = self.entropy.tps
        win       = self.entropy.current_window
        ts        = time.strftime("%H:%M:%S")

        arrow    = {"buy": "^", "sell": "v", "neutral": "-"}.get(direction, "-")
        sig_str  = "SIGNAL" if self.entropy.is_signal else "quiet "
        gate_str = "GATE" if tps < config.VOLUME_GATE_MIN_TPS else "    "

        print(
            f"[{ts}] "
            f"Price: {price:>10.4f} | "
            f"Vol: {tps:4.1f}/s W:{win:3d} {gate_str} | "
            f"PE: {entropy:.3f} ({sig_str}) | "
            f"OBI: {obi_val:+.3f}{arrow} | "
            f"Open: {n_open}/{config.TAKER_MAX_POSITIONS} | "
            f"Trades: {self.manager.trade_count} "
            f"W/L: {self.manager.wins}/{self.manager.losses} ({wr:.0f}%) | "
            f"rPnL: {r_pnl:+.4f} | "
            f"uPnL: {u_pnl:+.4f} USDT"
        )

        for s in self.manager.status(mid):
            arrow_p = "^" if s["dir"] == "long" else "v"
            print(
                f"  POS{s['id']} [{arrow_p}] "
                f"entry={s['entry']:.4f} "
                f"target={s['target']:.4f} "
                f"stop={s['stop']:.4f} | "
                f"uPnL: {s['upnl']:+.4f} ({s['upnl_pct']:+.3f}%) | "
                f"hold: {s['hold_s']}s"
            )

        for ev in self._events:
            print(f"  >>> {ev}")
        self._events.clear()
        print()
