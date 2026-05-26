"""
페이퍼 트레이더 — 실제 Binance WebSocket 데이터 + 가상 주문
실제 돈 없이 전략 검증

스트림:
  {symbol}@aggTrade  — 실시간 거래 (엔트로피 계산)
  {symbol}@depth20@100ms — 오더북 스냅샷 (OBI 계산)
"""

from __future__ import annotations
import asyncio
import json
import time

import websockets

from ..signals.entropy_rt import TradeEntropySignal
from ..signals.obi import OrderBookImbalance
from ..maker.manager import SegmentManager
from .. import config


import sys
sys.stdout.reconfigure(encoding="utf-8")

ARROW = {"buy": "↑", "sell": "↓", "neutral": "─"}
RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"
BOLD  = "\033[1m"


class PaperTrader:
    def __init__(self, symbol: str = config.SYMBOL):
        self.symbol  = symbol.lower()
        self.entropy = TradeEntropySignal(config.ENTROPY_WINDOW, config.ENTROPY_THRESHOLD)
        self.obi     = OrderBookImbalance(config.OBI_BOOK_LEVELS, config.OBI_THRESHOLD)
        self.manager = SegmentManager()

        self._trade_count       = 0
        self._last_display      = 0.0
        self._entropy_signaling = False   # 엣지 감지: 새 신호 발생 시에만 세그먼트 활성화
        self._events: list[str] = []

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        url = (
            f"{config.WS_BASE}?streams="
            f"{self.symbol}@aggTrade/"
            f"{self.symbol}@depth20@100ms"
        )
        print(f"{BOLD}연결 중... Binance WebSocket{RESET}")
        async with websockets.connect(url, ping_interval=20) as ws:
            print(f"{GREEN}연결됨. 페이퍼 트레이딩 시작: {self.symbol.upper()}{RESET}\n")
            async for raw in ws:
                msg    = json.loads(raw)
                stream = msg.get("stream", "")
                data   = msg.get("data", {})
                if "aggTrade" in stream:
                    self._on_trade(data)
                elif "depth" in stream:
                    self._on_depth(data)

    # ── 거래 수신 ─────────────────────────────────────────────────────────────

    def _on_trade(self, data: dict) -> None:
        price          = float(data["p"])
        is_buyer_maker = bool(data["m"])     # True = SELL 거래
        trade_side     = "sell" if is_buyer_maker else "buy"

        self.entropy.update(is_buyer_maker)
        events = self.manager.on_trade(price, trade_side)
        self._events.extend(events)
        self._trade_count += 1

        now = time.time()
        if now - self._last_display >= 2.0:
            self._display(price)
            self._last_display = now

    # ── 오더북 수신 ───────────────────────────────────────────────────────────

    def _on_depth(self, data: dict) -> None:
        self.obi.update_book(data.get("bids", []), data.get("asks", []))

        mid = self.obi.mid_price
        if mid <= 0:
            return

        is_signal   = self.entropy.is_signal
        direction   = self.obi.direction
        obi_abs     = abs(self.obi.obi)
        safe_to_act = obi_abs <= config.OBI_ACTIVATE_MAX  # Adverse Selection 방어

        if is_signal and direction != "neutral" and safe_to_act:
            # 신호 첫 발생 시에만 새 세그먼트 활성화
            if not self._entropy_signaling:
                self._entropy_signaling = True
                seg = self.manager.get_idle_segment()
                if seg:
                    self.manager.activate(seg, mid, self.obi.obi)
                    self._events.append(
                        f"SEG{seg.seg_id} ACTIVATED | entropy={self.entropy.entropy:.3f} | "
                        f"OBI={self.obi.obi:+.3f} ({direction})"
                    )
            # 활성 세그먼트 호가 갱신
            self.manager.refresh_all_quotes(mid, self.obi.obi)

        elif is_signal and obi_abs > config.OBI_ACTIVATE_MAX:
            # OBI 너무 강함 → Adverse Selection 위험 → 호가 철수
            self.manager.withdraw_all()

        else:
            # 신호 없음 → 호가 철수 (위험 구간)
            if self._entropy_signaling:
                self._entropy_signaling = False
                self.manager.withdraw_all()

    # ── 출력 ─────────────────────────────────────────────────────────────────

    def _display(self, price: float) -> None:
        mid       = self.obi.mid_price or price
        entropy   = self.entropy.entropy
        obi_val   = self.obi.obi
        direction = self.obi.direction
        r_pnl     = self.manager.total_realized_pnl
        u_pnl     = self.manager.total_unrealized_pnl(mid)
        active    = self.manager.active_count
        stops     = self.manager.stopped_count

        sig_str = f"{RED}SIGNAL{RESET}" if self.entropy.is_signal else f"{GRAY}quiet {RESET}"
        dir_str = ARROW.get(direction, "─")
        pnl_col = GREEN if r_pnl >= 0 else RED
        ts      = time.strftime("%H:%M:%S")

        print(
            f"[{ts}] "
            f"Price: {BOLD}{price:>10,.2f}{RESET} | "
            f"PE: {entropy:.3f} ({sig_str}) | "
            f"OBI: {obi_val:+.3f} {dir_str} | "
            f"Active: {active}/{config.N_SEGMENTS} | "
            f"Stops: {stops} | "
            f"rPnL: {pnl_col}{r_pnl:+.4f}{RESET} | "
            f"uPnL: {u_pnl:+.4f} USDT | "
            f"T: {self._trade_count:,}"
        )

        # 활성/청산 세그먼트 상세
        for s in self.manager.segments:
            if s.active or s.stopped_out:
                st = s.status(mid)
                state = f"{RED}STOP{RESET}" if st["stopped"] else f"{GREEN} ACT{RESET}"
                bid_s = f"{st['bid_price']}" if st["bid_price"] else "  ─  "
                ask_s = f"{st['ask_price']}" if st["ask_price"] else "  ─  "
                print(
                    f"  SEG{st['id']} [{state}] "
                    f"Inv: {st['inventory']:.6f} | "
                    f"Entry: {st['avg_entry']:>10.2f} | "
                    f"Bid: {bid_s:>10} / Ask: {ask_s:>10} | "
                    f"uPnL: {st['unrealized']:+.4f} | "
                    f"rPnL: {st['realized']:+.4f}"
                )

        # 이벤트 로그 (체결, 스톱)
        for ev in self._events:
            print(f"  {CYAN}>>> {ev}{RESET}")
        self._events.clear()

        print()  # 빈 줄로 구분
