"""
포지션 상태 관리.

state.json 구조:
{
  "positions": {
    "BTCUSDT": {
      "qty": 0.00015,
      "entry_price": 65000.0,
      "entry_time": "2024-01-15T14:00:00+00:00",
      "kelly_frac": 0.15,
      "notional": 10.0
    }
  },
  "total_realized_pnl": 0.0,
  "trade_count": 0
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from live.config import STATE_FILE, COIN_CAPITAL, MIN_NOTIONAL, FEE_RATE
from live.logger_setup import get_logger

log = get_logger()


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            s = json.load(f)
        # 구버전 state 호환: Earn 필드 없으면 추가
        s.setdefault("earn_subscribed",    0.0)
        s.setdefault("earn_total_interest", 0.0)
        s.setdefault("earn_apy",           0.0)
        return s
    return {
        "positions":            {},
        "total_realized_pnl":  0.0,
        "trade_count":         0,
        "earn_subscribed":     0.0,
        "earn_total_interest": 0.0,
        "earn_apy":            0.0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def calc_position_size(kelly_frac: float, coin_capital: float = COIN_CAPITAL) -> float:
    """
    코인 자본 × Kelly 분수. MIN_NOTIONAL 미달 시 MIN_NOTIONAL 사용.
    coin_capital: 동적 자본 계산값 (기본값은 config 고정값)
    반환: 투자할 USDT 금액
    """
    notional = coin_capital * kelly_frac
    if notional < MIN_NOTIONAL:
        log.info(f"Kelly 계산값 {notional:.2f} USDT < MIN_NOTIONAL {MIN_NOTIONAL} → {MIN_NOTIONAL} USDT 사용")
        notional = MIN_NOTIONAL
    # 코인 자본 초과 방지
    notional = min(notional, coin_capital)
    return notional


def open_position(state: dict, sym: str, qty: float,
                  entry_price: float, kelly_frac: float, notional: float) -> dict:
    state["positions"][sym] = {
        "qty":         qty,
        "entry_price": entry_price,
        "entry_time":  datetime.now(timezone.utc).isoformat(),
        "kelly_frac":  kelly_frac,
        "notional":    notional,
    }
    return state


def close_position(state: dict, sym: str, exit_price: float, reason: str) -> dict:
    pos = state["positions"].get(sym)
    if pos is None:
        return state

    entry_price = pos["entry_price"]
    qty         = pos["qty"]
    notional    = pos["notional"]

    gross_pnl   = (exit_price - entry_price) / entry_price
    net_pnl_pct = gross_pnl - FEE_RATE           # 진입 슬리피지는 이미 entry_price에 반영
    net_pnl_usdt = net_pnl_pct * notional

    state["total_realized_pnl"] += net_pnl_usdt
    state["trade_count"]        += 1
    del state["positions"][sym]

    log.info(
        f"[CLOSE] {sym} | 진입 {entry_price:.4f} → 청산 {exit_price:.4f} "
        f"| PnL {net_pnl_pct*100:+.3f}% ({net_pnl_usdt:+.4f} USDT) "
        f"| 이유: {reason} | 누적 PnL: {state['total_realized_pnl']:+.4f} USDT"
    )
    return state


def has_position(state: dict, sym: str) -> bool:
    return sym in state["positions"]


def get_held_hours(state: dict, sym: str) -> int:
    pos = state["positions"].get(sym)
    if pos is None:
        return 0
    entry = datetime.fromisoformat(pos["entry_time"])
    now   = datetime.now(timezone.utc)
    return int((now - entry).total_seconds() / 3600)
