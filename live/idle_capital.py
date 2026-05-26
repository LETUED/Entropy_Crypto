"""
대기 자본 운용 — Binance Simple Earn Flexible.

흐름:
  진입 신호 → redeem(필요금액) → 매수
  청산 후   → subscribe(회수금액) → Earn 재예치
  매 사이클 → 현재 잔액 + 누적 이자 조회 → state 갱신

비교 지표:
  trade_pnl  : 스윙 전략 실현 손익 (USDT)
  earn_pnl   : Earn 누적 이자 (USDT)
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException

from live.config import API_KEY, API_SECRET
from live.logger_setup import get_logger

log = get_logger()

_EARN_ASSET = "USDT"


def _client() -> Client:
    return Client(API_KEY, API_SECRET)


def _get_product_id() -> str | None:
    """USDT Flexible Earn 상품 ID 조회."""
    client = _client()
    try:
        resp = client.get_simple_earn_flexible_product_list(asset=_EARN_ASSET)
        rows = resp.get("rows", [])
        if rows:
            return rows[0]["productId"]
        log.warning("USDT Flexible Earn 상품 없음")
        return None
    except BinanceAPIException as e:
        log.error(f"Earn 상품 조회 실패: {e}")
        return None


def get_earn_position() -> dict:
    """
    현재 Flexible Earn 포지션 반환.
    {
      "amount":   보유 원금 USDT,
      "interest": 누적 이자 USDT (오늘까지),
      "apy":      현재 APY (소수, 예: 0.042),
    }
    """
    client = _client()
    try:
        resp = client.get_simple_earn_flexible_product_position(asset=_EARN_ASSET)
        rows = resp.get("rows", [])
        if not rows:
            return {"amount": 0.0, "interest": 0.0, "apy": 0.0}
        row = rows[0]
        return {
            "amount":   float(row.get("totalAmount", 0)),
            "interest": float(row.get("totalInterest", 0)),
            "apy":      float(row.get("latestAnnualPercentageRate", 0)),
        }
    except BinanceAPIException as e:
        log.error(f"Earn 포지션 조회 실패: {e}")
        return {"amount": 0.0, "interest": 0.0, "apy": 0.0}


def subscribe(amount: float) -> bool:
    """
    amount USDT를 Flexible Earn에 예치.
    최소 예치 금액: 0.01 USDT.
    """
    if amount < 0.01:
        return False
    product_id = _get_product_id()
    if not product_id:
        return False

    client = _client()
    try:
        client.subscribe_simple_earn_flexible_product(
            productId=product_id,
            amount=f"{amount:.8f}",
        )
        log.info(f"[EARN] 예치 {amount:.4f} USDT → {product_id}")
        return True
    except BinanceAPIException as e:
        log.error(f"[EARN] 예치 실패: {e}")
        return False


def redeem(amount: float) -> bool:
    """
    amount USDT를 Flexible Earn에서 인출 (즉시 가능).
    """
    if amount < 0.01:
        return False
    product_id = _get_product_id()
    if not product_id:
        return False

    client = _client()
    try:
        client.redeem_simple_earn_flexible_product(
            productId=product_id,
            amount=f"{amount:.8f}",
            redeemAll=False,
        )
        log.info(f"[EARN] 인출 {amount:.4f} USDT ← {product_id}")
        return True
    except BinanceAPIException as e:
        log.error(f"[EARN] 인출 실패: {e}")
        return False


def redeem_all() -> bool:
    """전액 인출 (긴급용)."""
    product_id = _get_product_id()
    if not product_id:
        return False
    client = _client()
    try:
        client.redeem_simple_earn_flexible_product(
            productId=product_id,
            redeemAll=True,
        )
        log.info("[EARN] 전액 인출 완료")
        return True
    except BinanceAPIException as e:
        log.error(f"[EARN] 전액 인출 실패: {e}")
        return False


def update_earn_state(state: dict) -> dict:
    """
    현재 Earn 포지션으로 state 갱신.
    earn_total_interest = Binance가 리포트하는 누적 이자.
    """
    pos = get_earn_position()
    state["earn_subscribed"]    = pos["amount"]
    state["earn_total_interest"] = pos["interest"]
    state["earn_apy"]            = pos["apy"]
    return state


def log_comparison(state: dict) -> None:
    """트레이딩 PnL vs 대기자본 이자 비교 로그."""
    trade_pnl = state.get("total_realized_pnl", 0.0)
    earn_pnl  = state.get("earn_total_interest", 0.0)
    earn_apy  = state.get("earn_apy", 0.0)
    subscribed = state.get("earn_subscribed", 0.0)
    n_trades  = state.get("trade_count", 0)

    log.info("─" * 50)
    log.info("수익 비교")
    log.info(f"  트레이딩 PnL  : {trade_pnl:+.4f} USDT  ({n_trades}건)")
    log.info(f"  대기자본 이자 : {earn_pnl:+.4f} USDT  "
             f"({subscribed:.2f} USDT × APY {earn_apy*100:.2f}%)")
    log.info(f"  합산 수익     : {trade_pnl + earn_pnl:+.4f} USDT")
    log.info("─" * 50)
