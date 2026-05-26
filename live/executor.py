"""
Binance Spot 주문 실행.

진입: 시장가 매수 (notional USDT 기준)
청산: 전량 시장가 매도
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException

from live.config import API_KEY, API_SECRET, FEE_RATE
from live.logger_setup import get_logger

log = get_logger()


def _client() -> Client:
    return Client(API_KEY, API_SECRET)


def get_symbol_info(sym: str) -> dict:
    """심볼의 LOT_SIZE, MIN_NOTIONAL 필터 조회."""
    client = _client()
    info = client.get_symbol_info(sym)
    filters = {f["filterType"]: f for f in info["filters"]}
    step_size   = float(filters["LOT_SIZE"]["stepSize"])
    min_qty     = float(filters["LOT_SIZE"]["minQty"])
    min_notional = float(filters.get("NOTIONAL", {}).get("minNotional",
                         filters.get("MIN_NOTIONAL", {}).get("minNotional", 10.0)))
    return {
        "step_size":    step_size,
        "min_qty":      min_qty,
        "min_notional": min_notional,
        "base_prec":    info["baseAssetPrecision"],
        "quote_prec":   info["quoteAssetPrecision"],
    }


def floor_qty(qty: float, step_size: float) -> float:
    """LOT_SIZE step에 맞게 수량 내림."""
    factor = 1.0 / step_size
    return int(qty * factor) / factor


def market_buy(sym: str, notional: float) -> dict | None:
    """
    시장가 매수. notional USDT 기준 quoteOrderQty 사용.
    반환: {"qty": ..., "avg_price": ..., "notional": ...} 또는 None(실패)
    """
    client = _client()
    try:
        order = client.order_market_buy(
            symbol=sym,
            quoteOrderQty=round(notional, 2),
        )
        fills      = order.get("fills", [])
        qty        = float(order["executedQty"])
        avg_price  = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                      / qty) if fills else notional / qty
        entry_price = avg_price * (1 + FEE_RATE)  # 수수료 슬리피지 반영

        log.info(
            f"[BUY ] {sym} | qty={qty:.6f} | avg={avg_price:.4f} "
            f"| entry_price(+fee)={entry_price:.4f} | notional={notional:.2f} USDT"
        )
        return {"qty": qty, "avg_price": avg_price,
                "entry_price": entry_price, "notional": notional}

    except BinanceAPIException as e:
        log.error(f"[BUY ] {sym} 주문 실패: {e}")
        return None


def market_sell(sym: str, qty: float) -> dict | None:
    """
    전량 시장가 매도.
    반환: {"exit_price": ...} 또는 None(실패)
    """
    client = _client()
    info       = get_symbol_info(sym)
    qty_floored = floor_qty(qty, info["step_size"])

    if qty_floored < info["min_qty"]:
        log.error(f"[SELL] {sym} qty={qty_floored} < min_qty={info['min_qty']}")
        return None

    try:
        order     = client.order_market_sell(symbol=sym, quantity=qty_floored)
        fills     = order.get("fills", [])
        exec_qty  = float(order["executedQty"])
        exit_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                      / exec_qty) if fills else 0.0

        log.info(f"[SELL] {sym} | qty={exec_qty:.6f} | exit_price={exit_price:.4f}")
        return {"exit_price": exit_price}

    except BinanceAPIException as e:
        log.error(f"[SELL] {sym} 주문 실패: {e}")
        return None


def get_balance(asset: str = "USDT") -> float:
    """현재 자산 잔액 조회."""
    client = _client()
    try:
        bal = client.get_asset_balance(asset=asset)
        return float(bal["free"]) if bal else 0.0
    except BinanceAPIException as e:
        log.error(f"잔액 조회 실패: {e}")
        return 0.0
