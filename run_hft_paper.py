"""
엔트로피 HFT 페이퍼 트레이더 실행

실제 돈 없음 — Binance WebSocket 실시간 데이터로 전략 검증만 수행

사용법:
    python run_hft_paper.py
    python run_hft_paper.py --symbol SOLUSDT
    python run_hft_paper.py --symbol ETHUSDT --capital 5000
"""

import argparse
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from hft import config
from hft.simulation.paper_trader import PaperTrader

BANNER = """
===========================================================
  ENTROPY HFT - PAPER TRADING MODE
  Segments: 5  |  Stop Loss: 2%  |  No Real Orders
  Signal: Entropy(timing) + OBI(direction) -> Maker Quotes
===========================================================
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy HFT Paper Trader")
    parser.add_argument("--symbol",  default=config.SYMBOL,              help="거래 심볼 (예: BTCUSDT, SOLUSDT)")
    parser.add_argument("--capital", default=config.TOTAL_CAPITAL_USDT,  type=float, help="총 가상 자본 USDT")
    return parser.parse_args()


async def main():
    args = parse_args()

    # 런타임 파라미터 오버라이드
    config.SYMBOL               = args.symbol
    config.TOTAL_CAPITAL_USDT   = args.capital
    config.CAPITAL_PER_SEGMENT  = args.capital / config.N_SEGMENTS

    print(BANNER)
    print(f"  심볼:       {config.SYMBOL}")
    print(f"  총 자본:    {config.TOTAL_CAPITAL_USDT:,.0f} USDT")
    print(f"  세그먼트:   {config.N_SEGMENTS}개 × {config.CAPITAL_PER_SEGMENT:,.0f} USDT")
    print(f"  스톱로스:   {config.STOP_LOSS_PCT*100:.0f}%")
    print(f"  스프레드:   ±{config.MAKER_SPREAD_PCT*100:.3f}%")
    print(f"  엔트로피:   윈도우={config.ENTROPY_WINDOW} / 임계값={config.ENTROPY_THRESHOLD}")
    print(f"  OBI:        레벨={config.OBI_BOOK_LEVELS} / 임계값={config.OBI_THRESHOLD}")
    print()

    trader = PaperTrader(symbol=config.SYMBOL)
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\n\n페이퍼 트레이딩 종료.")
        mid = trader.obi.mid_price
        print(f"최종 실현 PnL: {trader.manager.total_realized_pnl:+.4f} USDT")
        print(f"최종 미실현 PnL: {trader.manager.total_unrealized_pnl(mid):+.4f} USDT")
        print(f"스톱로스 발동:  {trader.manager.stopped_count}회")
        print(f"총 거래 수신:   {trader._trade_count:,}건")


if __name__ == "__main__":
    asyncio.run(main())
