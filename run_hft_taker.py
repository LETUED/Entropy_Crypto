"""
Taker 방향성 전략 페이퍼 트레이더

사용법:
    python run_hft_taker.py
    python run_hft_taker.py --symbol SOLUSDT
    python run_hft_taker.py --symbol SOLUSDT --capital 5000
"""

import argparse
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from hft import config
from hft.simulation.paper_trader_taker import TakerPaperTrader

BANNER = """
===========================================================
  ENTROPY HFT - TAKER DIRECTIONAL (PAPER MODE)
  Signal:  Entropy(timing) + OBI(direction)
  Entry:   Taker @ market price
  Target:  +0.2% | Stop: -0.1% | Max hold: 60s
===========================================================
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",  default=config.SYMBOL,             help="심볼 (SOLUSDT, BTCUSDT ...)")
    p.add_argument("--capital", default=config.TOTAL_CAPITAL_USDT, type=float, help="총 자본 USDT")
    return p.parse_args()


async def main():
    args = parse_args()
    config.SYMBOL             = args.symbol
    config.TOTAL_CAPITAL_USDT = args.capital

    print(BANNER)
    print(f"  심볼:         {config.SYMBOL}")
    print(f"  총 자본:      {config.TOTAL_CAPITAL_USDT:,.0f} USDT")
    print(f"  포지션 크기:  {config.TAKER_POSITION_PCT*100:.0f}% = {config.TOTAL_CAPITAL_USDT * config.TAKER_POSITION_PCT:,.0f} USDT")
    print(f"  최대 동시:    {config.TAKER_MAX_POSITIONS}개")
    print(f"  목표 / 스톱:  +{config.TAKER_TARGET_PCT*100:.1f}% / -{config.TAKER_STOP_PCT*100:.1f}%")
    print(f"  엔트로피:     타깃={config.ADAPTIVE_WINDOW_SECONDS}초  범위={config.ENTROPY_LOWER}~{config.ENTROPY_THRESHOLD}  (윈도우 {config.ADAPTIVE_WINDOW_MIN}~{config.ADAPTIVE_WINDOW_MAX})")
    print(f"  볼륨 게이트:  {config.VOLUME_GATE_MIN_TPS} t/s 미만 진입 차단")
    print(f"  OBI 범위:     {config.OBI_THRESHOLD} ~ {config.OBI_ACTIVATE_MAX}")
    print()

    trader = TakerPaperTrader(symbol=config.SYMBOL)
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\n페이퍼 트레이딩 종료.")
        print(f"\n--- 최종 결과 ---")
        print(f"총 거래:   {trader.manager.trade_count}건")
        print(f"승 / 패:   {trader.manager.wins} / {trader.manager.losses}")
        print(f"승률:      {trader.manager.win_rate*100:.1f}%")
        print(f"실현 PnL:  {trader.manager.total_pnl:+.4f} USDT")


if __name__ == "__main__":
    asyncio.run(main())
