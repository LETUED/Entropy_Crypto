"""
라이브 트레이딩 메인 루프.

매시간 크론으로 호출:
  Linux:  0 * * * * cd /path/to/Entropy_Crypto && python -m live.main
  Windows Task Scheduler: 동일 명령어 매시간

각 실행 사이클:
  1. 상태 로드
  2. Earn 포지션 갱신 (누적 이자 업데이트)
  3. 열린 포지션 청산 조건 체크 → 청산 시 Earn 재예치
  4. 빈 슬롯 진입 신호 체크 → 진입 전 Earn 인출
  5. 상태 저장 + 비교 로그 (트레이딩 PnL vs 대기자본 이자)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from live.config import COINS, COIN_CAPITAL, TOTAL_CAPITAL, MIN_NOTIONAL, DRY_RUN
from live.data_feed import fetch_ohlcv, fetch_funding_rates, fetch_fear_greed
from live.signal_engine import check_signal, check_exit
from live.portfolio import (
    load_state, save_state,
    open_position, close_position,
    has_position, get_held_hours,
    calc_position_size,
)
from live.executor import market_buy, market_sell, get_balance
from live.idle_capital import (
    subscribe, redeem, update_earn_state, log_comparison
)
from live.logger_setup import get_logger

log = get_logger()


def run_cycle() -> None:
    now = datetime.now(timezone.utc)
    log.info("=" * 60)
    mode_tag = "[DRY RUN] " if DRY_RUN else ""
    log.info(f"{mode_tag}사이클 시작: {now.strftime('%Y-%m-%d %H:%M UTC')}")

    # ── 온체인 데이터 수집 ─────────────────────────────────────────────────
    funding_s = fetch_funding_rates()
    fg_s      = fetch_fear_greed()

    state = load_state()

    # ── 동적 자본 계산 (복리 자동화) ───────────────────────────────────────
    free_usdt  = get_balance("USDT")
    earn_usdt  = state.get("earn_subscribed", 0.0)
    pos_value  = sum(p["notional"] for p in state["positions"].values())
    actual_total = free_usdt + earn_usdt + pos_value
    # 실제 잔고가 설정값보다 크면 복리 반영, 작으면 설정값 유지
    total_capital = actual_total if actual_total >= TOTAL_CAPITAL else TOTAL_CAPITAL
    coin_capital  = total_capital / len(COINS)
    log.info(
        f"[CAPITAL] 총 {total_capital:.2f} USDT "
        f"(현물 {free_usdt:.2f} + Earn {earn_usdt:.2f} + 포지션 {pos_value:.2f}) "
        f"→ 코인당 {coin_capital:.2f} USDT"
    )

    # ── 1. Earn 포지션 갱신 (이자 누적 반영) ──────────────────────────────
    if not DRY_RUN:
        state = update_earn_state(state)

    # ── 2. 청산 체크 ────────────────────────────────────────────────────────
    for sym in list(state["positions"].keys()):
        try:
            df     = fetch_ohlcv(sym)
            held_h = get_held_hours(state, sym)
            should_exit, reason = check_exit(state["positions"][sym], df, held_h)

            if should_exit:
                log.info(f"[EXIT] {sym} 청산 시도 | 이유: {reason} | 보유: {held_h}H")
                if DRY_RUN:
                    log.info(f"[DRY RUN] {sym} 매도 건너뜀")
                    continue

                pos    = state["positions"][sym]
                result = market_sell(sym, pos["qty"])
                if result:
                    state = close_position(state, sym, result["exit_price"], reason)
                    # 청산 금액 즉시 Earn 재예치
                    recovered = pos["notional"] * (1 + (result["exit_price"] - pos["entry_price"]) / pos["entry_price"])
                    subscribe(recovered)
                else:
                    log.error(f"[EXIT] {sym} 청산 실패 — 다음 사이클 재시도")
            else:
                pos = state["positions"][sym]
                cur_price  = float(df["close"].iloc[-2])
                unrealized = (cur_price - pos["entry_price"]) / pos["entry_price"] * 100
                log.info(
                    f"[HOLD] {sym} | 진입 {pos['entry_price']:.4f} → 현재 {cur_price:.4f} "
                    f"| 미실현 {unrealized:+.3f}% | 보유 {held_h}H"
                )
        except Exception as e:
            log.error(f"청산 체크 오류 [{sym}]: {e}", exc_info=True)

    # ── 3. 진입 체크 ────────────────────────────────────────────────────────
    for sym in COINS:
        if has_position(state, sym):
            log.info(f"[SKIP] {sym} 이미 포지션 보유")
            continue

        try:
            df     = fetch_ohlcv(sym)
            signal, kelly, debug = check_signal(df, funding_s, fg_s, sym)

            log.info(
                f"[CHECK] {sym} | price={debug.get('price', 0):.4f} "
                f"| RSI={debug.get('rsi')} {'OK' if debug.get('rsi_ok') else 'X'} "
                f"| MPE_pct={debug.get('mpe_pct')}% {'OK' if debug.get('mpe_ok') else 'X'} "
                f"| MA200 {'OK' if debug.get('above_ma') else 'X'} "
                f"| OC {'OK' if debug.get('oc_ok') else 'X'} "
                f"| {'★ 진입!' if signal else '대기'}"
            )

            if signal:
                notional = calc_position_size(kelly, coin_capital)
                if notional < MIN_NOTIONAL:
                    log.warning(f"[SKIP] {sym} 최소 주문 미달 ({notional:.2f} < {MIN_NOTIONAL} USDT)")
                    continue

                if DRY_RUN:
                    log.info(
                        f"[DRY RUN] {sym} 매수 건너뜀 | Kelly={kelly*100:.0f}% "
                        f"| {notional:.2f} USDT"
                    )
                    continue

                # Earn에서 필요 금액 인출
                usdt_free = get_balance("USDT")
                if usdt_free < notional:
                    shortfall = notional - usdt_free
                    log.info(f"[EARN] {shortfall:.2f} USDT 인출 (Earn → Spot)")
                    if not redeem(shortfall):
                        log.error(f"[SKIP] {sym} Earn 인출 실패")
                        continue

                # 매수
                log.info(f"[ENTRY] {sym} | Kelly={kelly*100:.0f}% | {notional:.2f} USDT")
                result = market_buy(sym, notional)
                if result:
                    state = open_position(
                        state, sym,
                        qty=result["qty"],
                        entry_price=result["entry_price"],
                        kelly_frac=kelly,
                        notional=notional,
                    )

        except Exception as e:
            log.error(f"진입 체크 오류 [{sym}]: {e}", exc_info=True)

    # ── 4. 여유 USDT → Earn 자동 예치 ────────────────────────────────────
    if not DRY_RUN:
        free_usdt = get_balance("USDT")
        if free_usdt >= 0.01:
            log.info(f"[EARN] 여유 USDT {free_usdt:.4f} 자동 예치")
            subscribe(free_usdt)
            state = update_earn_state(state)

    # ── 5. 상태 저장 + 비교 로그 ──────────────────────────────────────────
    save_state(state)

    open_pos = len(state["positions"])
    log.info(
        f"사이클 완료 | 열린 포지션: {open_pos}개 | "
        f"누적 거래: {state['trade_count']}건"
    )
    log_comparison(state)
    log.info("=" * 60)


if __name__ == "__main__":
    run_cycle()
