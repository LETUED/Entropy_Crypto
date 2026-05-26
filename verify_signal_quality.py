"""
신호 품질 기록기 (거래 없음 - 순수 데이터 수집)

VER-01: 코인 무관 가설   → 4심볼 TPS/엔트로피/OBI 분포 비교
VER-02: MFE 현실성 검증  → 엔트로피 신호 중 실제 가격 이동 측정
VER-03: OBI 방향 정확도  → OBI 방향 vs 실제 가격 이동 방향

사용법:
    python verify_signal_quality.py
    python verify_signal_quality.py --symbols BTCUSDT SOLUSDT XRPUSDT
    python verify_signal_quality.py --duration 600   # 600초 후 자동 종료
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import websockets

sys.stdout.reconfigure(encoding="utf-8")

from hft import config
from hft.signals.entropy_rt import TradeEntropySignal
from hft.signals.obi import OrderBookImbalance

# ── 설정 ─────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TRACK_SECONDS    = 60                       # 신호 발생 후 추적 시간
SNAP_INTERVALS   = [1, 5, 10, 20, 30, 60]  # 가격 스냅샷 시점 (초)
OUTPUT_DIR       = "docs"
OUTPUT_CSV       = os.path.join(OUTPUT_DIR, "ver_signal_quality.csv")
PRINT_INTERVAL_S = 5.0                      # 콘솔 출력 주기

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    symbol:       str
    start_time:   float
    entry_price:  float
    obi_dir:      str    # "buy" | "sell" | "neutral"
    tps:          float
    window:       int
    entropy:      float
    price_history: list[tuple[float, float]] = field(default_factory=list)  # (t_offset, price)
    end_time:     Optional[float] = None
    exit_reason:  str = ""

    def feed(self, price: float) -> None:
        offset = time.time() - self.start_time
        self.price_history.append((offset, price))

    def is_expired(self) -> bool:
        return (time.time() - self.start_time) >= TRACK_SECONDS

    def finalize(self, exit_reason: str) -> None:
        self.end_time   = time.time()
        self.exit_reason = exit_reason

    # MFE / MAE 계산 (OBI 방향 기준)
    def _signed_move(self, price: float) -> float:
        """OBI 방향이 'buy'면 상승이 양수, 'sell'이면 하락이 양수"""
        if self.entry_price == 0:
            return 0.0
        raw = (price - self.entry_price) / self.entry_price * 100
        return raw if self.obi_dir != "sell" else -raw

    def mfe(self) -> float:
        """Max Favorable Excursion (%)"""
        if not self.price_history:
            return 0.0
        return max(self._signed_move(p) for _, p in self.price_history)

    def mae(self) -> float:
        """Max Adverse Excursion (%)"""
        if not self.price_history:
            return 0.0
        return min(self._signed_move(p) for _, p in self.price_history)

    def direction_correct(self) -> Optional[bool]:
        """60s 후 가격이 OBI 방향으로 움직였는가?"""
        final = [p for t, p in self.price_history if t >= 30]
        if not final:
            return None
        return self._signed_move(final[-1]) > 0

    def snap(self, interval_s: int) -> Optional[float]:
        """특정 시점의 signed move (%)"""
        candidates = [(abs(t - interval_s), p) for t, p in self.price_history]
        if not candidates:
            return None
        _, price = min(candidates, key=lambda x: x[0])
        return round(self._signed_move(price), 4)

    def to_row(self) -> dict:
        row = {
            "symbol":      self.symbol,
            "time":        time.strftime("%H:%M:%S", time.localtime(self.start_time)),
            "entry_price": round(self.entry_price, 6),
            "obi_dir":     self.obi_dir,
            "tps":         round(self.tps, 2),
            "window":      self.window,
            "entropy":     round(self.entropy, 4),
            "exit_reason": self.exit_reason,
            "duration_s":  round((self.end_time or time.time()) - self.start_time, 1),
            "mfe_pct":     round(self.mfe(), 4),
            "mae_pct":     round(self.mae(), 4),
            "dir_correct": self.direction_correct(),
        }
        for iv in SNAP_INTERVALS:
            row[f"move_t{iv}s"] = self.snap(iv)
        return row


# ── 심볼 모니터 ───────────────────────────────────────────────────────────────

class SymbolMonitor:
    def __init__(self, symbol: str):
        self.symbol  = symbol.upper()
        self.entropy = TradeEntropySignal(
            threshold      = config.ENTROPY_THRESHOLD,
            target_seconds = config.ADAPTIVE_WINDOW_SECONDS,
            min_window     = config.ADAPTIVE_WINDOW_MIN,
            max_window     = config.ADAPTIVE_WINDOW_MAX,
        )
        self.obi = OrderBookImbalance(config.OBI_BOOK_LEVELS, config.OBI_THRESHOLD)

        self._current_signal: Optional[SignalRecord] = None
        self._completed: list[SignalRecord]          = []
        self._last_price: float = 0.0

        # VER-01 분포 통계
        self.tps_samples: list[float]     = []
        self.entropy_samples: list[float] = []
        self.obi_samples: list[float]     = []
        self.signal_count: int            = 0
        self.tick_count: int              = 0
        self.start_time: float            = time.time()

    def on_trade(self, data: dict) -> None:
        self._last_price = float(data["p"])
        self.entropy.update(bool(data["m"]))
        self.tick_count += 1

        tps = self.entropy.tps
        if tps > 0:
            self.tps_samples.append(tps)
        self.entropy_samples.append(self.entropy.entropy)

        # 추적 중인 신호에 가격 피드
        if self._current_signal:
            self._current_signal.feed(self._last_price)
            if self._current_signal.is_expired():
                self._close_signal("timeout")

    def on_depth(self, data: dict) -> list[SignalRecord]:
        """오더북 업데이트 → 신호 개폐 판단. 완료된 레코드 반환."""
        self.obi.update_book(data.get("bids", []), data.get("asks", []))
        mid = self.obi.mid_price
        if mid > 0:
            self._last_price = mid
        self.obi_samples.append(self.obi.obi)

        is_signal = self.entropy.is_signal
        tps_ok    = self.entropy.tps >= config.VOLUME_GATE_MIN_TPS
        obi_abs   = abs(self.obi.obi)
        in_range  = config.OBI_THRESHOLD < obi_abs <= config.OBI_ACTIVATE_MAX

        # 신호 열기
        if is_signal and tps_ok and in_range and self._current_signal is None:
            self._current_signal = SignalRecord(
                symbol      = self.symbol,
                start_time  = time.time(),
                entry_price = self._last_price,
                obi_dir     = self.obi.direction,
                tps         = self.entropy.tps,
                window      = self.entropy.current_window,
                entropy     = self.entropy.entropy,
            )
            self.signal_count += 1

        # 신호 닫기 (엔트로피 정상화)
        elif not is_signal and self._current_signal:
            self._close_signal("entropy_normalized")

        completed, self._completed = self._completed, []
        return completed

    def _close_signal(self, reason: str) -> None:
        if self._current_signal:
            self._current_signal.finalize(reason)
            self._completed.append(self._current_signal)
            self._current_signal = None

    def ver01_stats(self) -> dict:
        """VER-01: 분포 통계"""
        elapsed = time.time() - self.start_time
        return {
            "symbol":         self.symbol,
            "elapsed_s":      round(elapsed, 0),
            "tick_count":     self.tick_count,
            "avg_tps":        round(np.mean(self.tps_samples), 2) if self.tps_samples else 0,
            "std_tps":        round(np.std(self.tps_samples), 2) if self.tps_samples else 0,
            "signal_count":   self.signal_count,
            "signal_rate":    round(self.signal_count / max(elapsed, 1) * 60, 2),  # per min
            "avg_entropy":    round(np.mean(self.entropy_samples), 4) if self.entropy_samples else 1.0,
            "pct_below_thr":  round(np.mean([e < config.ENTROPY_THRESHOLD for e in self.entropy_samples]) * 100, 1) if self.entropy_samples else 0,
            "avg_obi":        round(np.mean(self.obi_samples), 4) if self.obi_samples else 0,
            "std_obi":        round(np.std(self.obi_samples), 4) if self.obi_samples else 0,
        }


# ── CSV 기록기 ────────────────────────────────────────────────────────────────

class CsvWriter:
    FIELDNAMES = (
        ["symbol", "time", "entry_price", "obi_dir", "tps", "window",
         "entropy", "exit_reason", "duration_s", "mfe_pct", "mae_pct", "dir_correct"]
        + [f"move_t{iv}s" for iv in SNAP_INTERVALS]
    )

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f   = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._f, fieldnames=self.FIELDNAMES)
        self._csv.writeheader()
        self._f.flush()

    def write(self, record: SignalRecord) -> None:
        self._csv.writerow(record.to_row())
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── 요약 출력 ─────────────────────────────────────────────────────────────────

def print_ver01(monitors: dict[str, SymbolMonitor]) -> None:
    print("\n" + "=" * 80)
    print(f"  VER-01  코인 무관 가설  [{time.strftime('%H:%M:%S')}]")
    print(f"  {'심볼':<10} {'TPS(avg)':<10} {'TPS(std)':<10} {'엔트로피':<10} {'신호%':<8} {'신호/분':<8} {'OBI(std)'}")
    print("-" * 80)
    for s in monitors.values():
        st = s.ver01_stats()
        print(
            f"  {st['symbol']:<10} "
            f"{st['avg_tps']:>7.2f}   "
            f"{st['std_tps']:>7.2f}   "
            f"{st['avg_entropy']:>7.4f}   "
            f"{st['pct_below_thr']:>5.1f}%  "
            f"{st['signal_rate']:>6.2f}   "
            f"{st['std_obi']:>6.4f}"
        )
    print("=" * 80)


def print_ver02_03(completed: list[SignalRecord]) -> None:
    if not completed:
        return
    mfe_vals   = [r.mfe() for r in completed]
    mae_vals   = [r.mae() for r in completed]
    dir_flags  = [r.direction_correct() for r in completed if r.direction_correct() is not None]

    print(f"\n  VER-02  MFE 분석  (총 {len(completed)}건)")
    print(f"    MFE 중위값:  {np.median(mfe_vals):+.4f}%")
    print(f"    MFE p75:     {np.percentile(mfe_vals, 75):+.4f}%")
    print(f"    MFE p90:     {np.percentile(mfe_vals, 90):+.4f}%")
    print(f"    MAE 중위값:  {np.median(mae_vals):+.4f}%")
    print(f"    0.2% 달성률: {np.mean([m >= 0.2 for m in mfe_vals])*100:.1f}%")
    print(f"    0.1% 달성률: {np.mean([m >= 0.1 for m in mfe_vals])*100:.1f}%")
    if dir_flags:
        print(f"\n  VER-03  OBI 방향 정확도")
        print(f"    방향 맞음:   {np.mean(dir_flags)*100:.1f}% (n={len(dir_flags)}, 기준=30초 후 가격)")
        print(f"    vs 랜덤 55%: {'✓ 유의미' if np.mean(dir_flags) > 0.55 else '✗ 랜덤 수준'}")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def run(symbols: list[str], duration: Optional[int]) -> None:
    syms_lower = [s.lower() for s in symbols]
    streams    = []
    for sym in syms_lower:
        streams.append(f"{sym}@aggTrade")
        streams.append(f"{sym}@depth20@100ms")

    url = f"{config.WS_BASE}?streams=" + "/".join(streams)
    monitors = {s.upper(): SymbolMonitor(s) for s in symbols}
    writer   = CsvWriter(OUTPUT_CSV)
    all_completed: list[SignalRecord] = []
    last_print = 0.0
    start      = time.time()

    print(f"\n연결 중... {len(symbols)}개 심볼 동시 모니터링")
    print(f"심볼: {', '.join(s.upper() for s in symbols)}")
    print(f"출력: {OUTPUT_CSV}")
    print(f"{'─'*60}")

    try:
        async with websockets.connect(url, ping_interval=20) as ws:
            print("연결됨. 데이터 수집 시작 (Ctrl+C로 종료)\n")
            async for raw in ws:
                msg    = json.loads(raw)
                stream = msg.get("stream", "")
                data   = msg.get("data", {})
                sym    = stream.split("@")[0].upper()
                mon    = monitors.get(sym)
                if not mon:
                    continue

                if "aggTrade" in stream:
                    mon.on_trade(data)
                elif "depth" in stream:
                    completed = mon.on_depth(data)
                    for rec in completed:
                        writer.write(rec)
                        all_completed.append(rec)

                now = time.time()
                if now - last_print >= PRINT_INTERVAL_S:
                    print_ver01(monitors)
                    print_ver02_03(all_completed)
                    last_print = now

                if duration and (now - start) >= duration:
                    print(f"\n{duration}초 완료. 자동 종료.")
                    break

    except KeyboardInterrupt:
        print("\n\n수동 종료.")

    writer.close()
    print(f"\n{'='*60}")
    print(f"  최종 결과 ({time.strftime('%H:%M:%S')})")
    print(f"{'='*60}")
    print_ver01(monitors)
    if all_completed:
        print_ver02_03(all_completed)
        print(f"  CSV 저장: {OUTPUT_CSV}  ({len(all_completed)}건)")
    else:
        print("  신호 없음 (볼륨 부족 또는 시간 부족)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS, help="심볼 리스트")
    p.add_argument("--duration", type=int,  default=None,            help="자동 종료 초 (미지정 시 Ctrl+C)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.symbols, args.duration))
