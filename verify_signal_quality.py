"""
신호 품질 기록기 v2 — 신호 후 30분 추적

핵심 질문:
  "틱 엔트로피 신호 진입 후 5~30분 들고 있으면 수익이 나는가?"

측정 항목:
  - 신호 중 MFE (기존 VER-02)
  - 신호 종료 후 t+2m/5m/10m/20m/30m 가격 이동 (신규)
  - OBI 방향 정확도 (VER-03)
  - 심볼별 미세구조 비교 (VER-01)

사용법:
    python verify_signal_quality.py --duration 3600
    python verify_signal_quality.py --symbols BTCUSDT ETHUSDT --duration 7200
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

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRACK_SECONDS   = 1800              # 신호 시작 후 최대 추적 시간 (30분)
SNAP_INTRA      = [1, 5, 10, 30, 60]           # 신호 중 스냅샷
SNAP_POST       = [120, 300, 600, 1200, 1800]  # 신호 후 2m/5m/10m/20m/30m
SNAP_ALL        = SNAP_INTRA + SNAP_POST
OUTPUT_DIR      = "docs"
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, "ver_signal_quality_v2.csv")
PRINT_INTERVAL  = 30.0             # 콘솔 출력 주기 (초)

# ── 신호 레코드 ───────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    symbol:       str
    sig_id:       int
    start_time:   float
    entry_price:  float
    obi_dir:      str    # "buy" | "sell"
    tps:          float
    window:       int
    entropy:      float
    vol_pre60:    float = 0.0   # 신호 직전 60초 가격변동폭 % (완전 ex-ante)

    price_history: list[tuple[float, float]] = field(default_factory=list)

    # 신호 종료 정보 (엔트로피 정상화 시점)
    signal_end_offset: Optional[float] = None
    signal_end_price:  Optional[float] = None
    exit_reason:       str = ""

    def feed(self, price: float) -> None:
        offset = time.time() - self.start_time
        self.price_history.append((offset, price))

    def mark_signal_end(self, price: float, reason: str) -> None:
        if self.signal_end_offset is None:
            self.signal_end_offset = time.time() - self.start_time
            self.signal_end_price  = price
            self.exit_reason       = reason

    def is_collection_complete(self) -> bool:
        return (time.time() - self.start_time) >= TRACK_SECONDS

    # ── 분석 헬퍼 ────────────────────────────────────────────────────────────

    def _signed_move(self, price: float) -> float:
        """OBI 방향 기준 가격 이동 (%)"""
        if self.entry_price == 0:
            return 0.0
        raw = (price - self.entry_price) / self.entry_price * 100
        return raw if self.obi_dir != "sell" else -raw

    def snap(self, interval_s: int) -> Optional[float]:
        """특정 시점 signed move (%)"""
        candidates = [(abs(t - interval_s), p) for t, p in self.price_history]
        if not candidates:
            return None
        _, price = min(candidates, key=lambda x: x[0])
        return round(self._signed_move(price), 4)

    def mfe(self) -> float:
        if not self.price_history:
            return 0.0
        end = self.signal_end_offset or float("inf")
        intra = [p for t, p in self.price_history if t <= end]
        if not intra:
            return 0.0
        return max(self._signed_move(p) for p in intra)

    def to_row(self) -> dict:
        row = {
            "symbol":           self.symbol,
            "sig_id":           self.sig_id,
            "time":             time.strftime("%H:%M:%S", time.localtime(self.start_time)),
            "entry_price":      round(self.entry_price, 6),
            "obi_dir":          self.obi_dir,
            "tps":              round(self.tps, 1),
            "window":           self.window,
            "entropy":          round(self.entropy, 4),
            "vol_pre60":        round(self.vol_pre60, 4),
            "signal_duration_s":round(self.signal_end_offset, 1) if self.signal_end_offset else None,
            "exit_reason":      self.exit_reason,
            "mfe_intra":        round(self.mfe(), 4),
        }
        for iv in SNAP_ALL:
            row[f"t{iv}s"] = self.snap(iv)
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

        self._active: Optional[SignalRecord] = None  # 엔트로피 활성 신호
        self._tracking: list[SignalRecord]   = []    # 종료됐지만 30분 추적 중
        self._completed: list[SignalRecord]  = []    # 30분 추적 완료

        self._last_price: float = 0.0
        self._sig_counter: int  = 0
        self._obi_streak: int   = 0
        self._obi_streak_dir    = ""
        self._price_history: list[tuple[float, float]] = []  # (time, price)
        self.trend_filter: bool  = False
        self.vol_gate_pct: float = 0.0   # 0 = 비활성

        # VER-01 통계
        self.tick_count:      int         = 0
        self.signal_count:    int         = 0
        self.tps_samples:     list[float] = []
        self.entropy_samples: list[float] = []
        self.obi_samples:     list[float] = []
        self.start_time:      float       = time.time()

    def _trend_direction(self) -> str:
        """최근 30초 가격 기울기로 추세 방향 반환"""
        now = time.time()
        cutoff = now - 30.0
        recent = [(t, p) for t, p in self._price_history if t >= cutoff]
        if len(recent) < 5:
            return "neutral"
        prices = [p for _, p in recent]
        return "buy" if prices[-1] > prices[0] else "sell"

    def _vol_range_pct(self) -> float:
        """최근 60초 가격 변동폭 %"""
        now = time.time()
        cutoff = now - 60.0
        recent = [p for t, p in self._price_history if t >= cutoff]
        if len(recent) < 5:
            return 0.0
        return (max(recent) - min(recent)) / min(recent) * 100

    def on_trade(self, data: dict) -> None:
        self._last_price = float(data["p"])
        self._price_history.append((time.time(), self._last_price))
        self.entropy.update(bool(data["m"]))
        self.tick_count += 1
        self.entropy_samples.append(self.entropy.entropy)
        if self.entropy.tps > 0:
            self.tps_samples.append(self.entropy.tps)

        # 모든 추적 중 레코드에 가격 피드
        if self._active:
            self._active.feed(self._last_price)
        for rec in self._tracking:
            rec.feed(self._last_price)

    def on_depth(self, data: dict) -> list[SignalRecord]:
        """완료된 레코드 반환"""
        self.obi.update_book(data.get("bids", []), data.get("asks", []))
        mid = self.obi.mid_price
        if mid > 0:
            self._last_price = mid
        self.obi_samples.append(self.obi.obi)

        price     = self._last_price
        is_signal = config.ENTROPY_LOWER < self.entropy.entropy < config.ENTROPY_THRESHOLD
        tps_ok    = self.entropy.tps >= config.VOLUME_GATE_MIN_TPS
        obi_abs   = abs(self.obi.obi)
        in_range  = config.OBI_THRESHOLD < obi_abs <= config.OBI_ACTIVATE_MAX
        direction = self.obi.direction

        # OBI 지속성 추적
        if direction != "neutral" and in_range:
            if direction == self._obi_streak_dir:
                self._obi_streak += 1
            else:
                self._obi_streak     = 1
                self._obi_streak_dir = direction
        else:
            self._obi_streak     = 0
            self._obi_streak_dir = ""

        persist_ok = self._obi_streak >= config.OBI_PERSIST_MIN
        trend_ok   = (not self.trend_filter) or (self._trend_direction() == self._obi_streak_dir)
        vol_ok     = (self.vol_gate_pct <= 0) or (self._vol_range_pct() >= self.vol_gate_pct)

        # 새 신호 열기
        if is_signal and tps_ok and in_range and persist_ok and trend_ok and vol_ok and self._active is None:
            self._sig_counter += 1
            self._active = SignalRecord(
                symbol      = self.symbol,
                sig_id      = self._sig_counter,
                start_time  = time.time(),
                entry_price = price,
                obi_dir     = self._obi_streak_dir,
                tps         = self.entropy.tps,
                window      = self.entropy.current_window,
                entropy     = self.entropy.entropy,
                vol_pre60   = self._vol_range_pct(),
            )
            self.signal_count += 1

        # 신호 종료 (엔트로피 정상화)
        elif not is_signal and self._active:
            self._active.mark_signal_end(price, "entropy_normalized")
            self._tracking.append(self._active)
            self._active = None

        # 30분 추적 완료된 것 수확
        done = [r for r in self._tracking if r.is_collection_complete()]
        for r in done:
            self._tracking.remove(r)
            self._completed.append(r)

        completed, self._completed = self._completed, []
        return completed

    def ver01_stats(self) -> dict:
        elapsed = max(time.time() - self.start_time, 1)
        return {
            "symbol":        self.symbol,
            "elapsed_m":     round(elapsed / 60, 1),
            "avg_tps":       round(np.mean(self.tps_samples), 1) if self.tps_samples else 0,
            "avg_entropy":   round(np.mean(self.entropy_samples), 4) if self.entropy_samples else 1.0,
            "signal_count":  self.signal_count,
            "signal_per_h":  round(self.signal_count / elapsed * 3600, 1),
        }


# ── CSV 기록기 ────────────────────────────────────────────────────────────────

class CsvWriter:
    FIELDS = (
        ["symbol", "sig_id", "time", "entry_price", "obi_dir",
         "tps", "window", "entropy", "vol_pre60",
         "signal_duration_s", "exit_reason", "mfe_intra"]
        + [f"t{iv}s" for iv in SNAP_ALL]
    )

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f   = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._csv.writeheader()
        self._f.flush()

    def write(self, rec: SignalRecord) -> None:
        self._csv.writerow(rec.to_row())
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── 콘솔 출력 ────────────────────────────────────────────────────────────────

def print_status(monitors: dict[str, SymbolMonitor], completed: list[SignalRecord]) -> None:
    print(f"\n{'='*70}  [{time.strftime('%H:%M:%S')}]")

    # VER-01
    print(f"  {'심볼':<10} {'TPS':>6} {'엔트로피':>8} {'신호수':>6} {'신호/시':>8}")
    print(f"  {'-'*45}")
    for m in monitors.values():
        s = m.ver01_stats()
        print(f"  {s['symbol']:<10} {s['avg_tps']:>6.1f} {s['avg_entropy']:>8.4f} "
              f"{s['signal_count']:>6} {s['signal_per_h']:>8.1f}")

    if not completed:
        print(f"\n  아직 30분 추적 완료 없음 (활성 신호 추적 중)")
        print()
        return

    # VER-02: 신호 중 MFE
    mfes = [r.mfe() for r in completed]
    print(f"\n  [VER-02] 신호 중 MFE  (n={len(completed)})")
    print(f"  중위: {np.median(mfes):+.4f}%  p75: {np.percentile(mfes,75):+.4f}%  p90: {np.percentile(mfes,90):+.4f}%")
    print(f"  0.1% 달성: {np.mean([m>=0.1 for m in mfes])*100:.1f}%  "
          f"수수료(Taker 0.10%) 기준선")

    # 신호 후 지속성 — 핵심
    print(f"\n  [핵심] 신호 진입 후 시간별 이동 (OBI 방향 기준)")
    print(f"  {'시점':<12} {'중위':>8} {'p75':>8} {'p90':>8} {'승률':>8}  {'의미'}")
    print(f"  {'-'*60}")

    fee_taker = 0.10
    fee_maker = 0.04

    intervals = [(iv, f"t{iv}s") for iv in SNAP_ALL]
    for iv, col in intervals:
        vals = [r.snap(iv) for r in completed if r.snap(iv) is not None]
        if not vals:
            continue
        p50 = np.median(vals)
        p75 = np.percentile(vals, 75)
        p90 = np.percentile(vals, 90)
        wr  = np.mean([v > 0 for v in vals]) * 100

        if iv <= 60:
            tag = "(신호 중)"
        elif iv == 120:
            tag = "← 2분"
        elif iv == 300:
            tag = "← 5분  ★"
        elif iv == 600:
            tag = "← 10분 ★"
        elif iv == 1200:
            tag = "← 20분"
        else:
            tag = "← 30분"

        marker = ""
        if p90 >= fee_taker:
            marker = " [Taker 가능]"
        elif p90 >= fee_maker:
            marker = " [Maker 가능]"

        label = f"{iv}초" if iv < 60 else f"{iv//60}분"
        print(f"  {label:<12} {p50:>+8.4f}% {p75:>+8.4f}% {p90:>+8.4f}%  {wr:>6.1f}%  {tag}{marker}")

    # 신호 지속 시간
    durations = [r.signal_end_offset for r in completed if r.signal_end_offset]
    if durations:
        print(f"\n  신호 평균 지속: {np.mean(durations):.1f}s  "
              f"중위: {np.median(durations):.1f}s  "
              f"최대: {max(durations):.1f}s")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def _ws_loop(url, monitors, writer, all_completed, start, duration,
                   last_print_ref) -> bool:
    """단일 WebSocket 세션. True=정상종료, False=재연결 필요"""
    import websockets.exceptions
    try:
        async with websockets.connect(url, ping_interval=20, open_timeout=30) as ws:
            print(f"연결됨. ({time.strftime('%H:%M:%S')})\n")
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
                if now - last_print_ref[0] >= PRINT_INTERVAL:
                    print_status(monitors, all_completed)
                    last_print_ref[0] = now

                if duration and (now - start) >= duration:
                    print(f"\n{duration}초 완료.")
                    return True
        return True
    except (websockets.exceptions.ConnectionClosedError,
            ConnectionResetError, OSError) as e:
        print(f"\n  [재연결] {e}")
        return False


async def run(symbols: list[str], duration: Optional[int],
              trend_filter: bool = False, vol_gate: float = 0.0) -> None:
    syms_lower = [s.lower() for s in symbols]
    streams    = []
    for sym in syms_lower:
        streams.append(f"{sym}@aggTrade")
        streams.append(f"{sym}@depth20@100ms")

    url      = f"{config.WS_BASE}?streams=" + "/".join(streams)
    monitors = {s.upper(): SymbolMonitor(s) for s in symbols}
    for mon in monitors.values():
        mon.trend_filter = trend_filter
        mon.vol_gate_pct = vol_gate
    writer   = CsvWriter(OUTPUT_CSV)

    all_completed: list[SignalRecord] = []
    last_print_ref = [0.0]
    start      = time.time()

    print(f"\n틱 신호 → 30분 추적 검증기")
    print(f"심볼: {', '.join(s.upper() for s in symbols)}")
    print(f"엔트로피 범위: {config.ENTROPY_LOWER} ~ {config.ENTROPY_THRESHOLD}")
    print(f"볼륨 게이트: TPS ≥ {config.VOLUME_GATE_MIN_TPS}")
    print(f"OBI 지속성: {config.OBI_PERSIST_MIN}회  추세필터: {'ON' if trend_filter else 'OFF'}  변동성게이트: {vol_gate:.2f}%")
    print(f"출력: {OUTPUT_CSV}")
    print(f"{'─'*60}")
    print(f"신호 발생 → 30분 추적 → CSV 저장 (Ctrl+C로 종료)\n")

    try:
        while True:
            done = await _ws_loop(url, monitors, writer, all_completed,
                                  start, duration, last_print_ref)
            if done:
                break
            if duration and (time.time() - start) >= duration:
                break
            await asyncio.sleep(5)
            print(f"  재연결 시도... ({time.strftime('%H:%M:%S')})")
    except KeyboardInterrupt:
        print("\n\n수동 종료.")

    writer.close()
    print(f"\n{'='*70}")
    print(f"  최종 결과  ({time.strftime('%H:%M:%S')})")
    print_status(monitors, all_completed)

    w2 = CsvWriter(OUTPUT_CSV.replace(".csv", "_partial.csv"))
    for mon in monitors.values():
        for rec in mon._tracking + ([] if mon._active is None else [mon._active]):
            rec.mark_signal_end(mon._last_price, "session_end")
            w2.write(rec)
    w2.close()

    total = len(all_completed)
    print(f"  완료 저장: {OUTPUT_CSV} ({total}건)")
    print(f"  미완료 저장: {OUTPUT_CSV.replace('.csv','_partial.csv')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--duration", type=int,  default=None)
    p.add_argument("--persist",  type=int,  default=None,  help="OBI 지속성 N회 (기본: config값)")
    p.add_argument("--output",   type=str,  default=None,  help="CSV 출력 경로")
    p.add_argument("--trend",    action="store_true",       help="추세 필터 활성화")
    p.add_argument("--vol-gate", type=float, default=0.0,  help="최근 60초 변동폭 최소 %% (0=비활성)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.persist is not None:
        config.OBI_PERSIST_MIN = args.persist
    if args.output is not None:
        OUTPUT_CSV = args.output
    asyncio.run(run(args.symbols, args.duration, trend_filter=args.trend,
                    vol_gate=args.vol_gate))
