"""
실험 32: 시간 확장 (2017~2020 추가) — 코인간 분산 없이 독립 표본 늘려 유의성 정면 공략

프로젝트 의의(할루시네이션 닻): "잃지 않는 게 돈을 버는 거야" 자본보존 최우선.
  MPE=타이밍 필터(예측가능 국면에서만 진입), 방향은 RSI. 입증된 가치는 낙폭방어(-1% vs B&H -77%).

배경(Exp30/31A/31C 수렴 결론):
  엣지 실재 가능성 有(품질코인 gross +142bps)이나 2021~2025 표본서 통계 유의 미달(n=37, CI에 0포함).
  코인 확장(Exp31C)은 코인간 분산이 √breadth를 상쇄해 실패.
  → 남은 레버: '시간'을 늘리면 코인간 분산 없이 독립 표본만 증가 → CI 수축 기대.

데이터 제약(정직히 명시):
  - SOL/AVAX/DOT(O5 일부)은 2020년 상장 → 2017~2020 부재. 장기 히스토리 코인으로 유니버스 교체.
  - 펀딩비(2019-09+)·공포탐욕(2018-02+) → 온체인 필터는 시간확장 구간에 적용 불가 → 온체인 제거(S1_noOC).
  - 따라서 이건 "같은 코인 더 긴 시간"이 아니라 "다른(구세대) 코인 + 더 긴 시간"의 독립 OOS 증거.

설계:
  유니버스(2017~2018 상장): BTC ETH LTC NEO ADA XRP EOS XLM TRX ETC (10)
  조건 S1_noOC: 가격MPE<10%ile(train) AND 볼륨MPE>=50%ile(train) AND RSI<30 AND price>MA200
              Kelly(절대 MPE 분위수) 0.5/0.3/0.15.  청산 RSI>50 OR 168H.  (온체인 제외)
  WF 스케줄: 2017-08-01부터 train 12M / test 6M / step 6M (통일).
    FULL = 전 윈도우(2018-08~2024 test).  LATE = test_start>=2021 부분집합.
    → LATE ⊂ FULL 이므로 'FULL이 LATE보다 CI 하한이 0을 넘는가'가 깨끗한 시간확장 효과.

엔진: collect_abs는 Exp31C에서 앵커검증(O5=37/+142/+23.6)된 함수에서 온체인만 제거.
측정: Exp30 거래단위 + 블록 부트스트랩 95% CI. (시간당-곡선 Sharpe 사용 안 함)

실행: py exp32_time_extension_wf.py   (선행: 10코인 2017-08~2025-01 OHLCV+MPE 캐시 필요)
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-", "") not in ("utf8", "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

from src.data.binance_collector import collect
from src.entropy.calculators import rolling_mpe
from src.analysis.h4_backtest import MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi

RESULTS_DIR = Path("results")
START, END  = "2017-08-01", "2025-01-01"
TRAIN_MONTHS, TEST_MONTHS = 12, 6
YEARS_FULL  = (pd.Timestamp(END) - pd.Timestamp("2018-08-01")).days / 365.25  # FULL OOS 연수 근사

UNIVERSE = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT", "ADAUSDT",
            "XRPUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT"]

def _lbl(s):
    return s.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
ENTROPY_PCT, VOL_PCT, RSI_OVERSOLD = 10, 50, 30
COST_LO, COST_HI = 0.0010, 0.0020
BOOT_B, BOOT_SEED = 5000, 42


def _setup_font():
    try:
        cands = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if cands:
            plt.rcParams["font.family"] = fm.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 측정 (Exp30 동일) ─────────────────────────────────────────────────────────
def net_return(t, c):
    g = t["p_exit"] / (t["p_entry"] * (1.0 + c)) - 1.0
    return (1.0 + t["kfrac"] * g) * (1.0 - c) - 1.0


def pooled_stats(trades, c, years):
    if not trades:
        return None
    r = np.array([net_return(t, c) for t in trades])
    n = len(r)
    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else 0.0
    gross = np.array([t["gross_pct"] for t in trades])
    tpy = n / years
    return {"n": n, "tpy": tpy,
            "gross_bps": gross.mean() * 100.0, "mean_bps": mean * 1e4,
            "t": mean / (std / np.sqrt(n)) if std > 0 else float("nan"),
            "winrate": float((r > 0).mean()),
            "sharpe_ann": (mean / std) * np.sqrt(tpy) if std > 0 else float("nan")}


def block_bootstrap(trades, c, years, B=BOOT_B, seed=BOOT_SEED):
    if len(trades) < 2:
        return None
    blocks = {}
    for t in trades:
        blocks.setdefault(t["block"], []).append(net_return(t, c))
    arrs = [np.array(v) for v in blocks.values()]
    rng = np.random.default_rng(seed)
    nb = len(arrs)
    means, sharpes = [], []
    for _ in range(B):
        pick = rng.integers(0, nb, size=nb)
        r = np.concatenate([arrs[j] for j in pick])
        if len(r) < 2:
            continue
        m, sd = r.mean(), r.std(ddof=1)
        means.append(m * 1e4)
        if sd > 0:
            sharpes.append((m / sd) * np.sqrt(len(r) / years))
    if not means:
        return None
    return {"mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5),
            "shp_lo": np.percentile(sharpes, 2.5) if sharpes else float("nan"),
            "shp_hi": np.percentile(sharpes, 97.5) if sharpes else float("nan")}


# ── 절대 Exp23-D 거래 수집 (온체인 제거) — Exp31C collect_abs와 동일 골격 ──────
def collect_abs_noOC(coin_lbl, df_test, p_test, v_test, p_train, v_train, k1, k5, k10):
    rsi = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()
    price_thr = np.percentile(p_train, ENTROPY_PCT)
    vt = v_train.dropna()
    vol_thr = np.percentile(vt, VOL_PCT) if len(vt) > 10 else np.nan

    def _kelly(v):
        if v <= k1: return 0.5
        if v <= k5: return 0.3
        if v <= k10: return 0.15
        return 0.0

    closes = df_test["close"]
    position = 0
    p_entry = entry_i = 0
    kfrac = 0.0
    trades = []
    for i, idx in enumerate(df_test.index):
        price = closes.loc[idx]
        rsi_val = rsi.loc[idx] if idx in rsi.index else 50.0
        ma_val = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe = p_test.loc[idx] if idx in p_test.index else np.nan
        v_mpe = v_test.loc[idx] if idx in v_test.index else np.nan
        vol_ok = (not np.isnan(v_mpe) and not np.isnan(vol_thr) and v_mpe >= vol_thr)

        if position == 1:
            held = i - entry_i
            if rsi_val > 50 or held >= MAX_HOLD_H:
                trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                               "kfrac": kfrac, "gross_pct": (price - p_entry) / p_entry * 100.0})
                position = 0

        if (position == 0 and rsi_val < RSI_OVERSOLD and not np.isnan(p_mpe)
                and not np.isnan(ma_val) and p_mpe <= price_thr and price > ma_val and vol_ok):
            kf = _kelly(p_mpe)
            if kf > 0:
                position = 1
                p_entry, entry_i, kfrac = price, i, kf

    if position == 1:
        price = closes.iloc[-1]
        trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                       "kfrac": kfrac, "gross_pct": (price - p_entry) / p_entry * 100.0})
    return trades


def wf_windows():
    """2017-08-01부터 train 12M/test 6M/step 6M 통일 스케줄."""
    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end = pd.Timestamp(END)
    wins = []
    while cursor + train_delta + test_delta <= end:
        ts = cursor + train_delta
        te = ts + test_delta
        wins.append((cursor, ts, ts, te))  # (train_start, train_end, test_start, test_end)
        cursor += test_delta
    return wins


def main():
    print("=" * 84)
    print("Exp32: 시간 확장 (2017~2020 추가) — 독립 표본 늘려 유의성 정면 공략")
    print(f"  유니버스(장기): {', '.join(_lbl(s) for s in UNIVERSE)}")
    print(f"  조건: S1_noOC (MPE<10% AND vol>=50% AND RSI<30 AND MA200, 온체인 제외)")
    print(f"  비교: LATE(test>=2021) vs FULL(2018~2024) — LATE⊂FULL, 시간확장 효과 격리")
    print("=" * 84)

    print("\n[10코인 2017-08~2025-01 데이터 + MPE 로드(캐시)...]")
    data = {}
    for sym in UNIVERSE:
        df = collect(sym, "1h", START, END)
        pm = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        vm = rolling_mpe(df["volume"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_volume_{START}_{END}")
        data[sym] = {"df": df, "pm": pm, "vm": vm}
        print(f"  [{_lbl(sym):5s}] {df.index[0].date()}~{df.index[-1].date()} {len(df):,}봉")

    wins = wf_windows()
    print(f"\n[WF 윈도우 {len(wins)}개 — 거래 수집...]")

    # 윈도우별·코인별 거래 수집
    per_window = []   # (test_start, trades)
    for (tr_s, tr_e, ts, te) in wins:
        wtrades = []
        for sym in UNIVERSE:
            d = data[sym]
            p_train = d["pm"][tr_s:tr_e].dropna()
            if len(p_train) < 200:
                continue
            k1, k5, k10 = (np.percentile(p_train, q) for q in (1, 5, 10))
            df_test = d["df"][ts:te]
            if len(df_test) < 168:
                continue
            tr = collect_abs_noOC(_lbl(sym), df_test, d["pm"][ts:te], d["vm"][ts:te],
                                  p_train, d["vm"][tr_s:tr_e], k1, k5, k10)
            period = ts.strftime("%Y-%m")
            for t in tr:
                t["period"] = period
                t["block"] = f"{_lbl(sym)}|{period}"
            wtrades.extend(tr)
        per_window.append((ts, wtrades))

    # 풀 구성
    full_trades = [t for (_, wt) in per_window for t in wt]
    late_trades = [t for (ts, wt) in per_window if ts >= pd.Timestamp("2021-01-01") for t in wt]
    early_trades = [t for (ts, wt) in per_window if ts < pd.Timestamp("2021-01-01") for t in wt]

    years_late = (pd.Timestamp(END) - pd.Timestamp("2021-01-01")).days / 365.25
    years_early = (pd.Timestamp("2021-01-01") - pd.Timestamp("2018-08-01")).days / 365.25
    years_full = years_late + years_early

    # ── 윈도우별 표 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  윈도우별 거래 (시간확장이 어떤 시기 표본을 더하나)")
    print("=" * 84)
    print(f"  {'test_start':>10} {'거래':>5} {'gross/거래':>10} {'net@0.1%':>9}  {'시기':>6}")
    print("  " + "-" * 50)
    for (ts, wt) in per_window:
        if not wt:
            print(f"  {ts.strftime('%Y-%m'):>10} {0:>5}        (거래없음)")
            continue
        st = pooled_stats(wt, COST_LO, years=1.0)
        era = "EARLY" if ts < pd.Timestamp("2021-01-01") else "LATE"
        print(f"  {ts.strftime('%Y-%m'):>10} {st['n']:>5} {st['gross_bps']:>+10.0f} "
              f"{st['mean_bps']:>+9.1f}  {era:>6}")

    # ── 핵심 비교: LATE vs FULL ──────────────────────────────────────────────
    def report(name, trades, years):
        st = pooled_stats(trades, COST_LO, years)
        st2 = pooled_stats(trades, COST_HI, years)
        ci = block_bootstrap(trades, COST_LO, years)
        if not st:
            print(f"\n  [{name}] 거래 없음")
            return
        flag = "CI하한>0 ✅ 유의달성!" if (ci and ci["mean_lo"] > 0) else "0포함 ⚠"
        print(f"\n  [{name}]  거래 {st['n']} ({st['tpy']:.0f}/년)")
        print(f"    gross/거래       : {st['gross_bps']:+.0f} bps")
        print(f"    net@0.1% / 0.2%  : {st['mean_bps']:+.1f} / {st2['mean_bps']:+.1f} bps")
        print(f"    t (@0.1% / 0.2%) : {st['t']:+.2f} / {st2['t']:+.2f}")
        print(f"    승률 / Sharpe_ann: {st['winrate']:.1%} / {st['sharpe_ann']:+.3f}")
        if ci:
            print(f"    부트스트랩 평균CI: [{ci['mean_lo']:+.1f}, {ci['mean_hi']:+.1f}] bps → {flag}")
            print(f"    부트스트랩 ShpCI : [{ci['shp_lo']:+.2f}, {ci['shp_hi']:+.2f}]")

    print("\n" + "=" * 84)
    print("  핵심: 시간 확장이 CI 하한을 0 위로 끌어올리는가")
    print("=" * 84)
    report("EARLY (2018~2020, 독립 OOS)", early_trades, years_early)
    report("LATE  (2021~2024)", late_trades, years_late)
    report("FULL  (2018~2024, 헤드라인)", full_trades, years_full)

    # ── 코인별 진단 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  코인별 거래당 엣지 (FULL, 진단용)")
    print("=" * 84)
    print(f"  {'코인':>6} {'거래':>5} {'gross':>7} {'net@0.1%':>9} {'승률':>6}")
    print("  " + "-" * 42)
    by_coin = {}
    for t in full_trades:
        by_coin.setdefault(t["coin"], []).append(t)
    for lbl, trs in sorted(by_coin.items(), key=lambda x: -pooled_stats(x[1], COST_LO, 1.0)["gross_bps"]):
        st = pooled_stats(trs, COST_LO, 1.0)
        print(f"  {lbl:>6} {st['n']:>5} {st['gross_bps']:>+7.0f} {st['mean_bps']:>+9.1f} {st['winrate']:>6.1%}")

    print("\n[시각화 생성...]")
    plot_results(per_window, early_trades, late_trades, full_trades,
                 years_early, years_late, years_full)
    print("\nExp32 완료!")


def plot_results(per_window, early, late, full, y_e, y_l, y_f):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    pools = [("EARLY", early, y_e, "#6acc65"), ("LATE", late, y_l, "#888888"),
             ("FULL", full, y_f, "#ee854a")]

    # (1) 부트스트랩 평균 CI — 핵심
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (nm, tr, yr, col) in enumerate(pools):
        st = pooled_stats(tr, COST_LO, yr)
        ci = block_bootstrap(tr, COST_LO, yr)
        if st and ci:
            ax1.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=col, capsize=7, ms=10, lw=2)
    ax1.axhline(0, color="red", lw=1.0, ls="--")
    ax1.set_xticks(range(len(pools))); ax1.set_xticklabels([p[0] for p in pools])
    ax1.set_ylabel("거래당 평균 net (bps) @0.1%")
    ax1.set_title("핵심: 시간확장 → CI 하한이 0을 넘는가")
    ax1.grid(axis="y", alpha=0.3)

    # (2) 윈도우별 거래수 (시기 색)
    ax2 = fig.add_subplot(gs[0, 1])
    xs = [ts.strftime("%y-%m") for (ts, _) in per_window]
    ns = [len(wt) for (_, wt) in per_window]
    cols = ["#6acc65" if ts < pd.Timestamp("2021-01-01") else "#888888" for (ts, _) in per_window]
    ax2.bar(range(len(xs)), ns, color=cols)
    ax2.set_xticks(range(len(xs))); ax2.set_xticklabels(xs, rotation=60, fontsize=7)
    ax2.set_ylabel("윈도우 거래수")
    ax2.set_title("윈도우별 표본 (초록=EARLY 신규, 회색=LATE)")
    ax2.grid(axis="y", alpha=0.3)

    # (3) 윈도우별 net@0.1%
    ax3 = fig.add_subplot(gs[1, 0])
    nets = []
    for (ts, wt) in per_window:
        nets.append(pooled_stats(wt, COST_LO, 1.0)["mean_bps"] if wt else 0)
    ax3.bar(range(len(xs)), nets, color=cols)
    ax3.axhline(0, color="red", lw=0.8, ls="--")
    ax3.set_xticks(range(len(xs))); ax3.set_xticklabels(xs, rotation=60, fontsize=7)
    ax3.set_ylabel("윈도우 net@0.1% (bps)")
    ax3.set_title("윈도우별 엣지 (EARLY가 양수 표본 더하나?)")
    ax3.grid(axis="y", alpha=0.3)

    # (4) 거래수 vs Sharpe_ann
    ax4 = fig.add_subplot(gs[1, 1])
    for nm, tr, yr, col in pools:
        st = pooled_stats(tr, COST_LO, yr)
        if st:
            ax4.scatter(st["n"], st["sharpe_ann"], s=140, color=col, zorder=5)
            ax4.annotate(nm, (st["n"], st["sharpe_ann"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax4.axhline(0, color="black", lw=0.6, ls="--")
    ax4.set_xlabel("거래수"); ax4.set_ylabel("연율 Sharpe (net 0.1%)")
    ax4.set_title("Breadth(시간) 효과")
    ax4.grid(alpha=0.3)

    plt.suptitle("Exp32: 시간 확장(2017~2020) — 독립 표본으로 유의성 공략", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp32_time_extension.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
