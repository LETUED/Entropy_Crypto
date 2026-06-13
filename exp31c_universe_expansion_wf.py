"""
실험 31C: 절대 Exp23-D 조건을 코인 유니버스 확장 (조건 희석 0, 순수 breadth)

프로젝트 의의 (잊지 말 것):
  "잃지 않는 게 돈을 버는 거야" — 자본 보존 최우선. MPE = 타이밍 필터(방향 아님, 방향은 RSI).
  범용성: 자산 특화 데이터 최소화 → 코인 확장이 철학과 부합.

배경 (감사 + Exp30/31A 정직한 좌표):
  - 진짜 엣지는 실재하나 작음: 거래당 gross +142bps, 거래단위 Sharpe ~0.78.
  - 병목 = n=37 → 통계 유의성 미달(부트스트랩 평균 CI에 0 포함, t=1.56<1.96).
  - Exp31A 확인: 절대 저MPE 조건은 load-bearing — 횡단면/완화로는 못 늘림(엣지 붕괴).

Exp31C 가설:
  신호 본체(절대 Exp23-D 조건)를 전혀 건드리지 않고 '같은 절대조건'을 더 많은 코인에 적용하면
  거래수 n이 늘어 √breadth 효과로 CI 하한이 0을 넘을 수 있다 (Grinold-Kahn: IR=IC×√breadth, IC>0 전제).

진입(전 코인 동일, Exp23-D 절대):
  가격MPE<10%ile(train) AND 볼륨MPE>=50%ile(train) AND RSI<30 AND price>MA200 AND 온체인<40%ile(train)
  Kelly 사이징(절대 MPE 분위수): 1%→0.5, 5%→0.3, 10%→0.15
청산: RSI>50 OR 168H (Exp30에서 TP+2%보다 견고 확인)
온체인: 시장-와이드(BTC 펀딩 + 공포탐욕) — Exp30/31A와 동일, 코인별 펀딩 불필요(범용성).

할루시네이션 방지 — 내장 검증 앵커:
  Exp31C의 O5 부분집합은 Exp30/31A의 S1(37거래, gross+142bps, net@0.1%+23.6bps)을 정확히 재현해야 함.
  불일치 시 버그 → FAIL 출력.

데이터 스누핑 방지:
  헤드라인 = ALL-21(선별 0, 완전 OOS). 코인별 성과표는 진단용.
  MAJORS는 '이번 데이터 결과'가 아니라 '이전 실험 근거'로만 사전 정의(BNB=Exp08, DOGE=밈, ETH=Exp19).

측정: Exp30 거래단위 프레임 + 블록 부트스트랩 95% CI. (시간당-곡선 Sharpe 사용 안 함)

실행: py exp31c_universe_expansion_wf.py
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
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h4_backtest import MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6
YEARS        = 4.0

O5 = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
WIDE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
        "LINKUSDT", "ATOMUSDT", "LTCUSDT", "XRPUSDT", "MATICUSDT", "NEARUSDT",
        "AAVEUSDT", "ALGOUSDT", "DOGEUSDT", "FILUSDT", "GRTUSDT", "SUSHIUSDT",
        "TRXUSDT", "UNIUSDT", "BNBUSDT"]
# 사전(a-priori) 메이저: 이전 실험 근거로만 제외 — BNB(Exp08 거래소토큰), DOGE(밈),
# ETH(Exp19 주식유사), MATIC(데이터 짧음/Exp20 하위), ALGO/FIL/GRT/SUSHI/TRX(소형/특수)
MAJORS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",
          "LINKUSDT", "ATOMUSDT", "LTCUSDT", "XRPUSDT", "UNIUSDT", "AAVEUSDT", "NEARUSDT"]

def _lbl(sym):
    return sym.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
ENTROPY_PCT, VOL_PCT, RSI_OVERSOLD = 10, 50, 30
COST_LO, COST_HI = 0.0010, 0.0020
BOOT_B, BOOT_SEED = 5000, 42

# Exp30/31A에서 확정된 S1(O5) 기준값 — 검증 앵커
ANCHOR = {"n": 37, "gross_bps": 142.0, "net_lo_bps": 23.6}


def _setup_font():
    try:
        cands = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if cands:
            plt.rcParams["font.family"] = fm.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 측정 (Exp30과 동일 회계) ──────────────────────────────────────────────────
def net_return(t, c):
    g = t["p_exit"] / (t["p_entry"] * (1.0 + c)) - 1.0
    return (1.0 + t["kfrac"] * g) * (1.0 - c) - 1.0


def pooled_stats(trades, c):
    if not trades:
        return None
    r = np.array([net_return(t, c) for t in trades])
    n = len(r)
    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else 0.0
    gross = np.array([t["gross_pct"] for t in trades])
    return {"n": n, "tpy": n / YEARS,
            "gross_bps": gross.mean() * 100.0, "mean_bps": mean * 1e4,
            "t": mean / (std / np.sqrt(n)) if std > 0 else float("nan"),
            "winrate": float((r > 0).mean()),
            "sharpe_raw": mean / std if std > 0 else float("nan"),
            "sharpe_ann": (mean / std) * np.sqrt(n / YEARS) if std > 0 else float("nan")}


def block_bootstrap(trades, c, B=BOOT_B, seed=BOOT_SEED):
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
            sharpes.append((m / sd) * np.sqrt(len(r) / YEARS))
    if not means:
        return None
    return {"mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5),
            "shp_lo": np.percentile(sharpes, 2.5) if sharpes else float("nan"),
            "shp_hi": np.percentile(sharpes, 97.5) if sharpes else float("nan")}


# ── 절대 Exp23-D 거래 수집 (Exp31A collect_abs와 동일 — O5 재현 보장) ─────────
def collect_abs(coin_lbl, df_test, p_test, v_test, h_test,
                p_train, v_train, oc_threshold, k1, k5, k10):
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
        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold
        vol_ok = (not np.isnan(v_mpe) and not np.isnan(vol_thr) and v_mpe >= vol_thr)

        if position == 1:
            held = i - entry_i
            if rsi_val > 50 or held >= MAX_HOLD_H:
                trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                               "kfrac": kfrac, "gross_pct": (price - p_entry) / p_entry * 100.0})
                position = 0

        if (position == 0 and rsi_val < RSI_OVERSOLD and not np.isnan(p_mpe)
                and not np.isnan(ma_val) and p_mpe <= price_thr and price > ma_val
                and oc_ok and vol_ok):
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
    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end = pd.Timestamp(END)
    wins = []
    while True:
        if cursor + train_delta + test_delta > end:
            break
        wins.append((cursor, cursor + train_delta,
                     cursor + train_delta, cursor + train_delta + test_delta))
        cursor += test_delta
    return wins


def report_pool(name, trades):
    st_lo = pooled_stats(trades, COST_LO)
    st_hi = pooled_stats(trades, COST_HI)
    ci = block_bootstrap(trades, COST_LO)
    if not st_lo:
        print(f"  {name}: 거래 없음")
        return None
    flag = "CI하한>0 ✅ 유의" if (ci and ci["mean_lo"] > 0) else "0포함 ⚠"
    warn = "⚠표본<30" if st_lo["n"] < 30 else ""
    print(f"\n  [{name}]  거래 {st_lo['n']} ({st_lo['tpy']:.0f}/년) {warn}")
    print(f"    gross/거래        : {st_lo['gross_bps']:+.0f} bps")
    print(f"    net@0.1% / 0.2%   : {st_lo['mean_bps']:+.1f} / {st_hi['mean_bps']:+.1f} bps")
    print(f"    t (@0.1% / 0.2%)  : {st_lo['t']:+.2f} / {st_hi['t']:+.2f}")
    print(f"    승률 / Sharpe_ann : {st_lo['winrate']:.1%} / {st_lo['sharpe_ann']:+.3f}")
    if ci:
        print(f"    부트스트랩 평균CI : [{ci['mean_lo']:+.1f}, {ci['mean_hi']:+.1f}] bps  → {flag}")
        print(f"    부트스트랩 ShpCI  : [{ci['shp_lo']:+.2f}, {ci['shp_hi']:+.2f}]")
    return {"st_lo": st_lo, "st_hi": st_hi, "ci": ci}


def main():
    print("=" * 82)
    print("Exp31C: 절대 Exp23-D 코인 확장 (조건 희석 0, 순수 breadth로 유의성 검증)")
    print(f"  유니버스: O5(5) / MAJORS({len(MAJORS)}) / ALL({len(WIDE)})  |  기간 {START}~{END}")
    print(f"  측정: 거래단위 + 블록 부트스트랩(B={BOOT_B})  |  청산 RSI>50 OR 168H")
    print("=" * 82)

    print("\n[온체인 로드 (시장-와이드: BTC펀딩+공포탐욕)...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg = collect_fear_greed(START, END)

    print("\n[21코인 데이터 + 가격/볼륨 MPE 로드(캐시)...]")
    coin_data = {}
    for sym in WIDE:
        df = collect(sym, "1h", START, END)
        pm = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        vm = rolling_mpe(df["volume"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_volume_{START}_{END}")
        h_oc = combined_onchain_entropy(funding_entropy(funding, df.index),
                                        fear_greed_entropy(fg, df.index))
        coin_data[sym] = {"df": df, "pm": pm, "vm": vm, "h_oc": h_oc}
    print(f"  {len(WIDE)}코인 로드 완료")

    wins = wf_windows()

    # ── WF 거래 수집 (코인별) ────────────────────────────────────────────────
    print("\n[Walk-forward 거래 수집 (절대 Exp23-D, 코인별)...]")
    per_coin = {}
    for sym in WIDE:
        d = coin_data[sym]
        lbl = _lbl(sym)
        trades = []
        for (tr_s, tr_e, ts, te) in wins:
            p_train = d["pm"][tr_s:tr_e].dropna()
            if len(p_train) < 200:
                continue
            k1, k5, k10 = (np.percentile(p_train, q) for q in (1, 5, 10))
            hoc_tr = d["h_oc"][tr_s:tr_e].dropna()
            oc_thr = np.percentile(hoc_tr, 40) if len(hoc_tr) > 0 else None
            df_test = d["df"][ts:te]
            if len(df_test) < 168:
                continue
            tr = collect_abs(lbl, df_test, d["pm"][ts:te], d["vm"][ts:te],
                             d["h_oc"][ts:te], p_train, d["vm"][tr_s:tr_e],
                             oc_thr, k1, k5, k10)
            period = ts.strftime("%Y-%m")
            for t in tr:
                t["period"] = period
                t["block"] = f"{lbl}|{period}"
            trades.extend(tr)
        per_coin[lbl] = trades

    def pool(syms):
        out = []
        for s in syms:
            out.extend(per_coin[_lbl(s)])
        return out

    # ── 검증 앵커: O5 = S1 재현 ──────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("  검증 앵커: O5 풀이 Exp30/31A의 S1을 재현하는가")
    print("=" * 82)
    o5_trades = pool(O5)
    o5_st = pooled_stats(o5_trades, COST_LO)
    ok_n = (o5_st["n"] == ANCHOR["n"])
    ok_g = abs(o5_st["gross_bps"] - ANCHOR["gross_bps"]) < 3
    ok_net = abs(o5_st["mean_bps"] - ANCHOR["net_lo_bps"]) < 1.5
    verdict = "PASS ✅" if (ok_n and ok_g and ok_net) else "FAIL ❌ (버그 의심 — 결과 신뢰 불가)"
    print(f"  O5 실측: n={o5_st['n']} (기대 {ANCHOR['n']}), "
          f"gross={o5_st['gross_bps']:+.0f}bps (기대 {ANCHOR['gross_bps']:+.0f}), "
          f"net@0.1%={o5_st['mean_bps']:+.1f}bps (기대 {ANCHOR['net_lo_bps']:+.1f})")
    print(f"  → {verdict}")

    # ── 코인별 진단표 (selection bias 회피: 진단용, 유의성 근거 아님) ─────────
    print("\n" + "=" * 82)
    print("  코인별 거래당 엣지 (진단용 — 사후선별 풀의 CI는 유의성 근거 아님)")
    print("=" * 82)
    print(f"  {'코인':>6} {'거래':>5} {'gross':>7} {'net@0.1%':>9} {'승률':>6} {'in-MAJORS':>10}")
    print("  " + "-" * 52)
    maj_lbls = {_lbl(s) for s in MAJORS}
    rows = []
    for sym in WIDE:
        lbl = _lbl(sym)
        st = pooled_stats(per_coin[lbl], COST_LO)
        if not st:
            print(f"  {lbl:>6}   거래 없음")
            continue
        rows.append((lbl, st))
    for lbl, st in sorted(rows, key=lambda x: -x[1]["gross_bps"]):
        tag = "○" if lbl in maj_lbls else ""
        print(f"  {lbl:>6} {st['n']:>5} {st['gross_bps']:>+7.0f} {st['mean_bps']:>+9.1f} "
              f"{st['winrate']:>6.1%} {tag:>10}")

    # ── 3개 풀 비교 (헤드라인 = ALL21, 선별 0) ───────────────────────────────
    print("\n" + "=" * 82)
    print("  풀 비교 — 헤드라인은 ALL21(선별 0, 완전 OOS). 핵심: CI 하한>0 도달?")
    print("=" * 82)
    res = {}
    res["O5(기준)"]    = report_pool("O5  (기준 S1)", pool(O5))
    res[f"ALL{len(WIDE)}"] = report_pool(f"ALL{len(WIDE)} (선별0, 헤드라인)", pool(WIDE))
    res[f"MAJORS{len(MAJORS)}"] = report_pool(f"MAJORS{len(MAJORS)} (사전제외, 데이터무관)", pool(MAJORS))

    print("\n[시각화 생성...]")
    plot_results(per_coin, pool, maj_lbls)
    print("\nExp31C 완료!")


def plot_results(per_coin, pool, maj_lbls):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    pools = {"O5": pool(O5), f"ALL{len(WIDE)}": pool(WIDE), f"MAJORS{len(MAJORS)}": pool(MAJORS)}
    pal = {"O5": "#888888", f"ALL{len(WIDE)}": "#ee854a", f"MAJORS{len(MAJORS)}": "#4878cf"}
    names = list(pools.keys())

    # (1) 부트스트랩 평균 CI
    ax1 = fig.add_subplot(gs[0, 0])
    for i, nm in enumerate(names):
        st = pooled_stats(pools[nm], COST_LO)
        ci = block_bootstrap(pools[nm], COST_LO)
        if st and ci:
            ax1.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=pal[nm], capsize=7, ms=10, lw=2)
    ax1.axhline(0, color="red", lw=1.0, ls="--")
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names)
    ax1.set_ylabel("거래당 평균 net (bps) @0.1%")
    ax1.set_title("핵심: 부트스트랩 95% CI 하한이 0을 넘는가")
    ax1.grid(axis="y", alpha=0.3)

    # (2) 거래수 vs 연율 Sharpe
    ax2 = fig.add_subplot(gs[0, 1])
    for nm in names:
        st = pooled_stats(pools[nm], COST_LO)
        if st:
            ax2.scatter(st["n"], st["sharpe_ann"], s=140, color=pal[nm], zorder=5)
            ax2.annotate(nm, (st["n"], st["sharpe_ann"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_xlabel("거래수"); ax2.set_ylabel("연율 Sharpe (net 0.1%)")
    ax2.set_title("Breadth 효과: 거래수 vs Sharpe")
    ax2.grid(alpha=0.3)

    # (3) 코인별 gross 엣지
    ax3 = fig.add_subplot(gs[1, :])
    rows = []
    for lbl, trs in per_coin.items():
        st = pooled_stats(trs, COST_LO)
        if st:
            rows.append((lbl, st["gross_bps"], st["n"], lbl in maj_lbls))
    rows.sort(key=lambda x: -x[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    bcol = ["#4878cf" if r[3] else "#cccccc" for r in rows]
    ax3.bar(range(len(labels)), vals, color=bcol)
    ax3.axhline(142, color="green", ls=":", lw=1.0, label="O5 기준 gross +142bps")
    ax3.axhline(0, color="red", lw=0.8, ls="--")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels([f"{r[0]}\n({r[2]})" for r in rows], fontsize=7)
    ax3.set_ylabel("거래당 gross (bps)")
    ax3.set_title("코인별 거래당 엣지 (파랑=MAJORS, 괄호=거래수) — 진단용")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp31C: 절대 Exp23-D 코인 확장 — 조건 유지한 순수 breadth", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp31c_universe_expansion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
