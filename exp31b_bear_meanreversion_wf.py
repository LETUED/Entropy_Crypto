"""
실험 31B: 약세 평균회귀 — 마지막 구조적으로 다른 메커니즘

기존 모든 실험: 상승추세(price>MA200) 눌림목 매수 = 조건부 베타. Exp30~35서 단독 엣지·사이징가치 미지지.
이 실험: 정반대 — 하락추세(price<MA200)에서 극단 과매도(capitulation) 후 반등 매수.
  price<MA200(약세) ⊥ price>MA200(강세) 상호배타 → 양의 엣지면 기존 전략과 자동 직교(결합 가능 독립 수익원).

이론 근거:
  Corbet & Katsiampa (2018) — 암호화폐 비대칭 평균회귀
    https://www.sciencedirect.com/science/article/abs/pii/S1057521918306136
  (극단 하락 후 기술적 반등이 약세장에서도 발생)

비교 모드 (방향 롱, 청산 RSI>50 OR 168H, 고정 사이징 kfrac=0.3):
  UP  (참조): price>MA200 + RSI<30 + MPE<10%   (기존 상승추세 눌림목, 대조군)
  B0 (순수):  price<MA200 + RSI<25              (약세 capitulation 반등)
  B1 (+MPE):  price<MA200 + RSI<25 + MPE<10%    (엔트로피가 약세 regime서 돕나?)
  B2 (깊은):  price<MA200 + RSI<20              (더 깊은 capitulation, 고확신)

측정: Exp30 거래단위 + 블록 부트스트랩 95% CI (룩어헤드 없음, 임계값 train).
베이스: 장기 10코인(2017~2025 캐시), WF train12M/test6M/step6M.
핵심: B0/B1/B2 중 부트스트랩 평균CI 하한>0 = 진짜 양의 엣지 = 직교 수익원 발견.
실행: py exp31b_bear_meanreversion_wf.py
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
START, END = "2017-08-01", "2025-01-01"
TRAIN_MONTHS, TEST_MONTHS = 12, 6
YEARS_FULL = (pd.Timestamp(END) - pd.Timestamp("2018-08-01")).days / 365.25

UNIVERSE = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT", "ADAUSDT",
            "XRPUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT"]

def _lbl(s):
    return s.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
ENTROPY_PCT = 10
KFRAC = 0.3
COST_LO, COST_HI = 0.0010, 0.0020
BOOT_B, BOOT_SEED = 5000, 42

# 모드: (regime, rsi_thresh, use_mpe). regime: 'up'=price>MA200, 'down'=price<MA200
MODES = {
    "UP": ("up", 30, True),
    "B0": ("down", 25, False),
    "B1": ("down", 25, True),
    "B2": ("down", 20, False),
}
MDESC = {
    "UP": "참조: 상승추세 눌림목 (RSI<30+MPE<10%)",
    "B0": "약세 capitulation (price<MA200+RSI<25)",
    "B1": "약세 + MPE<10%",
    "B2": "약세 깊은 capitulation (RSI<20)",
}


def _setup_font():
    try:
        cands = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if cands:
            plt.rcParams["font.family"] = fm.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


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
    return {"n": n, "tpy": tpy, "gross_bps": gross.mean() * 100.0, "mean_bps": mean * 1e4,
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
    means = []
    for _ in range(B):
        pick = rng.integers(0, nb, size=nb)
        r = np.concatenate([arrs[j] for j in pick])
        if len(r) >= 2:
            means.append(r.mean() * 1e4)
    if not means:
        return None
    return {"mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5)}


def collect_mode(coin_lbl, df_test, p_test, p_train, regime, rsi_thresh, use_mpe):
    rsi = compute_rsi(df_test["close"])
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()
    price_thr = np.percentile(p_train, ENTROPY_PCT) if use_mpe else None

    closes = df_test["close"]
    position = 0
    p_entry = entry_i = 0
    trades = []
    for i, idx in enumerate(df_test.index):
        price = closes.loc[idx]
        rsi_val = rsi.loc[idx] if idx in rsi.index else 50.0
        ma_val = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe = p_test.loc[idx] if idx in p_test.index else np.nan

        if position == 1:
            held = i - entry_i
            if rsi_val > 50 or held >= MAX_HOLD_H:
                trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                               "kfrac": KFRAC, "gross_pct": (price - p_entry) / p_entry * 100.0})
                position = 0

        if position == 0 and not np.isnan(ma_val) and rsi_val < rsi_thresh:
            regime_ok = (price > ma_val) if regime == "up" else (price < ma_val)
            mpe_ok = (not use_mpe) or (not np.isnan(p_mpe) and p_mpe <= price_thr)
            if regime_ok and mpe_ok:
                position = 1
                p_entry, entry_i = price, i

    if position == 1:
        price = closes.iloc[-1]
        trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                       "kfrac": KFRAC, "gross_pct": (price - p_entry) / p_entry * 100.0})
    return trades


def wf_windows():
    td = pd.DateOffset(months=TRAIN_MONTHS)
    sd = pd.DateOffset(months=TEST_MONTHS)
    cur = pd.Timestamp(START)
    end = pd.Timestamp(END)
    w = []
    while cur + td + sd <= end:
        w.append((cur, cur + td, cur + td, cur + td + sd))
        cur += sd
    return w


def main():
    print("=" * 84)
    print("Exp31B: 약세 평균회귀 — 마지막 구조적으로 다른 메커니즘")
    print(f"  베이스: 장기 10코인(2018~2024) | 방향 롱 | 청산 RSI>50 OR 168H | 고정 kfrac={KFRAC}")
    print(f"  UP참조 / B0약세capitulation / B1약세+MPE / B2약세깊은")
    print("=" * 84)

    print("\n[데이터 + MPE 로드(캐시)...]")
    data = {}
    for sym in UNIVERSE:
        df = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                          window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        data[sym] = {"df": df, "mpe": mpe}
    print(f"  {len(UNIVERSE)}코인 로드 완료")

    wins = wf_windows()
    trades_by_mode = {m: [] for m in MODES}
    per_window = {m: {} for m in MODES}

    for (tr_s, tr_e, ts, te) in wins:
        period = ts.strftime("%Y-%m")
        for sym in UNIVERSE:
            d = data[sym]
            p_train = d["mpe"][tr_s:tr_e].dropna()
            if len(p_train) < 200:
                continue
            df_test = d["df"][ts:te]
            if len(df_test) < 168:
                continue
            p_test = d["mpe"][ts:te]
            for m, (regime, rsi_t, use_mpe) in MODES.items():
                tr = collect_mode(_lbl(sym), df_test, p_test, p_train, regime, rsi_t, use_mpe)
                for t in tr:
                    t["period"] = period
                    t["block"] = f"{_lbl(sym)}|{period}"
                trades_by_mode[m].extend(tr)
                per_window[m].setdefault(period, []).extend(tr)

    # ── 요약 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  모드별 거래단위 통계 (핵심: 약세 모드 CI 하한>0?)")
    print("=" * 84)
    print(f"  {'모드':>4} {'거래':>5} {'/년':>5} {'gross':>7} {'net0.1%':>8} {'net0.2%':>8} "
          f"{'t':>6} {'승률':>6} {'평균CI(0.1%)':>17}")
    print("  " + "-" * 76)
    summ = {}
    for m in MODES:
        st = pooled_stats(trades_by_mode[m], COST_LO, YEARS_FULL)
        st2 = pooled_stats(trades_by_mode[m], COST_HI, YEARS_FULL)
        ci = block_bootstrap(trades_by_mode[m], COST_LO, YEARS_FULL)
        if not st:
            print(f"  {m:>4}  거래 없음")
            continue
        ci_s = f"[{ci['mean_lo']:+.1f},{ci['mean_hi']:+.1f}]" if ci else "—"
        flag = " ✅" if (ci and ci["mean_lo"] > 0) else ""
        warn = "⚠" if st["n"] < 30 else " "
        print(f"  {m:>4} {st['n']:>4}{warn} {st['tpy']:>5.0f} {st['gross_bps']:>+7.0f} "
              f"{st['mean_bps']:>+8.1f} {st2['mean_bps']:>+8.1f} {st['t']:>+6.2f} "
              f"{st['winrate']:>6.1%} {ci_s:>15}{flag}")
        summ[m] = {"st": st, "st2": st2, "ci": ci}
    print("  " + "-" * 76)
    for m in MODES:
        print(f"  {m}: {MDESC[m]}")

    # ── 약세 윈도우 집중 분석 (2018·2022 약세장) ──────────────────────────────
    print("\n" + "=" * 84)
    print("  약세장 윈도우 집중 (B0 약세 capitulation이 약세장서 작동하나)")
    print("=" * 84)
    print(f"  {'윈도우':>8} {'UP net':>8} {'UP n':>6} {'B0 net':>8} {'B0 n':>6}")
    print("  " + "-" * 44)
    periods = sorted(set(per_window["UP"].keys()) | set(per_window["B0"].keys()))
    for p in periods:
        up_st = pooled_stats(per_window["UP"].get(p, []), COST_LO, 1.0)
        b0_st = pooled_stats(per_window["B0"].get(p, []), COST_LO, 1.0)
        up_net = f"{up_st['mean_bps']:+.0f}" if up_st else "-"
        up_n = up_st["n"] if up_st else 0
        b0_net = f"{b0_st['mean_bps']:+.0f}" if b0_st else "-"
        b0_n = b0_st["n"] if b0_st else 0
        print(f"  {p:>8} {up_net:>8} {up_n:>6} {b0_net:>8} {b0_n:>6}")

    print("\n[시각화 생성...]")
    plot_results(summ, trades_by_mode)
    print("\nExp31B 완료!")


def plot_results(summ, trades_by_mode):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(17, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    pal = {"UP": "#888888", "B0": "#4878cf", "B1": "#ee854a", "B2": "#6acc65"}
    modes = [m for m in MODES if m in summ]

    # (1) 부트스트랩 평균 CI
    ax1 = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(modes):
        st = summ[m]["st"]; ci = summ[m]["ci"]
        if ci:
            ax1.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=pal[m], capsize=7, ms=10, lw=2)
    ax1.axhline(0, color="red", lw=1.0, ls="--")
    ax1.set_xticks(range(len(modes))); ax1.set_xticklabels(modes)
    ax1.set_ylabel("거래당 평균 net (bps) @0.1%")
    ax1.set_title("약세 평균회귀 엣지: CI 하한>0?")
    ax1.grid(axis="y", alpha=0.3)

    # (2) gross/net bps
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(modes)); w = 0.35
    g = [summ[m]["st"]["gross_bps"] for m in modes]
    n1 = [summ[m]["st"]["mean_bps"] for m in modes]
    ax2.bar(x - w/2, g, w, label="gross", color="#bbbbbb")
    ax2.bar(x + w/2, n1, w, label="net 0.1%", color="#4878cf")
    ax2.axhline(0, color="red", lw=0.8, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(modes)
    ax2.set_ylabel("거래당 (bps)")
    ax2.set_title("gross vs net")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp31B: 약세 평균회귀 (마지막 직교 메커니즘)", fontsize=13, y=1.02)
    out = RESULTS_DIR / "exp31b_bear_meanreversion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
