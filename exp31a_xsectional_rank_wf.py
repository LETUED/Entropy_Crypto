"""
실험 31A: 횡단면(Cross-sectional) MPE 랭킹 — 거래수 확장으로 유의성 확보 시도

배경: Exp30이 입증 — 현재 전략의 진짜 거래단위 Sharpe ~0.7, 부트스트랩 CI에 0 포함
  (37거래로 '엣지=0' 기각 불가). 정교화로는 유의성에 도달 못 함. 표본 자체를 늘려야 함.

가설: 절대 임계값(MPE<10%ile, train) 대신 매 시점 N코인 중 MPE '횡단면 하위 k'를 롱하면,
  - 시장 전체가 저변동이 아니어도 매시간 '상대적으로 가장 예측가능한' 코인이 존재 → 거래 빈도 구조적 증가
  - 절대 저변동 국면에 묶이지 않으므로 기존 절대전략(S1=Exp23-D)과 상관 낮은 독립 거래 생성 가능
  - Fundamental Law of Active Management: IR ≈ IC × sqrt(breadth). 폭(breadth)을 늘려 유의성 확보.

이론 근거:
  Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning", RFS
    https://academic.oup.com/rfs/article/33/5/2223/5758276  (횡단면 신호의 자산가격 예측력)
  Grinold & Kahn, Fundamental Law of Active Management (IR=IC*sqrt(breadth))

look-ahead 없음:
  - MPE는 인과적(과거 168봉). 횡단면 랭킹은 동시점(contemporaneous) t의 MPE만 사용.
  - RSI/MA200도 인과적. test 구간 거래만 집계(OOS).
  - 코인 상장 전(NaN MPE)은 랭킹에서 자동 제외 → 미래 정보 없음.

측정(Exp30 프레임 재사용): 거래단위 풀링 통계 + 블록 부트스트랩 95% CI.
  시간당-곡선 Sharpe는 사용하지 않음(희소곡선 왜곡 회피).

비교 런:
  S1_abs    : O5, 절대 Exp23-D (MPE<10% AND vol>=50% AND RSI<30 AND MA200 AND 온체인<40%)  [기준]
  X_O5      : O5(5코인),  횡단면 하위2 MPE + MA200 + RSI<30
  X_W       : WIDE(21코인), 횡단면 하위2 MPE + MA200 + RSI<30   (유니버스 효과 격리)
  X_W_noRSI : WIDE(21코인), 횡단면 하위2 MPE + MA200            (RSI 게이트 제거)
  X_W_b1    : WIDE(21코인), 횡단면 하위1 MPE + MA200            (최저 1개만)

청산(전 런 동일): RSI>50 OR 168H  (Exp30에서 TP+2%보다 견고 확인)

실행: py exp31a_xsectional_rank_wf.py
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

def _lbl(sym):  # 표시용 라벨
    return sym.replace("USDT", "")

MPE_WINDOW = 168
MPE_M      = 3
MPE_SCALES = [1, 2, 4, 8]

ENTROPY_PCT  = 10
VOL_PCT      = 50
RSI_OVERSOLD = 30

K_RANK = {1: 0.30, 2: 0.15}   # 횡단면 랭크별 Kelly(고정): 1위 0.30, 2위 0.15
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


# ── 측정 헬퍼 (Exp30과 동일 회계) ─────────────────────────────────────────────
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
    return {
        "n": n, "tpy": n / YEARS,
        "gross_bps": gross.mean() * 100.0,   # gross_pct는 %, ×100 = bps
        "mean_bps": mean * 1e4,
        "t": mean / (std / np.sqrt(n)) if std > 0 else float("nan"),
        "winrate": float((r > 0).mean()),
        "sharpe_raw": mean / std if std > 0 else float("nan"),
        "sharpe_ann": (mean / std) * np.sqrt(n / YEARS) if std > 0 else float("nan"),
    }


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
    return {
        "mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5),
        "shp_lo": np.percentile(sharpes, 2.5) if sharpes else float("nan"),
        "shp_hi": np.percentile(sharpes, 97.5) if sharpes else float("nan"),
    }


# ── 거래 수집: 횡단면 랭킹 ────────────────────────────────────────────────────
def collect_xsec(coin_lbl, df_test, xrank_test, rsi, ma200, k_bottom, use_rsi):
    closes = df_test["close"]
    position = 0
    p_entry = entry_i = 0
    kfrac = 0.0
    trades = []
    for i, idx in enumerate(df_test.index):
        price   = closes.loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        rnk     = xrank_test.loc[idx] if idx in xrank_test.index else np.nan

        if position == 1:
            held = i - entry_i
            if rsi_val > 50 or held >= MAX_HOLD_H:
                trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                               "kfrac": kfrac, "held_h": held,
                               "gross_pct": (price - p_entry) / p_entry * 100.0,
                               "entry_i": entry_i, "exit_i": i})
                position = 0

        if position == 0 and not np.isnan(rnk) and not np.isnan(ma_val):
            in_bottom = rnk <= k_bottom
            gate = price > ma_val
            if use_rsi:
                gate = gate and (rsi_val < RSI_OVERSOLD)
            if in_bottom and gate:
                kf = K_RANK.get(int(rnk), 0.15)
                position = 1
                p_entry, entry_i, kfrac = price, i, kf

    if position == 1:
        price = closes.iloc[-1]
        trades.append({"coin": coin_lbl, "p_entry": p_entry, "p_exit": price,
                       "kfrac": kfrac, "held_h": len(df_test) - entry_i,
                       "gross_pct": (price - p_entry) / p_entry * 100.0,
                       "entry_i": entry_i, "exit_i": len(df_test) - 1})
    return trades


# ── 거래 수집: 절대 Exp23-D (S1 기준선) ───────────────────────────────────────
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
                               "kfrac": kfrac, "held_h": held,
                               "gross_pct": (price - p_entry) / p_entry * 100.0,
                               "entry_i": entry_i, "exit_i": i})
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
                       "kfrac": kfrac, "held_h": len(df_test) - entry_i,
                       "gross_pct": (price - p_entry) / p_entry * 100.0,
                       "entry_i": entry_i, "exit_i": len(df_test) - 1})
    return trades


def wf_windows():
    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end = pd.Timestamp(END)
    wins = []
    while True:
        ts, te = cursor + train_delta, cursor + train_delta + test_delta
        if te > end:
            break
        wins.append((cursor, cursor + train_delta, ts, te))
        cursor += test_delta
    return wins


# ── 독립성: S1 거래와 시간 겹침 ───────────────────────────────────────────────
def independence(x_trades, s1_trades):
    """X 거래 중 비-O5 코인 비중 + O5 코인에서 S1과 보유구간 겹치는 비율."""
    o5_lbls = {_lbl(s) for s in O5}
    s1_iv = {}
    for t in s1_trades:
        s1_iv.setdefault((t["coin"], t["period"]), []).append((t["entry_i"], t["exit_i"]))
    n = len(x_trades)
    n_non_o5 = sum(1 for t in x_trades if t["coin"] not in o5_lbls)
    n_o5 = n - n_non_o5
    overlap = 0
    for t in x_trades:
        if t["coin"] in o5_lbls:
            for (a, b) in s1_iv.get((t["coin"], t["period"]), []):
                if t["entry_i"] <= b and a <= t["exit_i"]:
                    overlap += 1
                    break
    overlap_rate = overlap / n_o5 if n_o5 else 0.0
    indep_share = (n_non_o5 + (n_o5 - overlap)) / n if n else 0.0
    return {"n": n, "n_non_o5": n_non_o5, "n_o5": n_o5,
            "overlap_rate": overlap_rate, "indep_share": indep_share}


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Exp31A: 횡단면 MPE 랭킹 — 거래수 확장으로 유의성 확보 시도")
    print(f"  유니버스: O5(5) vs WIDE({len(WIDE)})  |  기간 {START}~{END}")
    print(f"  측정: 거래단위 통계 + 블록 부트스트랩(B={BOOT_B})  |  청산 RSI>50 OR 168H")
    print("=" * 80)

    print("\n[온체인 로드(S1용)...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg = collect_fear_greed(START, END)

    print("\n[WIDE 코인 데이터 + MPE 로드(캐시)...]")
    data = {}
    price_mpe_map = {}
    for sym in WIDE:
        df = collect(sym, "1h", START, END)
        pm = pd.Series(rolling_mpe(df["close"].values, m=MPE_M, scales=MPE_SCALES,
                                   window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}"),
                       index=df.index)
        rsi = compute_rsi(df["close"])
        ma200 = df["close"].rolling(MA_PERIOD).mean()
        data[sym] = {"df": df, "rsi": rsi, "ma200": ma200}
        price_mpe_map[_lbl(sym)] = pm
        print(f"  [{_lbl(sym):5s}] {len(df):,}봉  유효MPE {pm.notna().sum():,}")

    # O5 전용: 볼륨 MPE + 온체인 (S1용)
    o5_extra = {}
    for sym in O5:
        df = data[sym]["df"]
        vm = pd.Series(rolling_mpe(df["volume"].values, m=MPE_M, scales=MPE_SCALES,
                                   window=MPE_WINDOW, cache_key=f"{sym}_1h_volume_{START}_{END}"),
                       index=df.index)
        h_oc = combined_onchain_entropy(funding_entropy(funding, df.index),
                                        fear_greed_entropy(fg, df.index))
        o5_extra[sym] = {"vol_mpe": vm, "h_oc": h_oc,
                         "price_mpe": pd.Series(rolling_mpe(df["close"].values, m=MPE_M,
                            scales=MPE_SCALES, window=MPE_WINDOW,
                            cache_key=f"{sym}_1h_{START}_{END}"), index=df.index)}

    # 횡단면 랭크 행렬 (WIDE / O5)
    print("\n[횡단면 MPE 랭크 계산...]")
    mat_wide = pd.concat({lbl: price_mpe_map[lbl] for lbl in [_lbl(s) for s in WIDE]}, axis=1)
    xrank_wide = mat_wide.rank(axis=1, method="min")
    mat_o5 = pd.concat({_lbl(s): price_mpe_map[_lbl(s)] for s in O5}, axis=1)
    xrank_o5 = mat_o5.rank(axis=1, method="min")

    wins = wf_windows()

    # 런 정의: (universe, xrank_df, k_bottom, use_rsi)
    runs = {
        "X_O5":      {"uni": O5,   "xr": xrank_o5,   "k": 2, "rsi": True},
        "X_W":       {"uni": WIDE, "xr": xrank_wide, "k": 2, "rsi": True},
        "X_W_noRSI": {"uni": WIDE, "xr": xrank_wide, "k": 2, "rsi": False},
        "X_W_b1":    {"uni": WIDE, "xr": xrank_wide, "k": 1, "rsi": False},
    }
    RUN_DESC = {
        "X_O5":      "O5  하위2+MA200+RSI<30",
        "X_W":       f"W{len(WIDE)} 하위2+MA200+RSI<30",
        "X_W_noRSI": f"W{len(WIDE)} 하위2+MA200",
        "X_W_b1":    f"W{len(WIDE)} 하위1+MA200",
    }

    all_trades = {k: [] for k in runs}
    s1_trades = []

    print("\n[Walk-forward 거래 수집...]")
    for (tr_s, tr_e, ts, te) in wins:
        period = ts.strftime("%Y-%m")

        # 횡단면 런
        for rk, cfg in runs.items():
            for sym in cfg["uni"]:
                lbl = _lbl(sym)
                df = data[sym]["df"]
                df_test = df[ts:te]
                if len(df_test) < 168:
                    continue
                xr_test = cfg["xr"][lbl][ts:te]
                tr = collect_xsec(lbl, df_test, xr_test,
                                  data[sym]["rsi"][ts:te], data[sym]["ma200"][ts:te],
                                  cfg["k"], cfg["rsi"])
                for t in tr:
                    t["period"] = period
                    t["block"] = f"{lbl}|{period}"
                all_trades[rk].extend(tr)

        # S1 절대 기준선 (O5)
        for sym in O5:
            df = data[sym]["df"]
            pm = o5_extra[sym]["price_mpe"]
            p_train = pm[tr_s:tr_e].dropna()
            if len(p_train) < 200:
                continue
            k1, k5, k10 = (np.percentile(p_train, q) for q in (1, 5, 10))
            vm = o5_extra[sym]["vol_mpe"]
            h_oc = o5_extra[sym]["h_oc"]
            oc_thr = None
            hoc_tr = h_oc[tr_s:tr_e].dropna()
            if len(hoc_tr) > 0:
                oc_thr = np.percentile(hoc_tr, 40)
            df_test = df[ts:te]
            if len(df_test) < 168:
                continue
            tr = collect_abs(_lbl(sym), df_test, pm[ts:te], vm[ts:te],
                             h_oc[ts:te] if h_oc is not None else None,
                             p_train, vm[tr_s:tr_e], oc_thr, k1, k5, k10)
            for t in tr:
                t["period"] = period
                t["block"] = f"{_lbl(sym)}|{period}"
            s1_trades.extend(tr)

    # ── 리포트 ────────────────────────────────────────────────────────────────
    order = ["S1_abs", "X_O5", "X_W", "X_W_noRSI", "X_W_b1"]
    pools = {"S1_abs": s1_trades, **all_trades}
    desc = {"S1_abs": "O5 절대 Exp23-D [기준]", **RUN_DESC}

    print("\n" + "=" * 80)
    print("  거래단위 풀링 통계 (look-ahead 없는 OOS, 청산 RSI>50/168H)")
    print("=" * 80)
    print(f"  {'런':>10} {'거래':>5} {'/년':>5} {'gross':>7} "
          f"{'net0.1%':>8} {'t':>6} {'승률':>6} {'Sh_raw':>7} {'Sh_ann':>7}")
    print("  " + "-" * 76)
    for k in order:
        st = pooled_stats(pools[k], COST_LO)
        if not st:
            continue
        warn = "⚠" if st["n"] < 30 else " "
        sig = "*" if (not np.isnan(st["t"]) and abs(st["t"]) > 1.96) else " "
        print(f"  {k:>10} {st['n']:>4}{warn} {st['tpy']:>5.0f} {st['gross_bps']:>+7.0f} "
              f"{st['mean_bps']:>+8.1f} {st['t']:>5.2f}{sig} {st['winrate']:>6.1%} "
              f"{st['sharpe_raw']:>7.3f} {st['sharpe_ann']:>7.3f}")
    print("  " + "-" * 76)
    print("  gross/net = 거래당 평균(bps,gross는수수료前) | Sh_raw=거래당 | Sh_ann=연율(=raw×√(거래/년))")
    print("  * = |t|>1.96 (5% 유의)")

    print("\n" + "=" * 80)
    print(f"  비용 0.20%(현실 taker)에서 net + 블록 부트스트랩 95% CI @0.10%")
    print("=" * 80)
    print(f"  {'런':>10} {'net0.2%':>8} {'t0.2%':>6} | {'평균CI(0.1%,bps)':>22} {'Sharpe_ann CI':>20}")
    print("  " + "-" * 74)
    for k in order:
        st_hi = pooled_stats(pools[k], COST_HI)
        ci = block_bootstrap(pools[k], COST_LO)
        if not st_hi or not ci:
            continue
        flag = "CI하한>0 ✅" if ci["mean_lo"] > 0 else "0포함 ⚠"
        mci = f"[{ci['mean_lo']:+.1f}, {ci['mean_hi']:+.1f}]"
        sci = f"[{ci['shp_lo']:+.2f}, {ci['shp_hi']:+.2f}]"
        print(f"  {k:>10} {st_hi['mean_bps']:>+8.1f} {st_hi['t']:>5.2f}  | "
              f"{mci:>22} {sci:>20}  {flag}")

    print("\n" + "=" * 80)
    print("  S1(절대)과의 독립성 — 비-O5 비중 + O5 보유구간 겹침률")
    print("=" * 80)
    print(f"  {'런':>10} {'거래':>5} {'비-O5':>6} {'O5거래':>6} {'겹침률':>7} {'독립거래비중':>10}")
    print("  " + "-" * 60)
    for k in ["X_O5", "X_W", "X_W_noRSI", "X_W_b1"]:
        ind = independence(all_trades[k], s1_trades)
        print(f"  {k:>10} {ind['n']:>5} {ind['n_non_o5']:>6} {ind['n_o5']:>6} "
              f"{ind['overlap_rate']:>6.1%} {ind['indep_share']:>9.1%}")

    print("\n[시각화 생성...]")
    plot_results(pools, order, desc, all_trades, s1_trades)
    print("\nExp31A 완료!")


def plot_results(pools, order, desc, all_trades, s1_trades):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)
    palette = {"S1_abs": "#888888", "X_O5": "#4878cf", "X_W": "#ee854a",
               "X_W_noRSI": "#6acc65", "X_W_b1": "#c44e52"}

    # (1) 거래수 vs 연율 Sharpe (breadth 효과)
    ax1 = fig.add_subplot(gs[0, 0])
    for k in order:
        st = pooled_stats(pools[k], COST_LO)
        if st:
            ax1.scatter(st["n"], st["sharpe_ann"], s=130, color=palette[k], zorder=5)
            ax1.annotate(k, (st["n"], st["sharpe_ann"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax1.axhline(0, color="black", lw=0.6, ls="--")
    ax1.set_xlabel("총 거래수"); ax1.set_ylabel("연율 Sharpe (net 0.1%)")
    ax1.set_title("Breadth 효과: 거래수 vs 연율 Sharpe")
    ax1.grid(alpha=0.3)

    # (2) 부트스트랩 평균 CI @0.1%
    ax2 = fig.add_subplot(gs[0, 1])
    for i, k in enumerate(order):
        st = pooled_stats(pools[k], COST_LO)
        ci = block_bootstrap(pools[k], COST_LO)
        if st and ci:
            ax2.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=palette[k], capsize=6, ms=9, lw=2)
    ax2.axhline(0, color="red", lw=1.0, ls="--")
    ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order, rotation=20, fontsize=8)
    ax2.set_ylabel("거래당 평균 net (bps)")
    ax2.set_title("블록 부트스트랩 95% CI @0.1% (하한>0?)")
    ax2.grid(axis="y", alpha=0.3)

    # (3) 거래당 gross/net bps (비용 잠식)
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(order)); w = 0.27
    g = [pooled_stats(pools[k], COST_LO)["gross_bps"] for k in order]
    n1 = [pooled_stats(pools[k], COST_LO)["mean_bps"] for k in order]
    n2 = [pooled_stats(pools[k], COST_HI)["mean_bps"] for k in order]
    ax3.bar(x - w, g, w, label="gross", color="#bbbbbb")
    ax3.bar(x, n1, w, label="net 0.1%", color="#4878cf")
    ax3.bar(x + w, n2, w, label="net 0.2%", color="#ee854a")
    ax3.axhline(0, color="red", lw=0.8, ls="--")
    ax3.set_xticks(x); ax3.set_xticklabels(order, rotation=20, fontsize=8)
    ax3.set_ylabel("거래당 평균 (bps)"); ax3.set_title("비용 잠식: gross→net")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # (4) 독립성
    ax4 = fig.add_subplot(gs[1, 1])
    ks = ["X_O5", "X_W", "X_W_noRSI", "X_W_b1"]
    inds = [independence(all_trades[k], s1_trades) for k in ks]
    ax4.bar(range(len(ks)), [d["indep_share"] for d in inds],
            color=[palette[k] for k in ks], alpha=0.85)
    ax4.axhline(0.7, color="green", ls=":", lw=1.0, label="독립 0.7 기준")
    ax4.set_xticks(range(len(ks))); ax4.set_xticklabels(ks, rotation=20, fontsize=8)
    ax4.set_ylabel("S1 대비 독립 거래 비중"); ax4.set_ylim(0, 1.05)
    ax4.set_title("S1(절대)과의 독립성"); ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp31A: 횡단면 MPE 랭킹 (거래수 확장 + 독립성 검증)", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp31a_xsectional_rank.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
