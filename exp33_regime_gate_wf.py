"""
실험 33: 레짐 조건부 안전 게이트 — "이상치(붕괴) 레짐은 사전 제외, 정상에서 안 잃기"

프로젝트 의의(닻): "잃지 않는 게 돈을 버는 거야". MPE=타이밍 필터(방향 RSI).
  Exp30/31A/31C/32 수렴: 단독 수익엣지는 통계 유의 미달, 엣지는 配置·레짐 의존.
  사용자 통찰: 붕괴에선 잃는 게 맞다 → 안전장치로 진입을 매우 보수적으로.
              이상치(하방 붕괴 레짐) 제외하고 정상 레짐에서 양의 기대값 + 꼬리 한정.

핵심 가설(검증 대상):
  사전(ex-ante) 안전 게이트로 '나쁜 레짐' 거래를 빼면,
  남은 정상 레짐 거래의 net CI 하한이 0 쪽으로 수축/돌파하고 꼬리(최악 윈도우)가 줄어드는가.
  단 '수익은 상방 아웃라이어(고변동 눌림)에서도 나옴' → 엣지 보존율을 함께 측정(목욕물/아기).

게이트 (모두 ex-ante, look-ahead 없음, 파라미터 사전 고정=과적합 방지):
  G0  기준        : 게이트 없음 (Exp32 FULL 재현)
  G1  분산-EWS    : 진입직전 168H 로그수익률 분산이 train 90%ile 초과면 진입보류
                    근거: Guttal 외(2016) PLOS ONE, Tu 외(2020) RSOS — 붕괴 전 분산 상승(자기상관 아님)
  G2  분산+음왜도 : 분산 90%ile초과 AND 왜도 train 33%ile미만(좌꼬리)일 때만 보류
                    → '건강한 고변동(상승눌림)'은 통과, '붕괴前 불안정'만 차단
  G3  CUSUM 서킷  : 시장(BTC) 로그수익률 하방 CUSUM(k=0.5,h=5) 경보 시 전코인 진입중단
                    근거: Pepelyshev & Polunchenko(2015) 실시간 금융감시

베이스: Exp32 유니버스 10코인(BTC ETH LTC NEO ADA XRP EOS XLM TRX ETC), 2018~2024,
        조건 S1_noOC(MPE<10% AND vol>=50% AND RSI<30 AND MA200), 청산 RSI>50 OR 168H.
측정: Exp30 거래단위 + 블록 부트스트랩 95% CI.
성공기준(재정의): ①정상레짐 CI 하한 상승/돌파 ②꼬리(최악윈도우) 축소 ③엣지 보존율 높음.

실행: py exp33_regime_gate_wf.py   (선행: Exp32 데이터 캐시)
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

UNIVERSE = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT", "ADAUSDT",
            "XRPUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT"]

def _lbl(s):
    return s.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
ENTROPY_PCT, VOL_PCT, RSI_OVERSOLD = 10, 50, 30
COST_LO, COST_HI = 0.0010, 0.0020
BOOT_B, BOOT_SEED = 5000, 42

# 게이트 파라미터 (사전 고정 — 과적합 방지)
VAR_PCT  = 90    # 분산 상위 분위수 (이 초과면 위험)
SKEW_PCT = 33    # 왜도 하위 분위수 (이 미만이면 좌꼬리)
CUSUM_K, CUSUM_H = 0.5, 5.0

GATES = ["G0", "G1", "G2", "G3"]
GATE_DESC = {
    "G0": "기준(게이트 없음)",
    "G1": "분산-EWS (분산>90%ile 보류)",
    "G2": "분산+음왜도 (둘 다일 때만 보류)",
    "G3": "CUSUM 서킷 (BTC 붕괴경보 중단)",
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
    means = []
    for _ in range(B):
        pick = rng.integers(0, nb, size=nb)
        r = np.concatenate([arrs[j] for j in pick])
        if len(r) >= 2:
            means.append(r.mean() * 1e4)
    if not means:
        return None
    return {"mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5)}


# ── 게이트 차단 타임스탬프 집합 계산 (ex-ante) ────────────────────────────────
def blocked_set(gate, var_test, skew_test, var_thr, skew_thr, cusum_alarm_idx):
    if gate == "G0":
        return set()
    if gate == "G1":
        return set(var_test.index[var_test > var_thr])
    if gate == "G2":
        mask = (var_test > var_thr) & (skew_test < skew_thr)
        return set(var_test.index[mask.fillna(False)])
    if gate == "G3":
        return cusum_alarm_idx
    return set()


def cusum_alarm_timestamps(btc_logret, tr_s, tr_e, ts, te, k=CUSUM_K, h=CUSUM_H):
    """train 평균/표준편차로 표준화한 BTC 로그수익률의 하방 CUSUM 경보 타임스탬프(test 구간)."""
    r_train = btc_logret[tr_s:tr_e].dropna()
    if len(r_train) < 50:
        return set()
    mu, sd = r_train.mean(), r_train.std()
    if sd == 0:
        return set()
    r_test = btc_logret[ts:te]
    S = 0.0
    alarms = set()
    for idx, rv in r_test.items():
        if np.isnan(rv):
            continue
        z = (rv - mu) / sd
        S = max(0.0, S - z - k)   # 음수 수익률(폭락) 누적 → S 상승
        if S > h:
            alarms.add(idx)
    return alarms


# ── 거래 수집 (게이트 적용) ───────────────────────────────────────────────────
def collect_gated(coin_lbl, df_test, p_test, v_test, p_train, v_train,
                  k1, k5, k10, blocked):
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

        if (position == 0 and idx not in blocked
                and rsi_val < RSI_OVERSOLD and not np.isnan(p_mpe)
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
    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end = pd.Timestamp(END)
    wins = []
    while cursor + train_delta + test_delta <= end:
        ts = cursor + train_delta
        wins.append((cursor, ts, ts, ts + test_delta))
        cursor += test_delta
    return wins


def main():
    print("=" * 86)
    print("Exp33: 레짐 조건부 안전 게이트 — 이상치(붕괴) 사전제외, 정상에서 안 잃기")
    print(f"  베이스: {', '.join(_lbl(s) for s in UNIVERSE)} (2018~2024, S1_noOC)")
    print(f"  게이트: G0기준 / G1분산-EWS / G2분산+음왜도 / G3 CUSUM서킷  (모두 ex-ante)")
    print("=" * 86)

    print("\n[데이터 + MPE + 분산/왜도 로드(캐시)...]")
    data = {}
    for sym in UNIVERSE:
        df = collect(sym, "1h", START, END)
        pm = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        vm = rolling_mpe(df["volume"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                         window=MPE_WINDOW, cache_key=f"{sym}_1h_volume_{START}_{END}")
        logret = np.log(df["close"]).diff()
        var168 = logret.rolling(MPE_WINDOW).var()
        skew168 = logret.rolling(MPE_WINDOW).skew()
        data[sym] = {"df": df, "pm": pm, "vm": vm,
                     "var": var168, "skew": skew168, "logret": logret}
    btc_logret = data["BTCUSDT"]["logret"]
    print(f"  {len(UNIVERSE)}코인 로드 완료")

    wins = wf_windows()
    years_full = (pd.Timestamp(END) - pd.Timestamp("2018-08-01")).days / 365.25

    # ── 게이트별 거래 수집 ────────────────────────────────────────────────────
    print(f"\n[WF {len(wins)}윈도우 × {len(GATES)}게이트 거래 수집...]")
    trades_by_gate = {g: [] for g in GATES}
    per_window_net = {g: {} for g in GATES}   # gate -> period -> net_bps

    for (tr_s, tr_e, ts, te) in wins:
        period = ts.strftime("%Y-%m")
        cusum_alarms = cusum_alarm_timestamps(btc_logret, tr_s, tr_e, ts, te)
        win_trades = {g: [] for g in GATES}
        for sym in UNIVERSE:
            d = data[sym]
            p_train = d["pm"][tr_s:tr_e].dropna()
            if len(p_train) < 200:
                continue
            k1, k5, k10 = (np.percentile(p_train, q) for q in (1, 5, 10))
            df_test = d["df"][ts:te]
            if len(df_test) < 168:
                continue
            var_test = d["var"][ts:te]
            skew_test = d["skew"][ts:te]
            var_tr = d["var"][tr_s:tr_e].dropna()
            skew_tr = d["skew"][tr_s:tr_e].dropna()
            var_thr = np.percentile(var_tr, VAR_PCT) if len(var_tr) > 50 else np.inf
            skew_thr = np.percentile(skew_tr, SKEW_PCT) if len(skew_tr) > 50 else -np.inf

            for g in GATES:
                blk = blocked_set(g, var_test, skew_test, var_thr, skew_thr, cusum_alarms)
                tr = collect_gated(_lbl(sym), df_test, d["pm"][ts:te], d["vm"][ts:te],
                                   p_train, d["vm"][tr_s:tr_e], k1, k5, k10, blk)
                for t in tr:
                    t["period"] = period
                    t["block"] = f"{_lbl(sym)}|{period}"
                win_trades[g].extend(tr)

        for g in GATES:
            trades_by_gate[g].extend(win_trades[g])
            st = pooled_stats(win_trades[g], COST_LO, 1.0)
            per_window_net[g][period] = (st["mean_bps"] if st else 0.0, len(win_trades[g]))

    # ── 게이트별 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("  게이트별 요약 (베이스 G0 대비 — 정상레짐 엣지·꼬리·보존)")
    print("=" * 86)
    print(f"  {'게이트':>4} {'거래':>5} {'유지%':>6} {'gross':>7} {'net0.1%':>8} {'net0.2%':>8} "
          f"{'t':>6} {'평균CI(0.1%)':>18} {'최악윈도우':>9}")
    print("  " + "-" * 82)
    base_n = pooled_stats(trades_by_gate["G0"], COST_LO, years_full)["n"]
    summary = {}
    for g in GATES:
        st = pooled_stats(trades_by_gate[g], COST_LO, years_full)
        st2 = pooled_stats(trades_by_gate[g], COST_HI, years_full)
        ci = block_bootstrap(trades_by_gate[g], COST_LO, years_full)
        worst = min(v[0] for v in per_window_net[g].values()) if per_window_net[g] else 0.0
        keep = st["n"] / base_n * 100 if base_n else 0
        ci_s = f"[{ci['mean_lo']:+.1f},{ci['mean_hi']:+.1f}]" if ci else "—"
        flag = " ✅" if (ci and ci["mean_lo"] > 0) else ""
        print(f"  {g:>4} {st['n']:>5} {keep:>5.0f}% {st['gross_bps']:>+7.0f} "
              f"{st['mean_bps']:>+8.1f} {st2['mean_bps']:>+8.1f} {st['t']:>+6.2f} "
              f"{ci_s:>18}{flag} {worst:>+8.1f}")
        summary[g] = {"st": st, "st2": st2, "ci": ci, "worst": worst, "keep": keep}
    print("  " + "-" * 82)
    print("  유지%=G0 대비 남은 거래 비율(보수성) | 최악윈도우=최저 윈도우 net(꼬리) | ✅=CI하한>0")

    # ── 윈도우별 net 비교 (어떤 레짐이 걸러졌나) ──────────────────────────────
    print("\n" + "=" * 86)
    print("  윈도우별 net@0.1% (게이트가 어떤 레짐을 거르나)")
    print("=" * 86)
    periods = sorted(per_window_net["G0"].keys())
    print(f"  {'윈도우':>8}" + "".join(f"{g:>9}" for g in GATES) + "   (괄호=거래수)")
    print("  " + "-" * 78)
    for p in periods:
        row = f"  {p:>8}"
        for g in GATES:
            net, n = per_window_net[g][p]
            row += f"{net:>+7.0f}"
            row += f"·{n:<2}" if n < 100 else f"{n:>3}"
        # baseline 부호 표시
        b_net = per_window_net["G0"][p][0]
        tag = "  ← 손실레짐" if b_net < -5 else ("  ← 수익레짐" if b_net > 5 else "")
        print(row + tag)

    # ── 손실/수익 레짐 분리 효과 ──────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("  레짐 분리 효과 (G0 기준 손실/수익 윈도우에서 게이트별 net 합)")
    print("=" * 86)
    loss_periods = [p for p in periods if per_window_net["G0"][p][0] < -5]
    gain_periods = [p for p in periods if per_window_net["G0"][p][0] > 5]
    print(f"  손실레짐({len(loss_periods)}개): {', '.join(loss_periods)}")
    print(f"  수익레짐({len(gain_periods)}개): {', '.join(gain_periods)}")
    print(f"\n  {'게이트':>4} {'손실레짐 net합':>14} {'수익레짐 net합':>14}  (이상적: 손실↓0, 수익 보존)")
    print("  " + "-" * 60)
    for g in GATES:
        loss_sum = sum(per_window_net[g][p][0] for p in loss_periods)
        gain_sum = sum(per_window_net[g][p][0] for p in gain_periods)
        print(f"  {g:>4} {loss_sum:>+14.0f} {gain_sum:>+14.0f}")

    print("\n[시각화 생성...]")
    plot_results(summary, per_window_net, periods)
    print("\nExp33 완료!")


def plot_results(summary, per_window_net, periods):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)
    pal = {"G0": "#888888", "G1": "#4878cf", "G2": "#6acc65", "G3": "#ee854a"}

    # (1) 게이트별 평균CI
    ax1 = fig.add_subplot(gs[0, 0])
    for i, g in enumerate(GATES):
        st = summary[g]["st"]; ci = summary[g]["ci"]
        if ci:
            ax1.errorbar(i, st["mean_bps"],
                         yerr=[[st["mean_bps"] - ci["mean_lo"]], [ci["mean_hi"] - st["mean_bps"]]],
                         fmt="o", color=pal[g], capsize=7, ms=10, lw=2)
    ax1.axhline(0, color="red", lw=1.0, ls="--")
    ax1.set_xticks(range(len(GATES))); ax1.set_xticklabels(GATES)
    ax1.set_ylabel("거래당 평균 net (bps) @0.1%")
    ax1.set_title("정상레짐 엣지: CI 하한이 0 위로?")
    ax1.grid(axis="y", alpha=0.3)

    # (2) 최악 윈도우(꼬리) vs 거래유지율
    ax2 = fig.add_subplot(gs[0, 1])
    for g in GATES:
        ax2.scatter(summary[g]["keep"], summary[g]["worst"], s=140, color=pal[g], zorder=5)
        ax2.annotate(g, (summary[g]["keep"], summary[g]["worst"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_xlabel("거래 유지율 (%) — 낮을수록 보수적")
    ax2.set_ylabel("최악 윈도우 net (bps) — 높을수록 꼬리 약함")
    ax2.set_title("보수성 vs 꼬리 방어 (우상향이 좋음)")
    ax2.grid(alpha=0.3)

    # (3) 윈도우별 net (G0 vs G2)
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(periods))
    w = 0.2
    for j, g in enumerate(GATES):
        nets = [per_window_net[g][p][0] for p in periods]
        ax3.bar(x + (j - 1.5) * w, nets, w, color=pal[g], label=g)
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xticks(x); ax3.set_xticklabels(periods, rotation=55, fontsize=7)
    ax3.set_ylabel("윈도우 net@0.1% (bps)")
    ax3.set_title("윈도우별 net — 게이트가 손실레짐을 거르고 수익레짐을 보존하는가")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp33: 레짐 조건부 안전 게이트 (이상치 사전제외 + 정상레짐 엣지)", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp33_regime_gate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
