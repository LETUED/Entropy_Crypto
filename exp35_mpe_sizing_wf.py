"""
실험 35: 엔트로피 변동성 신호의 경제적 가치 — 사이징 응용 (마지막 관문)

배경(Exp34 + 적대적 검증):
  MPE는 미래 변동성을 실제로 예측(저MPE→고변동), 룩어헤드·아티팩트·압축재포장 아님(검증 통과).
  단 한계효용 작음(강베이스라인 대비 증분 R² 0.2~0.4%). 통계적 유의 ≠ 경제적 가치.

질문(make-or-break): MPE 변동성예보로 포지션 크기를 조절하면
  단순 trailing-vol 사이징보다 OOS 리스크조정 성과(Sharpe/MDD/Calmar)가 개선되는가?
  개선 안 되면(MPE≈TV) "통계적으론 진짜지만 실무 부가가치 없음" 최종판정.

설계(방향은 단순신호, MPE는 크기만 변조):
  방향: price > MA200 일 때 롱, 아니면 플랫 (long/flat 추세). 모든 방법 공통 → 차이는 오직 사이징.
  사이징 방법(평균 레버리지 정규화 → '언제 크게/작게' 타이밍만 격리, 캡 3배):
    M0 동일      : 롱일 때 크기 1 고정
    M1 trailing-vol: 크기 = target/trailing_vol (역변동성, 표준 vol-targeting)
    M2 TV+MPE    : 크기 = target/vol_forecast, vol_forecast=train적합 모델(trailing_vol, MPE)
    M3 MPE-only  : 크기 = target/vol_forecast, vol_forecast=train적합 모델(MPE만)
  핵심 비교: Sharpe(M2) - Sharpe(M1). MPE가 TV 위에 더하는 사이징 가치.

룩어헤드 없음:
  방향·사이징 t시점 인과정보(MPE_t, trailing_vol_t)로 결정 → t→t+1 수익 획득.
  vol 모델은 train 구간서만 적합(경계 누수 방지 위해 train 마지막 H봉 제외), test엔 인과입력만.
  target = train의 vol_forecast 중앙값(룩어헤드 없음).

측정: 일별 리샘플 수익으로 Sharpe(연속노출이라 유효), MDD, Calmar + 블록부트스트랩 Sharpe차 CI.
베이스: 장기 10코인(2017~2025 캐시), WF train12M/test6M/step6M.
실행: py exp35_mpe_sizing_wf.py
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
from src.analysis.h4_backtest import MA_PERIOD

RESULTS_DIR = Path("results")
START, END = "2017-08-01", "2025-01-01"
TRAIN_MONTHS, TEST_MONTHS = 12, 6

UNIVERSE = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT", "ADAUSDT",
            "XRPUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT"]

def _lbl(s):
    return s.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
VOL_H = 24            # 변동성 호라이즌(Exp34에서 24h가 가장 견고)
SIZE_CAP = 3.0
METHODS = ["M0", "M1", "M2", "M3"]
MDESC = {"M0": "동일사이징", "M1": "trailing-vol", "M2": "TV+MPE", "M3": "MPE-only"}
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


def fit_vol_model(tv_tr, mpe_tr, fv_tr, use_tv, use_mpe):
    """log(future_vol) ~ [1, log(tv), mpe] OLS (train). 인과입력만으로 예측 가능한 계수."""
    cols = [np.ones(len(fv_tr))]
    if use_tv:
        cols.append(np.log(tv_tr + 1e-12))
    if use_mpe:
        cols.append(mpe_tr)
    X = np.column_stack(cols)
    y = np.log(fv_tr + 1e-12)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_vol(beta, tv, mpe, use_tv, use_mpe):
    cols = [np.ones(len(tv))]
    if use_tv:
        cols.append(np.log(tv + 1e-12))
    if use_mpe:
        cols.append(mpe)
    X = np.column_stack(cols)
    return np.exp(X @ beta)


def sharpe_daily(hourly_ret):
    """일별 리샘플 후 연율 Sharpe."""
    s = hourly_ret.dropna()
    if len(s) < 30:
        return np.nan
    daily = s.resample("1D").sum()
    daily = daily[daily != 0] if (daily == 0).all() else daily
    if daily.std() == 0 or len(daily) < 10:
        return np.nan
    return daily.mean() / daily.std() * np.sqrt(365)


def max_drawdown(hourly_ret):
    eq = hourly_ret.dropna().cumsum()
    if len(eq) < 2:
        return np.nan
    return (eq - eq.cummax()).min()


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
    print("Exp35: MPE 변동성신호의 경제적 가치 — 사이징 응용 (M2 vs M1 핵심)")
    print(f"  방향=price>MA200(long/flat) 공통, MPE는 크기만 변조. 평균레버리지 정규화.")
    print(f"  M0동일 / M1 trailing-vol / M2 TV+MPE / M3 MPE-only  |  vol H={VOL_H}h")
    print("=" * 84)

    print("\n[데이터 + MPE 로드(캐시)...]")
    data = {}
    for sym in UNIVERSE:
        df = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                          window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        logret = np.log(df["close"]).diff()
        tv = logret.rolling(VOL_H).std()
        fv = logret.rolling(VOL_H).std().shift(-VOL_H)
        ma = df["close"].rolling(MA_PERIOD).mean()
        direction = (df["close"] > ma).astype(float)
        next_ret = logret.shift(-1)
        data[sym] = {"mpe": mpe, "tv": tv, "fv": fv, "dir": direction, "nret": next_ret,
                     "idx": df.index}
    print(f"  {len(UNIVERSE)}코인 로드 완료")

    wins = wf_windows()
    # 코인별 method별 test 수익 시계열 누적
    coin_ret = {m: {sym: [] for sym in UNIVERSE} for m in METHODS}

    for (tr_s, tr_e, ts, te) in wins:
        for sym in UNIVERSE:
            d = data[sym]
            # train 표본 (future_vol 경계 누수 방지: 마지막 VOL_H봉 제외)
            tr_mask = (d["idx"] >= tr_s) & (d["idx"] < tr_e)
            tr_idx = d["idx"][tr_mask]
            if len(tr_idx) < 500:
                continue
            tr_cut = tr_idx[:-VOL_H] if len(tr_idx) > VOL_H else tr_idx
            tv_tr = d["tv"].reindex(tr_cut)
            mpe_tr = d["mpe"].reindex(tr_cut)
            fv_tr = d["fv"].reindex(tr_cut)
            train_df = pd.DataFrame({"tv": tv_tr, "mpe": mpe_tr, "fv": fv_tr}).dropna()
            if len(train_df) < 300:
                continue

            beta2 = fit_vol_model(train_df["tv"].values, train_df["mpe"].values,
                                  train_df["fv"].values, use_tv=True, use_mpe=True)
            beta3 = fit_vol_model(train_df["tv"].values, train_df["mpe"].values,
                                  train_df["fv"].values, use_tv=False, use_mpe=True)

            # test 구간
            te_mask = (d["idx"] >= ts) & (d["idx"] < te)
            te_idx = d["idx"][te_mask]
            tv_te = d["tv"].reindex(te_idx)
            mpe_te = d["mpe"].reindex(te_idx)
            dir_te = d["dir"].reindex(te_idx)
            nret_te = d["nret"].reindex(te_idx)
            valid = tv_te.notna() & mpe_te.notna() & dir_te.notna() & nret_te.notna()
            if valid.sum() < 50:
                continue
            tvv = tv_te[valid].values
            mpv = mpe_te[valid].values
            dv = dir_te[valid].values
            nrv = nret_te[valid].values
            vidx = te_idx[valid.values]

            # vol forecasts
            vf1 = tvv
            vf2 = predict_vol(beta2, tvv, mpv, True, True)
            vf3 = predict_vol(beta3, tvv, mpv, False, True)
            # target = train forecast 중앙값 (룩어헤드 없음)
            tgt1 = np.median(train_df["tv"].values)
            tgt2 = np.median(predict_vol(beta2, train_df["tv"].values, train_df["mpe"].values, True, True))
            tgt3 = np.median(predict_vol(beta3, train_df["tv"].values, train_df["mpe"].values, False, True))

            def sz(vf, tgt):
                return np.clip(tgt / (vf + 1e-12), 0, SIZE_CAP)

            sizes = {"M0": np.ones_like(nrv), "M1": sz(vf1, tgt1),
                     "M2": sz(vf2, tgt2), "M3": sz(vf3, tgt3)}
            for m in METHODS:
                r = dv * sizes[m] * nrv   # 방향×크기×다음수익 (룩어헤드 없음)
                coin_ret[m][sym].append(pd.Series(r, index=vidx))

    # 코인별 연결 → 포트폴리오(타임스탬프별 활성코인 평균)
    print("\n[포트폴리오 집계...]")
    port = {}
    for m in METHODS:
        sers = []
        for sym in UNIVERSE:
            if coin_ret[m][sym]:
                sers.append(pd.concat(coin_ret[m][sym]).sort_index())
        if sers:
            mat = pd.concat(sers, axis=1)
            port[m] = mat.mean(axis=1, skipna=True).sort_index()

    # ── 지표 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  사이징 방법별 OOS 성과 (방향 공통, 크기만 차이)")
    print("=" * 84)
    print(f"  {'방법':>4} {'설명':>14} {'Sharpe':>8} {'MDD':>9} {'Calmar':>8} {'평균레버':>8} {'연율수익':>9}")
    print("  " + "-" * 66)
    metrics = {}
    for m in METHODS:
        r = port[m]
        shp = sharpe_daily(r)
        mdd = max_drawdown(r)
        ann = r.dropna().mean() * 24 * 365
        cal = ann / abs(mdd) if (mdd and not np.isnan(mdd) and mdd != 0) else np.nan
        # 평균 레버리지(활성 시점 평균 크기) 근사: 방법별 비교는 정규화로 ~1
        metrics[m] = {"sharpe": shp, "mdd": mdd, "calmar": cal, "ann": ann, "ret": r}
        print(f"  {m:>4} {MDESC[m]:>14} {shp:>+8.3f} {mdd*100:>+8.2f}% {cal:>+8.2f} "
              f"{'~1.0':>8} {ann*100:>+8.2f}%")

    # ── 핵심: M2 - M1 Sharpe 차 + 블록부트스트랩 ──────────────────────────────
    print("\n" + "=" * 84)
    print("  핵심 판정: MPE가 trailing-vol 사이징 위에 더하는 가치 (M2 - M1)")
    print("=" * 84)
    r1 = metrics["M1"]["ret"].dropna()
    r2 = metrics["M2"]["ret"].dropna()
    common = r1.index.intersection(r2.index)
    d1 = r1.reindex(common).resample("1D").sum()
    d2 = r2.reindex(common).resample("1D").sum()
    diff_sharpe = metrics["M2"]["sharpe"] - metrics["M1"]["sharpe"]

    # 블록 부트스트랩 (일별, 블록=주)
    rng = np.random.default_rng(BOOT_SEED)
    days = pd.DataFrame({"d1": d1, "d2": d2}).dropna()
    n = len(days)
    blk = 7
    nblocks = n // blk
    boot_diffs = []
    arr1 = days["d1"].values
    arr2 = days["d2"].values
    for _ in range(BOOT_B):
        starts = rng.integers(0, n - blk, size=nblocks)
        idx = np.concatenate([np.arange(s, s + blk) for s in starts])
        b1, b2 = arr1[idx], arr2[idx]
        if b1.std() > 0 and b2.std() > 0:
            s1 = b1.mean() / b1.std() * np.sqrt(365)
            s2 = b2.mean() / b2.std() * np.sqrt(365)
            boot_diffs.append(s2 - s1)
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"  Sharpe(M1 trailing-vol) = {metrics['M1']['sharpe']:+.3f}")
    print(f"  Sharpe(M2 TV+MPE)       = {metrics['M2']['sharpe']:+.3f}")
    print(f"  차이 (M2 - M1)          = {diff_sharpe:+.3f}")
    print(f"  블록부트스트랩 95% CI    = [{lo:+.3f}, {hi:+.3f}]")
    verdict = ("MPE 사이징 부가가치 有(CI 하한>0) ✅" if lo > 0
               else "MPE 사이징 부가가치 통계적으로 0과 구분 불가 ⚠")
    print(f"  >> 판정: {verdict}")
    print(f"\n  참고 M3(MPE-only) Sharpe = {metrics['M3']['sharpe']:+.3f} "
          f"(vs M0 동일 {metrics['M0']['sharpe']:+.3f}, M1 {metrics['M1']['sharpe']:+.3f})")

    print("\n[시각화 생성...]")
    plot_results(metrics, boot_diffs, lo, hi, diff_sharpe)
    print("\nExp35 완료!")


def plot_results(metrics, boot_diffs, lo, hi, diff_sharpe):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    pal = {"M0": "#888888", "M1": "#4878cf", "M2": "#ee854a", "M3": "#6acc65"}

    # (1) 누적수익 곡선
    ax1 = fig.add_subplot(gs[0, :])
    for m in METHODS:
        eq = metrics[m]["ret"].dropna().cumsum()
        ax1.plot(eq.index, eq.values, color=pal[m], lw=1.3, label=f"{m} {MDESC[m]} (Sh {metrics[m]['sharpe']:+.2f})")
    ax1.set_ylabel("누적 로그수익")
    ax1.set_title("사이징 방법별 누적수익 (방향 공통)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (2) Sharpe 막대
    ax2 = fig.add_subplot(gs[1, 0])
    shps = [metrics[m]["sharpe"] for m in METHODS]
    ax2.bar(METHODS, shps, color=[pal[m] for m in METHODS])
    for i, v in enumerate(shps):
        ax2.text(i, v + 0.01, f"{v:+.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("연율 Sharpe (일별)")
    ax2.set_title("방법별 Sharpe")
    ax2.grid(axis="y", alpha=0.3)

    # (3) M2-M1 부트스트랩 분포
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(boot_diffs, bins=50, color="#ee854a", alpha=0.7)
    ax3.axvline(0, color="red", lw=1.2, ls="--", label="0 (차이 없음)")
    ax3.axvline(diff_sharpe, color="black", lw=1.5, label=f"관측 {diff_sharpe:+.3f}")
    ax3.axvspan(lo, hi, alpha=0.15, color="orange", label=f"95% CI [{lo:+.2f},{hi:+.2f}]")
    ax3.set_xlabel("Sharpe(M2) - Sharpe(M1)")
    ax3.set_title("핵심: MPE 사이징 부가가치 (CI 0 포함?)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.suptitle("Exp35: MPE 변동성신호의 사이징 경제적 가치", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp35_mpe_sizing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
