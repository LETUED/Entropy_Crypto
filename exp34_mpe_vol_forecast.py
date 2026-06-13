"""
실험 34: 엔트로피=변동성 신호 가설 검정 (foundational, 비용·매매 없음)

질문: MPE[t](인과적)가 미래 실현변동성(magnitude)을 예측하는가?
  그리고 단순 '변동성 지속성(trailing vol)'을 넘어서는 증분(incremental) 예측력이 있는가?

배경:
  Exp30~33 수렴 — MPE를 '방향성 진입 트리거'로 쓴 응용은 비용 넘는 단독 알파 미지지.
  검증된 단서(Singha 2025): 엔트로피는 '방향'이 아니라 '변동 크기'를 예측(저엔트로피→변동 2.89배).
  프로젝트 철학과도 일치: "엔트로피=타이밍/크기, 방향은 RSI".
  → 엔트로피를 '트리거'가 아니라 '상태/리스크(변동성) 측정'으로 재배치. 그 전제를 먼저 검정.

검정 설계 (룩어헤드 없음):
  예측변수: MPE[t] (과거 168봉, 캐시) — 인과적
  타깃:     미래 실현변동성 FV[t,H] = (t, t+H] 로그수익률 std — 측정 대상(미래는 타깃이라 정당)
  통제:     trailing vol TV[t] = (t-H, t] 로그수익률 std — 인과적
  지표:
    1) 코인별 Spearman(MPE, FV_H), H=24/168/720h. 부호·크기·코인간 일관성.
    2) 디사일 분석: MPE 10분위별 평균 FV — 단조성 + 스프레드 배수(Singha 2.89배 비교).
    3) 증분 예측력(핵심): 회귀 FV ~ TV vs FV ~ TV + MPE → MPE 계수 유의성·증분 R².
    4) 정직한 유의성: 비중첩(매 H봉) 부분표본 Spearman (중첩 자기상관 제거).

베이스: 장기 10코인(BTC ETH LTC NEO ADA XRP EOS XLM TRX ETC, 2017~2025 캐시).
실행: py exp34_mpe_vol_forecast.py
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
from scipy import stats

from src.data.binance_collector import collect
from src.entropy.calculators import rolling_mpe

RESULTS_DIR = Path("results")
START, END = "2017-08-01", "2025-01-01"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT", "ADAUSDT",
            "XRPUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT"]

def _lbl(s):
    return s.replace("USDT", "")

MPE_WINDOW = 168
MPE_M, MPE_TAU, MPE_SCALES = 3, 1, [1, 2, 4, 8]
HORIZONS = [24, 168, 720]   # 1일, 1주, 1개월


def _setup_font():
    try:
        cands = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if cands:
            plt.rcParams["font.family"] = fm.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def build_frame(df, mpe, H):
    """MPE[t], TV[t]=과거H std, FV[t]=미래H std 정렬 (룩어헤드 없음)."""
    logret = np.log(df["close"]).diff()
    tv = logret.rolling(H).std()                    # (t-H, t]  인과적
    fv = logret.rolling(H).std().shift(-H)          # (t, t+H]  타깃(미래)
    frame = pd.DataFrame({"mpe": mpe, "tv": tv, "fv": fv}).dropna()
    return frame


def partial_incremental(frame):
    """FV ~ TV (베이스) vs FV ~ TV + MPE. MPE 증분 R²·t값. 랭크 변환으로 robust."""
    r_mpe = stats.rankdata(frame["mpe"].values)
    r_tv  = stats.rankdata(frame["tv"].values)
    r_fv  = stats.rankdata(frame["fv"].values)
    n = len(r_fv)
    # 표준화
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    Z_mpe, Z_tv, Z_fv = z(r_mpe), z(r_tv), z(r_fv)

    # 베이스: FV ~ TV
    b_tv = np.corrcoef(Z_tv, Z_fv)[0, 1]
    r2_base = b_tv ** 2
    # full: FV ~ TV + MPE (정규방정식)
    X = np.column_stack([np.ones(n), Z_tv, Z_mpe])
    beta, *_ = np.linalg.lstsq(X, Z_fv, rcond=None)
    pred = X @ beta
    ss_res = np.sum((Z_fv - pred) ** 2)
    ss_tot = np.sum((Z_fv - Z_fv.mean()) ** 2)
    r2_full = 1 - ss_res / ss_tot
    # MPE 계수 t값 (OLS)
    resid = Z_fv - pred
    sigma2 = ss_res / (n - 3)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_mpe = np.sqrt(sigma2 * XtX_inv[2, 2])
    t_mpe = beta[2] / (se_mpe + 1e-12)
    return {"r2_base": r2_base, "r2_full": r2_full,
            "incr_r2": r2_full - r2_base, "beta_mpe": beta[2], "t_mpe": t_mpe}


def main():
    print("=" * 84)
    print("Exp34: 엔트로피=변동성 신호 가설 검정 (MPE → 미래 실현변동성)")
    print(f"  베이스: {', '.join(_lbl(s) for s in UNIVERSE)} (2017~2025)")
    print(f"  예측변수 MPE[t](인과적) → 타깃 미래 실현변동성. 통제: trailing vol")
    print(f"  지표: Spearman / 디사일 스프레드 / TV 대비 증분 / 비중첩 유의성")
    print("=" * 84)

    print("\n[데이터 + MPE 로드(캐시)...]")
    coins = {}
    for sym in UNIVERSE:
        df = collect(sym, "1h", START, END)
        mpe = rolling_mpe(df["close"], m=MPE_M, tau=MPE_TAU, scales=MPE_SCALES,
                          window=MPE_WINDOW, cache_key=f"{sym}_1h_{START}_{END}")
        coins[sym] = (df, mpe)
    print(f"  {len(UNIVERSE)}코인 로드 완료")

    results = {H: [] for H in HORIZONS}
    decile_curves = {H: [] for H in HORIZONS}

    for H in HORIZONS:
        for sym in UNIVERSE:
            df, mpe = coins[sym]
            frame = build_frame(df, mpe, H)
            if len(frame) < 1000:
                continue
            # 1) Spearman(MPE, FV)
            rho, _ = stats.spearmanr(frame["mpe"], frame["fv"])
            # 2) 디사일 스프레드: MPE 하위10% vs 상위10%의 평균 FV
            q10 = frame["mpe"].quantile(0.10)
            q90 = frame["mpe"].quantile(0.90)
            fv_low_mpe  = frame.loc[frame["mpe"] <= q10, "fv"].mean()   # 저MPE
            fv_high_mpe = frame.loc[frame["mpe"] >= q90, "fv"].mean()   # 고MPE
            spread_ratio = fv_low_mpe / (fv_high_mpe + 1e-12)
            # 3) 증분 예측력 (TV 통제)
            pi = partial_incremental(frame)
            # 4) 비중첩 부분표본 Spearman (매 H봉)
            sub = frame.iloc[::H]
            if len(sub) > 30:
                rho_sub, p_sub = stats.spearmanr(sub["mpe"], sub["fv"])
            else:
                rho_sub, p_sub = np.nan, np.nan
            results[H].append({
                "coin": _lbl(sym), "rho": rho, "spread": spread_ratio,
                "incr_r2": pi["incr_r2"], "t_mpe": pi["t_mpe"],
                "rho_sub": rho_sub, "p_sub": p_sub, "n_sub": len(sub),
            })
            # 디사일 곡선 (per coin)
            dec = frame.groupby(pd.qcut(frame["mpe"], 10, labels=False, duplicates="drop"))["fv"].mean()
            decile_curves[H].append(dec.values)

    # ── 리포트 ────────────────────────────────────────────────────────────────
    for H in HORIZONS:
        rows = results[H]
        if not rows:
            continue
        rhos = [r["rho"] for r in rows]
        spreads = [r["spread"] for r in rows]
        incrs = [r["incr_r2"] for r in rows]
        t_mpes = [r["t_mpe"] for r in rows]
        rho_subs = [r["rho_sub"] for r in rows if not np.isnan(r["rho_sub"])]
        p_subs = [r["p_sub"] for r in rows if not np.isnan(r["p_sub"])]

        sign = np.sign(np.median(rhos))
        consistent = sum(1 for r in rhos if np.sign(r) == sign)

        print("\n" + "=" * 84)
        print(f"  H = {H}시간 ({H//24}일) 후 실현변동성 예측")
        print("=" * 84)
        print(f"  {'코인':>6} {'Spearman':>9} {'저/고MPE배수':>12} {'증분R²(vsTV)':>13} {'MPE t값':>9} {'비중첩ρ':>9}")
        print("  " + "-" * 66)
        for r in rows:
            print(f"  {r['coin']:>6} {r['rho']:>+9.3f} {r['spread']:>12.2f} "
                  f"{r['incr_r2']*100:>12.3f}% {r['t_mpe']:>+9.1f} {r['rho_sub']:>+9.3f}")
        print("  " + "-" * 66)
        print(f"  중앙값 Spearman: {np.median(rhos):+.3f}  |  부호 일관성: {consistent}/{len(rhos)} 코인")
        print(f"  평균 저/고MPE 변동성 배수: {np.mean(spreads):.2f}배  (Singha 참조값 2.89배)")
        print(f"  평균 증분 R²(TV 통제 후 MPE 기여): {np.mean(incrs)*100:.3f}%")
        print(f"  비중첩 부분표본 Spearman 중앙값: {np.median(rho_subs):+.3f}  "
              f"(p<0.05 코인: {sum(1 for p in p_subs if p<0.05)}/{len(p_subs)})")

        # 해석
        med_incr = np.median(incrs) * 100
        verdict = ("MPE가 변동성 예측에 의미있는 증분 정보 제공 가능"
                   if (med_incr > 0.1 and consistent >= 8)
                   else "MPE의 변동성 예측 증분이 미미 — 대부분 trailing vol로 설명됨")
        print(f"  >> 판정: {verdict}")

    print("\n[시각화 생성...]")
    plot_results(results, decile_curves)
    print("\nExp34 완료!")


def plot_results(results, decile_curves):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    cols = {24: "#4878cf", 168: "#ee854a", 720: "#6acc65"}

    # (1) 코인별 Spearman (호라이즌별)
    ax1 = fig.add_subplot(gs[0, 0])
    coins_lbl = [r["coin"] for r in results[HORIZONS[0]]]
    x = np.arange(len(coins_lbl))
    w = 0.25
    for j, H in enumerate(HORIZONS):
        rhos = [r["rho"] for r in results[H]]
        ax1.bar(x + (j - 1) * w, rhos, w, color=cols[H], label=f"{H//24}일")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xticks(x); ax1.set_xticklabels(coins_lbl, rotation=45, fontsize=8)
    ax1.set_ylabel("Spearman(MPE, 미래변동성)")
    ax1.set_title("MPE↔미래변동성 상관 (코인·호라이즌별)")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # (2) 증분 R² (TV 통제 후 MPE)
    ax2 = fig.add_subplot(gs[0, 1])
    for j, H in enumerate(HORIZONS):
        incrs = [r["incr_r2"] * 100 for r in results[H]]
        ax2.bar(x + (j - 1) * w, incrs, w, color=cols[H], label=f"{H//24}일")
    ax2.set_xticks(x); ax2.set_xticklabels(coins_lbl, rotation=45, fontsize=8)
    ax2.set_ylabel("증분 R² (%) — TV 통제 후 MPE 기여")
    ax2.set_title("핵심: MPE가 trailing vol 넘어 더하는 정보")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    # (3) 디사일 곡선 (168h, 코인 평균)
    ax3 = fig.add_subplot(gs[1, 0])
    for H in HORIZONS:
        curves = [c for c in decile_curves[H] if len(c) == 10]
        if curves:
            mean_curve = np.mean(curves, axis=0)
            mean_curve = mean_curve / mean_curve.mean()   # 정규화
            ax3.plot(range(1, 11), mean_curve, "o-", color=cols[H], label=f"{H//24}일")
    ax3.set_xlabel("MPE 디사일 (1=저엔트로피 → 10=고엔트로피)")
    ax3.set_ylabel("평균 미래변동성 (정규화)")
    ax3.set_title("MPE 분위별 미래변동성 (단조성?)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # (4) 저/고 MPE 변동성 배수
    ax4 = fig.add_subplot(gs[1, 1])
    for j, H in enumerate(HORIZONS):
        spreads = [r["spread"] for r in results[H]]
        ax4.bar(x + (j - 1) * w, spreads, w, color=cols[H], label=f"{H//24}일")
    ax4.axhline(1.0, color="red", lw=0.8, ls="--", label="1.0(무관)")
    ax4.set_xticks(x); ax4.set_xticklabels(coins_lbl, rotation=45, fontsize=8)
    ax4.set_ylabel("저MPE/고MPE 미래변동성 배수")
    ax4.set_title("저엔트로피 구간이 미래변동성 크나/작나")
    ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp34: 엔트로피=변동성 신호 가설 (MPE→미래 실현변동성)", fontsize=13, y=1.005)
    out = RESULTS_DIR / "exp34_mpe_vol_forecast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


if __name__ == "__main__":
    main()
