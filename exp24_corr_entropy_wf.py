"""
실험 24: Cross-asset 상관 엔트로피 레짐 감지 + Kelly 동적화

이론 근거:
  arXiv:2512.06473 (2025): 140 암호화폐 cross-correlation 엔트로피
    → 상관행렬 고유값 Shannon entropy 감소 = 시스템 리스크 상승 (Terra/FTX 사전 감지)
  MDPI 1099-4300/23/12/1674: 포트폴리오 다양화 효율성 엔트로피로 측정
  원리: 코인들이 동조화되면 5코인 포트폴리오 = 실질적으로 1개 포지션
       → 분산 효과 없음 → Kelly 유지는 과도한 리스크

기준: Exp23-D (가격MPE < 10%ile AND 볼륨MPE >= 50%ile, WF Sharpe +1.291)

비교 조건 (O5: BTC, SOL, AVAX, ADA, DOT):
  A (기준):  Exp23-D — 이전 실험 최고 성과 (복제)
  B (필터):  A + 상관 엔트로피 >= 25%ile  (시스템 리스크 구간 진입 차단)
  C (동적):  A + Kelly *= (0.5 + 0.5 × rank)  (엔트로피 낮을수록 Kelly 50%까지 축소)
  D (혼합):  B + C (필터 + 동적 Kelly 동시 적용)

상관 엔트로피:
  5코인 1H 수익률의 168H 롤링 상관 행렬 → 고유값 → Shannon Entropy
  낮은 엔트로피 = 코인 동조화 = 분산 효과 상실 = 시스템 리스크
  높은 엔트로피 = 코인 독립 = 다양화 작동 = 안전한 진입

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp24_corr_entropy_wf.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-","") not in ("utf8","utf-8"):
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
from src.analysis.h4_backtest import compute_metrics, FEE_RATE, MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi, generate_signals

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
         "ADAUSDT": "ADA", "DOTUSDT": "DOT"}

MPE_WINDOW      = 168
MPE_M           = 3
MPE_SCALES      = [1, 2, 4, 8]
ENTROPY_PCT     = 10   # 가격 MPE 임계값
VOL_PCT_D       = 50   # Exp23-D: 볼륨 MPE >= 50%ile

CORR_WINDOW     = 168  # 상관 엔트로피 롤링 윈도우 (MPE 윈도우와 동일)
CORR_FILTER_PCT = 25   # 필터 임계값: 하위 25% = 고상관 = 시스템 리스크


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 상관 엔트로피 계산 ────────────────────────────────────────────────────────
def compute_corr_entropy(all_returns: pd.DataFrame, window: int = 168) -> pd.Series:
    """
    5코인 수익률의 rolling 상관행렬 Shannon Entropy

    낮은 entropy = 코인 동조화 = 시스템 리스크 (분산 효과 없음)
    높은 entropy = 코인 독립 = 포트폴리오 다양화 작동
    이론 최대값: log(N_coins) = log(5) ≈ 1.609
    """
    vals = all_returns.values.astype(float)
    n    = len(vals)
    entropies = np.full(n, np.nan)

    for i in range(window - 1, n):
        wd      = vals[i - window + 1 : i + 1, :]   # (window, 5)
        row_ok  = ~np.any(np.isnan(wd), axis=1)
        wd_c    = wd[row_ok]
        if wd_c.shape[0] < window // 2:
            continue
        try:
            C       = np.corrcoef(wd_c.T)            # (5, 5)
            eigvals = np.linalg.eigvalsh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            p       = eigvals / eigvals.sum()
            entropies[i] = float(-np.sum(p * np.log(p)))
        except Exception:
            pass

    return pd.Series(entropies, index=all_returns.index)


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, price_mpe_test, vol_mpe_test, corr_ent_test, h_test,
                price_mpe_train, vol_mpe_train, corr_ent_train, oc_threshold,
                k_pct1, k_pct5, k_pct10, mode="A"):
    """
    mode:
      A — Exp23-D 기준 (가격MPE<10%ile AND 볼륨MPE>=50%ile)
      B — A + 상관 엔트로피 >= 25%ile 필터 (시스템 리스크 구간 차단)
      C — A + Kelly *= (0.5 + 0.5 × corr_entropy_rank)  동적 Kelly
      D — B + C 혼합 (필터 + 동적 Kelly 동시 적용)
    """
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    price_thr = np.percentile(price_mpe_train.dropna(), ENTROPY_PCT)

    vol_train_clean = vol_mpe_train.dropna()
    vol_thr_50      = (np.percentile(vol_train_clean, VOL_PCT_D)
                       if len(vol_train_clean) > 10 else np.nan)

    # 상관 엔트로피 train 기간 분포 → 필터 임계값 + rank 함수
    corr_clean  = corr_ent_train.dropna().values
    corr_sorted = np.sort(corr_clean) if len(corr_clean) > 0 else np.array([])
    corr_thr_25 = (np.percentile(corr_sorted, CORR_FILTER_PCT)
                   if len(corr_sorted) > 0 else np.nan)

    def _corr_rank(val):
        """현재 엔트로피 값의 train 분포 내 백분위 rank (0.0~1.0)"""
        if np.isnan(val) or len(corr_sorted) == 0:
            return 0.5  # 데이터 없으면 중립값
        return float(np.searchsorted(corr_sorted, val)) / len(corr_sorted)

    def _kelly_base(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity      = 1.0
    position    = 0
    entry_price = entry_hour = 0
    entry_kfrac = 0.0
    trades      = []
    curve       = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe   = price_mpe_test.loc[idx] if idx in price_mpe_test.index else np.nan
        v_mpe   = vol_mpe_test.loc[idx]   if idx in vol_mpe_test.index   else np.nan
        c_ent   = corr_ent_test.loc[idx]  if idx in corr_ent_test.index  else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        # Exp23-D 볼륨 조건 (A/B/C/D 공통)
        vol_ok = (not np.isnan(v_mpe)
                  and not np.isnan(vol_thr_50)
                  and v_mpe >= vol_thr_50)

        # 상관 엔트로피 필터 (B, D)
        corr_ok = True
        if mode in ("B", "D"):
            corr_ok = (not np.isnan(c_ent)
                       and not np.isnan(corr_thr_25)
                       and c_ent >= corr_thr_25)

        # 동적 Kelly 스케일 (C, D): rank 0~1 → scale 0.5~1.0
        kelly_scale = 1.0
        if mode in ("C", "D"):
            kelly_scale = 0.5 + 0.5 * _corr_rank(c_ent)

        # 청산
        if position == 1:
            held = i - entry_hour
            if rsi_val > 50 or held >= MAX_HOLD_H:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)
                trades.append({"pnl_pct": pnl * 100, "held_h": held})
                position = 0

        # 진입
        if (position == 0
                and not np.isnan(p_mpe)
                and not np.isnan(ma_val)
                and p_mpe <= price_thr
                and price > ma_val
                and oc_ok
                and sig_val == 1
                and vol_ok
                and corr_ok):
            k_frac = _kelly_base(p_mpe) * kelly_scale
            if k_frac > 0:
                position    = 1
                entry_price = price * (1 + FEE_RATE)
                entry_hour  = i
                entry_kfrac = k_frac

        curve.append(equity)

    if position == 1:
        pnl = (df_test["close"].iloc[-1] - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - FEE_RATE)

    return pd.Series(curve, index=df_test.index), trades


# ── 단일 코인 Walk-forward ────────────────────────────────────────────────────
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, corr_entropy, h_onchain):
    results_by_mode = {m: [] for m in ["A", "B", "C", "D"]}

    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta  = pd.DateOffset(months=TEST_MONTHS)
    cursor = pd.Timestamp(START)
    end    = pd.Timestamp(END)

    while True:
        train_start = cursor
        train_end   = cursor + train_delta
        test_start  = train_end
        test_end    = test_start + test_delta
        if test_end > end:
            break

        p_train = price_mpe[train_start:train_end].dropna()
        if len(p_train) < 200:
            cursor += test_delta
            continue

        k_pct1  = np.percentile(p_train, 1)
        k_pct5  = np.percentile(p_train, 5)
        k_pct10 = np.percentile(p_train, 10)

        v_train = vol_mpe[train_start:train_end]
        c_train = corr_entropy[train_start:train_end]

        oc_threshold = None
        if h_onchain is not None:
            hoc_tr = h_onchain[train_start:train_end].dropna()
            if len(hoc_tr) > 0:
                oc_threshold = np.percentile(hoc_tr, 40)

        df_test = df[test_start:test_end]
        p_test  = price_mpe[test_start:test_end]
        v_test  = vol_mpe[test_start:test_end]
        c_test  = corr_entropy[test_start:test_end]
        h_test  = h_onchain[test_start:test_end] if h_onchain is not None else None

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, p_test, v_test, c_test, h_test,
                p_train, v_train, c_train, oc_threshold,
                k_pct1, k_pct5, k_pct10, mode=mode,
            )
            results_by_mode[mode].append({
                "label":      label,
                "short_lbl":  test_start.strftime("%Y-%m"),
                "test_start": test_start,
                "test_end":   test_end,
                "equity":     eq,
                "metrics":    compute_metrics(eq),
                "trades":     trades,
                "n_trades":   len(trades),
            })

        cursor += test_delta

    return results_by_mode


# ── 포트폴리오 집계 ───────────────────────────────────────────────────────────
def aggregate_portfolio(coin_results_by_mode, mode):
    ref         = COINS[0]
    port_results = []
    for win in coin_results_by_mode[ref][mode]:
        ts, te = win["test_start"], win["test_end"]
        equities, all_pnls, total_trades = [], [], 0
        for sym in COINS:
            match = next((w for w in coin_results_by_mode[sym][mode]
                          if w["test_start"] == ts and w["test_end"] == te), None)
            if match is None:
                continue
            equities.append(match["equity"])
            total_trades += match["n_trades"]
            all_pnls.extend([t["pnl_pct"] for t in match["trades"]])
        if not equities:
            continue
        n       = len(equities)
        port_eq = sum(eq / n for eq in equities)
        wr      = np.mean([p > 0 for p in all_pnls]) * 100 if all_pnls else 0.0
        port_results.append({
            "label":      win["label"],
            "short_lbl":  win["short_lbl"],
            "test_start": ts,
            "equity":     port_eq,
            "metrics":    compute_metrics(port_eq),
            "n_trades":   total_trades,
            "win_rate":   wr,
        })
    return port_results


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(port_results, name):
    sharpes = [float(r["metrics"]["Sharpe"]) for r in port_results]
    pos     = sum(s > 0 for s in sharpes)
    total_n = sum(r["n_trades"] for r in port_results)
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'기간':<20} {'Sharpe':>8} {'수익률':>9} {'거래수':>7} {'승률':>7}")
    print(f"  {'-'*55}")
    for r, s in zip(port_results, sharpes):
        m      = r["metrics"]
        marker = "[+]" if s > 0 else "[-]"
        print(f"  {marker} {r['label']:<20} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}%")
    print(f"  {'-'*55}")
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  양수: {pos}/{len(sharpes)}  |  총 거래: {total_n}")
    return np.mean(sharpes), pos, total_n


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port, corr_entropy_series):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": "A: Exp23-D 기준",
        "B": "B: +corr_entropy≥25%ile 필터",
        "C": "C: +Kelly 동적 (entropy비례)",
        "D": "D: B+C 혼합",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#f78166"}

    labels  = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins  = len(labels)

    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. 상관 엔트로피 시계열
    ax0 = fig.add_subplot(gs[0, :])
    _style(ax0)
    ce_clean = corr_entropy_series.dropna()
    ax0.plot(ce_clean.index, ce_clean.values, color="#79c0ff", linewidth=0.6, alpha=0.8)
    ax0.axhline(ce_clean.quantile(0.25), color="#f78166", linewidth=1.0,
                linestyle="--", label=f"25%ile={ce_clean.quantile(0.25):.3f} (필터 기준)")
    ax0.axhline(np.log(5), color="#56d364", linewidth=0.8,
                linestyle=":", label=f"이론 최대={np.log(5):.3f} (완전 독립)")
    ax0.set_title("5코인 상관 엔트로피 시계열 (낮을수록 동조화 = 시스템 리스크)",
                  color="#e6edf3")
    ax0.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax0.set_ylabel("Shannon Entropy", color="#8b949e")

    # 2. 누적 수익 비교
    ax1 = fig.add_subplot(gs[1, :])
    _style(ax1)
    for mode, prs in all_port.items():
        lw = 2.2 if mode != "A" else 1.0
        ls = "-"  if mode != "A" else ":"
        al = 0.9  if mode != "A" else 0.5
        for i, r in enumerate(prs):
            ax1.plot(r["equity"].index, r["equity"].values,
                     color=palette[mode], linewidth=lw, linestyle=ls, alpha=al,
                     label=mode_names[mode] if i == 0 else "_")
    ax1.axhline(1.0, color="#8b949e", linewidth=0.6, linestyle=":")
    ax1.set_title("포트폴리오 누적 수익 비교 (점선=A 기준)", color="#e6edf3")
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=2, loc="upper left")

    # 3. 구간별 Sharpe
    ax2 = fig.add_subplot(gs[2, :])
    _style(ax2)
    x = np.arange(n_wins)
    w = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w
    for ni, (mode, prs) in enumerate(all_port.items()):
        sharpes = [float(r["metrics"]["Sharpe"]) for r in prs]
        bars = ax2.bar(x + offsets[ni], sharpes, w, color=palette[mode], alpha=0.85,
                       label=f"{mode_names[mode]}  avg {np.mean(sharpes):+.3f}")
        for bar, v in zip(bars, sharpes):
            yp = v + 0.02 if v >= 0 else v - 0.09
            ax2.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.2f}",
                     ha="center", fontsize=7, color="#e6edf3", fontweight="bold")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", rotation=30, ha="right")
    ax2.set_title("구간별 포트폴리오 Sharpe (상관 엔트로피 조건별)", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 4. 평균 Sharpe 요약
    ax3 = fig.add_subplot(gs[3, 0])
    _style(ax3)
    avg_sharpes = [np.mean([float(r["metrics"]["Sharpe"]) for r in prs])
                   for prs in all_port.values()]
    pos_counts  = [sum(float(r["metrics"]["Sharpe"]) > 0 for r in prs)
                   for prs in all_port.values()]
    bars = ax3.bar(range(4), avg_sharpes, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v, pos in zip(bars, avg_sharpes, pos_counts):
        yp = v + 0.01 if v >= 0 else v - 0.07
        ax3.text(bar.get_x() + bar.get_width()/2, yp,
                 f"{v:+.3f}\n({pos}/{n_wins})", ha="center", fontsize=9,
                 color="#e6edf3", fontweight="bold")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax3.set_title("조건별 평균 Sharpe", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")

    # 5. 거래수 비교
    ax4 = fig.add_subplot(gs[3, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(4), total_trades, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=7.5)
    ax4.set_title("조건별 총 거래수", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")

    fig.suptitle(
        "Exp24: Cross-asset 상관 엔트로피 레짐 감지 + Kelly 동적화 (O5 포트폴리오)\n"
        "A(Exp23-D 기준) vs B(필터) vs C(동적Kelly) vs D(B+C 혼합)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp24_corr_entropy_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp24: Cross-asset 상관 엔트로피 레짐 감지 + Kelly 동적화")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print("  기준 (A): Exp23-D (가격MPE<10%ile AND 볼륨MPE>=50%ile, WF +1.291)")
    print("  조건  B:  A + 상관 엔트로피 >= 25%ile  (시스템 리스크 필터)")
    print("  조건  C:  A + Kelly *= (0.5 + 0.5×rank)  (엔트로피 낮을수록 Kelly 축소)")
    print("  조건  D:  B + C 혼합")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE 계산 (캐시 사용)...]")
    coin_data = {}
    close_dict = {}

    for sym in COINS:
        df = collect(sym, "1h", START, END)

        price_mpe = rolling_mpe(df["close"], window=MPE_WINDOW,
                                m=MPE_M, scales=MPE_SCALES,
                                cache_key=f"{sym}_1h_{START}_{END}")

        vol_series = df["volume"].replace(0, np.nan).ffill()
        vol_mpe    = rolling_mpe(vol_series, window=MPE_WINDOW,
                                 m=MPE_M, scales=MPE_SCALES,
                                 cache_key=f"{sym}_1h_volume_{START}_{END}")

        h = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )

        coin_data[sym]  = {"df": df, "price_mpe": price_mpe, "vol_mpe": vol_mpe, "h": h}
        close_dict[sym] = df["close"]

        p_valid = price_mpe.notna().sum()
        v_valid = vol_mpe.notna().sum()
        print(f"  [{LABEL[sym]:6s}] {len(df):,}봉 | 가격MPE {p_valid:,}봉 | 볼륨MPE {v_valid:,}봉")

    # ── 상관 엔트로피 계산 ────────────────────────────────────────────────────
    print("\n[Cross-asset 상관 엔트로피 계산 중 (168H 롤링, ~5-30초)...]")
    closes     = pd.DataFrame({sym: close_dict[sym] for sym in COINS}).sort_index()
    returns_df = closes.pct_change().replace([np.inf, -np.inf], np.nan)

    corr_entropy = compute_corr_entropy(returns_df, window=CORR_WINDOW)

    ce_clean = corr_entropy.dropna()
    max_ent  = np.log(len(COINS))  # log(5) ≈ 1.609
    print(f"  상관 엔트로피: min={ce_clean.min():.4f}  mean={ce_clean.mean():.4f}  "
          f"max={ce_clean.max():.4f}  (이론최대={max_ent:.4f})")
    print(f"  유효값: {len(ce_clean):,}개 / {len(corr_entropy):,}개")
    print(f"  25%ile={ce_clean.quantile(0.25):.4f}  "
          f"50%ile={ce_clean.quantile(0.50):.4f}  "
          f"75%ile={ce_clean.quantile(0.75):.4f}")

    # 레짐별 엔트로피 통계 (2022-07~12 특별 확인)
    bear_slice = corr_entropy["2022-07":"2022-12"].dropna()
    if len(bear_slice) > 0:
        bear_pct  = np.mean(ce_clean.values <= bear_slice.mean()) * 100
        print(f"\n  [레짐 분석] 2022-07~12 평균 엔트로피 = {bear_slice.mean():.4f} "
              f"(전체 분포의 {bear_pct:.1f}%ile) ← 낮을수록 고상관")
        print(f"  → Terra/FTX 붕괴 구간의 동조화 수준 확인")

    bull_slice = corr_entropy["2023-01":"2023-12"].dropna()
    if len(bull_slice) > 0:
        bull_pct  = np.mean(ce_clean.values <= bull_slice.mean()) * 100
        print(f"  [레짐 분석] 2023-01~12 평균 엔트로피 = {bull_slice.mean():.4f} "
              f"(전체 분포의 {bull_pct:.1f}%ile)")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print("\n[Walk-forward 실행 중 (4개 조건 × 5코인)...]")
    coin_results_by_mode = {}

    for sym in COINS:
        d       = coin_data[sym]
        results = run_coin_wf_all(
            sym, d["df"], d["price_mpe"], d["vol_mpe"], corr_entropy, d["h"]
        )
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            print(f"  [{LABEL[sym]:6s} {mode}] "
                  f"Sharpe={np.mean(sharpes):+.3f}  양수:{pos}/6  거래:{n_total}")

    # ── 포트폴리오 집계 ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  포트폴리오 결과 (O5 균등 배분)")
    print("=" * 70)

    all_port = {}
    summary  = {}
    for mode in ["A", "B", "C", "D"]:
        port                = aggregate_portfolio(coin_results_by_mode, mode)
        all_port[mode]      = port
        avg_s, pos_cnt, total_n = print_results(port, f"조건 {mode}")
        summary[mode] = {"avg_sharpe": avg_s, "pos_cnt": pos_cnt, "total_n": total_n}

    # ── 요약 비교표 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  요약 비교 (기준: A = Exp23-D +1.291)")
    print(f"  {'조건':<6} | {'평균 Sharpe':>12} | {'개선폭':>8} | {'양수 구간':>10} | {'거래수':>6}")
    print("  " + "-" * 52)
    base_sharpe = summary["A"]["avg_sharpe"]
    for mode in ["A", "B", "C", "D"]:
        s     = summary[mode]
        delta = s["avg_sharpe"] - base_sharpe
        mark  = "★" if s["avg_sharpe"] > base_sharpe else " "
        print(f"  {mode}{mark}     | {s['avg_sharpe']:>+12.3f} | {delta:>+8.3f} | "
              f"{s['pos_cnt']}/6{' ':>7} | {s['total_n']:>6}")
    print("=" * 70)

    # 거래수 경고
    for mode in ["A", "B", "C", "D"]:
        if summary[mode]["total_n"] < 30:
            print(f"  ⚠ 조건 {mode}: 총 거래수 {summary[mode]['total_n']} < 30 → 통계적 유의성 낮음")

    print("\n[시각화 생성...]")
    plot_results(all_port, corr_entropy)
    print("\nExp24 완료!")


if __name__ == "__main__":
    main()
