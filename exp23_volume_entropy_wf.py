"""
실험 23: 볼륨 엔트로피 이중 필터 Walk-forward 검증

이론 근거:
  - ScienceDirect 2013 (PE 기반 인과성 검정): 거래량 → 가격 단방향 인과성 확인
    → 볼륨이 가격보다 선행하는 독립 정보원
  - Singha (2025, arXiv:2512.15720): 주문 흐름 엔트로피로 변동성 스파이크 감지
  - "가격은 조용하지만 거래량이 집중" = 정보 거래자 활동 신호

비교 조건 (O5: BTC, SOL, AVAX, ADA, DOT):
  A (기준):   가격 MPE < 10%ile  ← Exp21/22 기준선
  B (볼륨):   가격 MPE < 10%ile  AND  볼륨 MPE < 50%ile
  C (강화):   가격 MPE < 10%ile  AND  볼륨 MPE < 30%ile
  D (역발상): 가격 MPE < 10%ile  AND  볼륨 MPE >= 50%ile
              (가격은 조용하지만 볼륨은 활발 — 정보 비대칭 신호)

  * 볼륨 MPE: 거래량 시계열에 동일 MPE(m=3, scales=[1,2,4,8], window=168) 적용
  * 볼륨 임계값은 동일 train 기간에서 계산 (look-ahead 없음)

방법: 학습 12개월 → 테스트 6개월 (2022~2024, 6구간)
실행: py exp23_volume_entropy_wf.py
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

MPE_WINDOW = 168
MPE_M      = 3
MPE_SCALES = [1, 2, 4, 8]
ENTROPY_PCT = 10   # 가격 MPE 임계값 (고정)

# 볼륨 MPE 임계값 후보
VOL_PCT_B = 50   # 조건 B: 중간 선별 (볼륨도 낮은 편)
VOL_PCT_C = 30   # 조건 C: 강화 선별 (볼륨 매우 낮음)


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 단일 윈도우 실행 ──────────────────────────────────────────────────────────
def _run_window(df_test, price_mpe_test, vol_mpe_test, h_test,
                price_mpe_train, vol_mpe_train, oc_threshold,
                k_pct1, k_pct5, k_pct10, mode="A"):
    """
    mode:
      A — 가격 MPE만 (기준)
      B — 가격 MPE + 볼륨 MPE < 50%ile (볼륨도 낮음 = 정보 집중)
      C — 가격 MPE + 볼륨 MPE < 30%ile (더 강한 선별)
      D — 가격 MPE + 볼륨 MPE >= 50%ile (가격은 조용, 볼륨은 활발 = 정보 비대칭)
    """
    rsi   = compute_rsi(df_test["close"])
    sig   = generate_signals(rsi)
    ma200 = df_test["close"].rolling(MA_PERIOD).mean()

    price_thr = np.percentile(price_mpe_train.dropna(), ENTROPY_PCT)

    # 볼륨 임계값 (train 기간 기준)
    vol_train_clean = vol_mpe_train.dropna()
    if len(vol_train_clean) > 10:
        vol_thr_50 = np.percentile(vol_train_clean, VOL_PCT_B)
        vol_thr_30 = np.percentile(vol_train_clean, VOL_PCT_C)
    else:
        vol_thr_50 = vol_thr_30 = np.nan

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity = 1.0
    position = 0
    entry_price = entry_hour = 0
    entry_kfrac = 0.0
    trades = []
    curve  = []

    for i, idx in enumerate(df_test.index):
        price   = df_test["close"].loc[idx]
        rsi_val = rsi.loc[idx]   if idx in rsi.index   else 50.0
        ma_val  = ma200.loc[idx] if idx in ma200.index else np.nan
        p_mpe   = price_mpe_test.loc[idx] if idx in price_mpe_test.index else np.nan
        v_mpe   = vol_mpe_test.loc[idx]   if idx in vol_mpe_test.index   else np.nan
        sig_val = sig.loc[idx]   if idx in sig.index   else 0

        oc_ok = True
        if h_test is not None and oc_threshold is not None and idx in h_test.index:
            oc_ok = h_test.loc[idx] <= oc_threshold

        # 볼륨 조건 평가
        if mode == "A":
            vol_ok = True
        elif mode == "B":
            vol_ok = (not np.isnan(v_mpe)) and (not np.isnan(vol_thr_50)) and (v_mpe < vol_thr_50)
        elif mode == "C":
            vol_ok = (not np.isnan(v_mpe)) and (not np.isnan(vol_thr_30)) and (v_mpe < vol_thr_30)
        else:  # D: 볼륨이 오히려 높음 (정보 비대칭 신호)
            vol_ok = (not np.isnan(v_mpe)) and (not np.isnan(vol_thr_50)) and (v_mpe >= vol_thr_50)

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
                and vol_ok):
            k_frac = _kelly(p_mpe)
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
def run_coin_wf_all(sym, df, price_mpe, vol_mpe, h_onchain):
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

        oc_threshold = None
        if h_onchain is not None:
            hoc_tr = h_onchain[train_start:train_end].dropna()
            if len(hoc_tr) > 0:
                oc_threshold = np.percentile(hoc_tr, 40)

        df_test  = df[test_start:test_end]
        p_test   = price_mpe[test_start:test_end]
        v_test   = vol_mpe[test_start:test_end]
        h_test   = h_onchain[test_start:test_end] if h_onchain is not None else None

        if len(df_test) < 168:
            cursor += test_delta
            continue

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        for mode in ["A", "B", "C", "D"]:
            eq, trades = _run_window(
                df_test, p_test, v_test, h_test,
                p_train, v_train, oc_threshold,
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
    ref = COINS[0]
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
        m = r["metrics"]
        marker = "[+]" if s > 0 else "[-]"
        print(f"  {marker} {r['label']:<20} {s:>8.3f} {m['총 수익률']:>9} "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1f}%")
    print(f"  {'-'*55}")
    print(f"  평균 Sharpe: {np.mean(sharpes):+.3f}  |  양수: {pos}/{len(sharpes)}  |  총 거래: {total_n}")
    return np.mean(sharpes), pos, total_n


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(all_port):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    mode_names = {
        "A": "A: 가격MPE만 (기준)",
        "B": "B: +볼륨MPE<50%ile",
        "C": "C: +볼륨MPE<30%ile",
        "D": "D: +볼륨MPE≥50%ile (역발상)",
    }
    palette = {"A": "#8b949e", "B": "#56d364", "C": "#f0c040", "D": "#f78166"}

    labels = [r["short_lbl"] for r in next(iter(all_port.values()))]
    n_wins = len(labels)

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    # 1. 누적 수익 비교
    ax1 = fig.add_subplot(gs[0, :])
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

    # 2. 구간별 Sharpe
    ax2 = fig.add_subplot(gs[1, :])
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
    ax2.set_title("구간별 포트폴리오 Sharpe (볼륨 엔트로피 조건별)", color="#e6edf3")
    ax2.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.set_ylabel("Sharpe Ratio", color="#8b949e")

    # 3. 평균 Sharpe 요약
    ax3 = fig.add_subplot(gs[2, 0])
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
    ax3.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=8)
    ax3.set_title("조건별 평균 Sharpe", color="#e6edf3")
    ax3.set_ylabel("평균 Sharpe", color="#8b949e")

    # 4. 거래수 비교
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    total_trades = [sum(r["n_trades"] for r in prs) for prs in all_port.values()]
    bars = ax4.bar(range(4), total_trades, color=[palette[m] for m in all_port], alpha=0.85)
    for bar, v in zip(bars, total_trades):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([mode_names[m] for m in all_port], color="#8b949e", fontsize=8)
    ax4.set_title("조건별 총 거래수", color="#e6edf3")
    ax4.set_ylabel("총 거래수", color="#8b949e")

    fig.suptitle(
        "Exp23: 볼륨 엔트로피 이중 필터 Walk-forward (O5 포트폴리오)\n"
        "A(가격MPE만) vs B(+볼륨<50%) vs C(+볼륨<30%) vs D(역발상: 볼륨≥50%)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp23_volume_entropy_wf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n차트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp23: 볼륨 엔트로피 이중 필터 Walk-forward 검증")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기간: {START} ~ {END}")
    print("  조건 A: 가격 MPE < 10%ile (기준)")
    print("  조건 B: A + 볼륨 MPE < 50%ile  (볼륨도 낮음)")
    print("  조건 C: A + 볼륨 MPE < 30%ile  (볼륨 매우 낮음)")
    print("  조건 D: A + 볼륨 MPE >= 50%ile (역발상: 볼륨 활발)")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print(f"\n[코인 데이터 + 볼륨 MPE 계산...]")
    coin_data = {}
    for sym in COINS:
        df = collect(sym, "1h", START, END)

        # 가격 MPE (기존 캐시 사용)
        price_mpe = rolling_mpe(df["close"], window=MPE_WINDOW,
                                m=MPE_M, scales=MPE_SCALES,
                                cache_key=f"{sym}_1h_{START}_{END}")

        # 볼륨 MPE (거래량 시계열에 동일 파라미터 적용)
        vol_series = df["volume"].copy()
        # 볼륨 0 방지 (log 스케일 아님, 순열 기반이므로 0도 괜찮지만 일관성 위해)
        vol_series = vol_series.replace(0, np.nan).ffill()
        vol_mpe = rolling_mpe(vol_series, window=MPE_WINDOW,
                              m=MPE_M, scales=MPE_SCALES,
                              cache_key=f"{sym}_1h_volume_{START}_{END}")

        h = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )
        coin_data[sym] = {"df": df, "price_mpe": price_mpe, "vol_mpe": vol_mpe, "h": h}

        p_valid = price_mpe.notna().sum()
        v_valid = vol_mpe.notna().sum()
        print(f"  [{LABEL[sym]:6s}] 데이터 {len(df):,}봉 | "
              f"가격MPE {p_valid:,}봉 | 볼륨MPE {v_valid:,}봉")

        # 볼륨 MPE 분포 확인
        v_clean = vol_mpe.dropna()
        if len(v_clean) > 0:
            print(f"           볼륨MPE 통계: min={v_clean.min():.4f} "
                  f"mean={v_clean.mean():.4f} max={v_clean.max():.4f} "
                  f"| 가격MPE mean={price_mpe.dropna().mean():.4f}")

    print("\n[Walk-forward 실행 중 (4개 조건 × 5코인)...]")
    coin_results_by_mode = {}
    for sym in COINS:
        d = coin_data[sym]
        results = run_coin_wf_all(sym, d["df"], d["price_mpe"], d["vol_mpe"], d["h"])
        coin_results_by_mode[sym] = results

        for mode in ["A", "B", "C", "D"]:
            sharpes = [float(w["metrics"]["Sharpe"]) for w in results[mode]]
            n_total = sum(w["n_trades"] for w in results[mode])
            pos     = sum(s > 0 for s in sharpes)
            print(f"  [{LABEL[sym]:6s}] {mode}: 평균 Sharpe {np.mean(sharpes):+.3f} | "
                  f"양수 {pos}/{len(sharpes)} | 거래 {n_total}")

    print("\n[포트폴리오 집계...]")
    all_port = {}
    summary  = {}
    baseline = None

    mode_desc = {
        "A": "A: 가격MPE만 (기준)     ",
        "B": "B: +볼륨MPE<50%ile     ",
        "C": "C: +볼륨MPE<30%ile     ",
        "D": "D: +볼륨MPE>=50%ile    ",
    }

    for mode in ["A", "B", "C", "D"]:
        port = aggregate_portfolio(coin_results_by_mode, mode)
        all_port[mode] = port
        avg_s, pos, total_n = print_results(port, mode_desc[mode].strip())
        summary[mode] = {"avg_sharpe": avg_s, "pos": pos, "total_n": total_n}
        if mode == "A":
            baseline = avg_s

    # 최종 비교
    print(f"\n{'='*70}")
    print("최종 비교 요약")
    print(f"{'='*70}")
    for mode, s in summary.items():
        diff   = s["avg_sharpe"] - baseline
        n_wins = len(all_port[mode])
        if mode == "A":
            tag = "  (기준)"
        elif diff >= 0.05:
            tag = "★ 개선"
        elif diff <= -0.05:
            tag = "▼ 열위"
        else:
            tag = "≈ 동일"
        print(f"  {tag} {mode_desc[mode]}  Sharpe {s['avg_sharpe']:+.3f}  "
              f"양수 {s['pos']}/{n_wins}  거래 {s['total_n']}  "
              f"{'(' + ('+' if diff >= 0 else '') + f'{diff:.3f} vs A)' if mode != 'A' else ''}")

    best = max(summary.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"\n  최고 조건: {best[0]}  (Sharpe {best[1]['avg_sharpe']:+.3f})")

    if best[0] == "A":
        print("  → 볼륨 엔트로피 추가 효과 없음 — 가격 MPE만으로 충분")
    elif best[0] in ("B", "C"):
        print(f"  → 볼륨 MPE 낮음 조건이 유효 — 가격+볼륨 동시 비효율 신호 작동")
    else:
        print("  → 역발상 조건(D)이 유효 — 가격 조용 + 볼륨 활발이 더 강한 신호")

    # 볼륨 엔트로피 독립성 분석
    print(f"\n{'='*70}")
    print("볼륨 MPE 독립성 분석 (가격 MPE와 상관관계)")
    print(f"{'='*70}")
    for sym in COINS:
        d = coin_data[sym]
        pm = d["price_mpe"].dropna()
        vm = d["vol_mpe"].dropna()
        common = pm.index.intersection(vm.index)
        if len(common) > 100:
            corr = pm.loc[common].corr(vm.loc[common])
            print(f"  [{LABEL[sym]:6s}] 가격MPE vs 볼륨MPE 상관계수: {corr:+.4f}  "
                  f"({'독립적' if abs(corr) < 0.3 else '상관 있음'})")

    plot_results(all_port)


if __name__ == "__main__":
    main()
