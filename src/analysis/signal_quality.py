"""
엔트로피 신호 품질 분석
- 리스크 관리(청산 전략)와 무관하게 신호 자체의 정확도 평가
- 진입 포인트에서 24H / 48H / 72H / 168H 후 승률 및 평균 수익 계산
- 랜덤 베이스라인과 비교 → 엔트로피 신호의 순수 엣지 측정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

from src.analysis.h3_validation import compute_rsi, generate_signals
from src.analysis.h4_backtest import MA_PERIOD, ENTROPY_PCT, MAX_HOLD_H

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
HORIZONS    = [24, 48, 72, 168]


def _setup_font():
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()


# ── 진입 포인트 탐색 ──────────────────────────────────────────────────────────
def _find_entries(df_test, mpe_test, hoc_test, mpe_threshold, oc_threshold):
    rsi    = compute_rsi(df_test["close"])
    signal = generate_signals(rsi)
    ma200  = df_test["close"].rolling(MA_PERIOD).mean()

    entries = []
    for idx in df_test.index:
        sig     = signal.loc[idx] if idx in signal.index else 0
        mpe_val = mpe_test.loc[idx] if idx in mpe_test.index else np.nan
        ma_val  = ma200.loc[idx]   if idx in ma200.index   else np.nan
        price   = df_test["close"].loc[idx]

        if np.isnan(mpe_val) or np.isnan(ma_val):
            continue

        oc_ok = True
        if hoc_test is not None and oc_threshold is not None:
            if idx in hoc_test.index:
                oc_ok = hoc_test.loc[idx] <= oc_threshold

        if sig == 1 and mpe_val <= mpe_threshold and price > ma_val and oc_ok:
            entries.append({"time": idx, "price": price, "mpe": mpe_val})

    return entries


def _forward_returns(close, entry_time, entry_price, horizons):
    try:
        loc = close.index.get_loc(entry_time)
    except KeyError:
        return {h: np.nan for h in horizons}

    return {
        h: (close.iloc[loc + h] - entry_price) / entry_price * 100
        if loc + h < len(close) else np.nan
        for h in horizons
    }


# ── 메인 분석 ─────────────────────────────────────────────────────────────────
def analyze_signal_quality(df, mpe, h_onchain=None,
                           train_months=12, test_months=6, step_months=6):
    results     = []
    train_delta = pd.DateOffset(months=train_months)
    test_delta  = pd.DateOffset(months=test_months)
    step_delta  = pd.DateOffset(months=step_months)

    cursor = df.index[0]
    end    = df.index[-1]

    while True:
        train_end  = cursor + train_delta
        test_start = train_end
        test_end   = test_start + test_delta

        if test_end > end:
            break

        mpe_train = mpe[cursor:train_end].dropna()
        if len(mpe_train) < 200:
            cursor += step_delta
            continue

        mpe_threshold = np.percentile(mpe_train, ENTROPY_PCT)
        oc_threshold  = None
        if h_onchain is not None:
            hoc_tr = h_onchain[cursor:train_end].dropna()
            if len(hoc_tr) > 0:
                oc_threshold = np.percentile(hoc_tr, 40)

        df_test  = df[test_start:test_end]
        mpe_test = mpe[test_start:test_end]
        hoc_test = h_onchain[test_start:test_end] if h_onchain is not None else None

        if len(df_test) < 168:
            cursor += step_delta
            continue

        # 엔트로피 신호 진입 포인트
        entries = _find_entries(df_test, mpe_test, hoc_test,
                                mpe_threshold, oc_threshold)

        close = df["close"]

        # 각 진입점 forward returns
        entry_rets = [
            _forward_returns(close, e["time"], e["price"], HORIZONS)
            for e in entries
        ]

        # 랜덤 베이스라인 (동일 기간, 3배 샘플)
        np.random.seed(42)
        valid_idx = df_test.index[MA_PERIOD:]
        n_rnd     = min(len(entries) * 3, len(valid_idx))
        rand_rets = []
        if n_rnd > 0:
            chosen = np.random.choice(len(valid_idx), n_rnd, replace=False)
            for ci in chosen:
                ri    = valid_idx[ci]
                rp    = df_test["close"].loc[ri]
                rand_rets.append(_forward_returns(close, ri, rp, HORIZONS))

        # 통계 집계
        win_rates   = {}
        avg_returns = {}
        random_wr   = {}

        for h in HORIZONS:
            ev  = [r[h] for r in entry_rets if not np.isnan(r.get(h, np.nan))]
            rv  = [r[h] for r in rand_rets  if not np.isnan(r.get(h, np.nan))]
            win_rates[h]   = np.mean([v > 0 for v in ev]) * 100 if ev else np.nan
            avg_returns[h] = np.mean(ev) if ev else np.nan
            random_wr[h]   = np.mean([v > 0 for v in rv]) * 100 if rv else np.nan

        label = (f"{test_start.strftime('%Y-%m')} ~ "
                 f"{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}")

        results.append({
            "label":         label,
            "short":         test_start.strftime("%Y-%m"),
            "test_start":    test_start,
            "n_entries":     len(entries),
            "win_rates":     win_rates,
            "avg_returns":   avg_returns,
            "random_wr":     random_wr,
            "mpe_threshold": mpe_threshold,
        })

        cursor += step_delta

    return results


# ── 출력 ──────────────────────────────────────────────────────────────────────
def print_signal_quality(results):
    print("\n" + "=" * 105)
    print(f"{'기간':<27} {'진입수':>5}  "
          f"{'24H WR':>8} {'48H WR':>8} {'72H WR':>8} {'168H WR':>9}  "
          f"{'24H Avg':>8} {'168H Avg':>9}")
    print("-" * 105)

    for r in results:
        wr, avg = r["win_rates"], r["avg_returns"]
        def _wr(h):  return f"{wr[h]:.1f}%"  if not np.isnan(wr.get(h,  np.nan)) else "  N/A"
        def _avg(h): return f"{avg[h]:+.2f}%" if not np.isnan(avg.get(h, np.nan)) else "  N/A"

        print(f"{r['label']:<27} {r['n_entries']:>5}  "
              f"{_wr(24):>8} {_wr(48):>8} {_wr(72):>8} {_wr(168):>9}  "
              f"{_avg(24):>8} {_avg(168):>9}")

    # 전체 평균 (진입 있는 구간만)
    all_wr = {h: [] for h in HORIZONS}
    all_avg= {h: [] for h in HORIZONS}
    all_rw = {h: [] for h in HORIZONS}

    for r in results:
        if r["n_entries"] == 0:
            continue
        for h in HORIZONS:
            if not np.isnan(r["win_rates"].get(h, np.nan)):
                all_wr[h].append(r["win_rates"][h])
                all_rw[h].append(r["random_wr"][h])
            if not np.isnan(r["avg_returns"].get(h, np.nan)):
                all_avg[h].append(r["avg_returns"][h])

    print("=" * 105)
    print("전체 평균 (진입 있는 구간):")
    for h in HORIZONS:
        wr  = np.mean(all_wr[h])  if all_wr[h]  else np.nan
        rw  = np.mean(all_rw[h])  if all_rw[h]  else np.nan
        avg = np.mean(all_avg[h]) if all_avg[h] else np.nan
        edge = wr - rw if not (np.isnan(wr) or np.isnan(rw)) else np.nan
        print(f"  {h:>3}H  승률 {wr:.1f}%  (랜덤 {rw:.1f}%,  엣지 {edge:+.1f}%p)  "
              f"평균수익 {avg:+.2f}%")


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_signal_quality(results, save=True):
    RESULTS_DIR.mkdir(exist_ok=True)

    valid = [r for r in results if r["n_entries"] > 0]
    n     = len(valid)
    if n == 0:
        print("진입 포인트 없음 — 그래프 생략")
        return

    HCOLORS = {24: "#58a6ff", 48: "#3fb950", 72: "#f7931a", 168: "#ff7b72"}
    labels  = [r["short"] for r in valid]
    x       = np.arange(n)

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32,
                            height_ratios=[1.6, 1.2, 1.2])

    # ── 1. 기간별 승률 (멀티 horizon) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)

    w       = 0.17
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for j, h in enumerate(HORIZONS):
        wr_v = [r["win_rates"].get(h, np.nan) for r in valid]
        ax1.bar(x + offsets[j] * w, wr_v, w,
                color=HCOLORS[h], alpha=0.85, label=f"{h}H 승률")

    rnd_24 = [r["random_wr"].get(24, np.nan) for r in valid]
    ax1.plot(x, rnd_24, color="#8b949e", linewidth=1.3,
             linestyle="--", marker="o", markersize=4, label="랜덤 24H 기준선")
    ax1.axhline(50, color="#ffffff", linewidth=0.6, linestyle=":", alpha=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="#8b949e", fontsize=9)
    ax1.set_ylabel("승률 (%)", color="#8b949e")
    ax1.set_ylim(0, 100)
    ax1.set_title(
        "기간별 엔트로피 신호 승률  vs  랜덤 베이스라인  (50% 이상 = 동전던지기 초과)",
        color="#e6edf3", fontsize=12)
    ax1.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", ncol=5)
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 2. 엣지 (승률 - 랜덤) ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)

    for h in HORIZONS:
        edges = [
            r["win_rates"].get(h, np.nan) - r["random_wr"].get(h, np.nan)
            if not (np.isnan(r["win_rates"].get(h, np.nan)) or
                    np.isnan(r["random_wr"].get(h, np.nan))) else np.nan
            for r in valid
        ]
        ax2.plot(x, edges, color=HCOLORS[h], linewidth=1.8,
                 marker="s", markersize=5, label=f"{h}H")

    ax2.axhline(0, color="#8b949e", linewidth=1.0, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=9)
    ax2.set_ylabel("엣지 (%p)", color="#8b949e")
    ax2.set_title("엔트로피 엣지  (0 이상 = 랜덤 대비 우위)", color="#e6edf3", fontsize=10)
    ax2.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 3. 평균 수익률 ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)

    for h in HORIZONS:
        avg_v = [r["avg_returns"].get(h, np.nan) for r in valid]
        ax3.plot(x, avg_v, color=HCOLORS[h], linewidth=1.8,
                 marker="o", markersize=5, label=f"{h}H")

    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color="#8b949e", fontsize=9)
    ax3.set_ylabel("평균 수익률 (%)", color="#8b949e")
    ax3.set_title("진입 후 평균 수익률", color="#e6edf3", fontsize=10)
    ax3.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax3.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 4. 진입 횟수 ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style(ax4)

    ax4.bar(x, [r["n_entries"] for r in valid], color="#627eea", alpha=0.82)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, color="#8b949e", fontsize=9)
    ax4.set_ylabel("진입 횟수", color="#8b949e")
    ax4.set_title("구간별 엔트로피 신호 진입 횟수", color="#e6edf3", fontsize=10)
    ax4.yaxis.grid(True, color="#21262d", linestyle="--")

    # ── 5. 전체 요약 테이블 ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style(ax5)
    ax5.axis("off")

    all_wr = {h: [] for h in HORIZONS}
    all_avg= {h: [] for h in HORIZONS}
    all_rw = {h: [] for h in HORIZONS}

    for r in valid:
        for h in HORIZONS:
            w_ = r["win_rates"].get(h, np.nan)
            a_ = r["avg_returns"].get(h, np.nan)
            rw_= r["random_wr"].get(h, np.nan)
            if not np.isnan(w_):  all_wr[h].append(w_)
            if not np.isnan(a_):  all_avg[h].append(a_)
            if not np.isnan(rw_): all_rw[h].append(rw_)

    def _fmt_wr(h):
        if not all_wr[h]: return "N/A"
        wr  = np.mean(all_wr[h])
        rw  = np.mean(all_rw[h]) if all_rw[h] else np.nan
        e   = f"{wr-rw:+.1f}%p" if not np.isnan(rw) else ""
        return f"{wr:.1f}%  ({e})"

    rows = [
        ["지표"] + [f"{h}H" for h in HORIZONS],
        ["평균 승률"] + [_fmt_wr(h) for h in HORIZONS],
        ["랜덤 승률"] + [
            f"{np.mean(all_rw[h]):.1f}%" if all_rw[h] else "N/A"
            for h in HORIZONS
        ],
        ["평균 수익"] + [
            f"{np.mean(all_avg[h]):+.2f}%" if all_avg[h] else "N/A"
            for h in HORIZONS
        ],
    ]

    tbl = ax5.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262d" if row == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
    ax5.set_title("전체 신호 품질 요약", color="#e6edf3", fontsize=10, pad=8)

    fig.suptitle(
        "엔트로피 신호 품질 분석  —  리스크 관리 제거, 순수 신호 정확도",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        path = RESULTS_DIR / "signal_quality.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"저장: {path}")
    plt.show()


def _style(ax):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.yaxis.grid(True, color="#21262d", linestyle="--")
    ax.xaxis.grid(True, color="#21262d", linestyle=":", linewidth=0.4)
