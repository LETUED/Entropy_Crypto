"""
실험 26B: SHAP 피처 중요도 분석 (ML 트레이딩 없음, 순수 인사이트 도구)

목적:
  "Exp26A-B 조건(RSI<35, 69건)의 과거 거래를 만든 것이 진짜 어떤 피처였나?"
  → price_mpe vs vol_mpe vs RSI vs Stochastic vs 온체인 상대 중요도 정량화
  → 결과는 트레이딩 미사용, Exp27 규칙 설계의 인사이트로만 활용

이론 근거:
  MDPI 2079-9302/15/6/1334: LightGBM + SHAP → 피처 중요도 정량화 (Sharpe 0.938)
  arXiv:2602.00776: SHAP으로 미시구조 변수의 신호 해석성 확보
  PMC12571449: 소수샘플 환경에서 분석 전용 ML 적용 방법

입력: Exp26A-B 조건의 Walk-forward 거래 재수집
피처 (8개, 진입 시점 기준):
  1. price_mpe_rank — MPE 백분위 [0,1]
  2. vol_mpe_rank   — 볼륨MPE 백분위 [0,1]
  3. rsi            — RSI 14기간 값 (0~100)
  4. stoch_k        — Stochastic %K (0~100)
  5. ma_dist_pct    — (price - MA200) / MA200 × 100
  6. onchain_h_rank — 온체인엔트로피 백분위 [0,1]
  7. btc_return_24h — BTC 24H 수익률 (%)
  8. vol_price_ratio — vol_mpe / (price_mpe + 1e-6)

타겟: 실제 pnl_pct > 0 → 1 (승리), 아니면 → 0 (패배)

모델: LightGBM (분석 전용, 배포 없음)
출력: results/exp26b_shap_summary.png, results/exp26b_feature_importance.csv

실행: py exp26b_shap_analysis.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower().replace("-","") not in ("utf8","utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import lightgbm as lgb
import shap

from src.data.binance_collector import collect
from src.data.onchain_collector import collect_funding_rate, collect_fear_greed
from src.entropy.calculators import rolling_mpe
from src.entropy.onchain_entropy import funding_entropy, fear_greed_entropy, combined_onchain_entropy
from src.analysis.h4_backtest import compute_metrics, FEE_RATE, MA_PERIOD, MAX_HOLD_H
from src.analysis.h3_validation import compute_rsi

RESULTS_DIR  = Path("results")
START, END   = "2021-01-01", "2025-01-01"
TRAIN_MONTHS = 12
TEST_MONTHS  = 6

COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
LABEL = {"BTCUSDT": "BTC", "SOLUSDT": "SOL", "AVAXUSDT": "AVAX",
         "ADAUSDT": "ADA", "DOTUSDT": "DOT"}

MPE_WINDOW  = 168
MPE_M       = 3
MPE_SCALES  = [1, 2, 4, 8]
ENTROPY_PCT = 10
VOL_PCT_D   = 50

# Exp26B는 B 조건(RSI<35, 69건)으로 SHAP 분석
RSI_THRESHOLD = 35


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun", "NanumGothic", "AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 기술적 지표 (로컬 정의) ───────────────────────────────────────────────────
def compute_stochastic(df: pd.DataFrame, k_period: int = 14) -> pd.Series:
    low_min  = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    return 100.0 * (df["close"] - low_min) / (high_max - low_min + 1e-12)


# ── Walk-forward 거래 재수집 (피처 포함) ──────────────────────────────────────
def collect_trades_with_features(coin_data: dict) -> pd.DataFrame:
    """
    Exp26A-B 조건(RSI<35)의 모든 WF 창에서 거래 수집.
    진입 시점 8개 피처 + 결과(pnl_pct, label) 반환.

    look-ahead 없음:
    - 피처는 모두 진입 시점(t) 이전 데이터로 계산
    - 타겟(168H 미래 수익률)은 이미 완료된 거래 결과 사용
    """
    all_records = []
    btc_close   = coin_data["BTCUSDT"]["df"]["close"]

    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta  = pd.DateOffset(months=TEST_MONTHS)

    for sym in COINS:
        d          = coin_data[sym]
        df         = d["df"]
        price_mpe  = d["price_mpe"]
        vol_mpe    = d["vol_mpe"]
        stoch_k    = d["stoch_k"]
        h_onchain  = d["h"]

        rsi   = compute_rsi(df["close"])
        ma200 = df["close"].rolling(MA_PERIOD).mean()

        cursor = pd.Timestamp(START)
        end    = pd.Timestamp(END)

        while True:
            train_start = cursor
            train_end   = cursor + train_delta
            test_start  = train_end
            test_end    = test_start + test_delta
            if test_end > end:
                break

            # Train 기간 분포 계산 (피처 정규화용)
            p_train     = price_mpe[train_start:train_end].dropna()
            v_train     = vol_mpe[train_start:train_end].dropna()
            h_train     = h_onchain[train_start:train_end].dropna() if h_onchain is not None else pd.Series(dtype=float)

            if len(p_train) < 200:
                cursor += test_delta
                continue

            p_sorted = np.sort(p_train.values)
            v_sorted = np.sort(v_train.values) if len(v_train) > 0 else np.array([])
            h_sorted = np.sort(h_train.values) if len(h_train) > 0 else np.array([])

            price_thr  = np.percentile(p_sorted, ENTROPY_PCT)
            vol_thr_50 = np.percentile(v_sorted, VOL_PCT_D) if len(v_sorted) > 0 else np.nan
            oc_thr     = np.percentile(h_sorted, 40) if len(h_sorted) > 0 else np.nan

            # Kelly 기준
            k_pct1  = np.percentile(p_sorted, 1)
            k_pct5  = np.percentile(p_sorted, 5)
            k_pct10 = np.percentile(p_sorted, 10)

            def _kelly(v):
                if v <= k_pct1:  return 0.5
                if v <= k_pct5:  return 0.3
                if v <= k_pct10: return 0.15
                return 0.0

            def _p_rank(val):
                return float(np.searchsorted(p_sorted, val)) / len(p_sorted) if len(p_sorted) > 0 else 0.5

            def _v_rank(val):
                return float(np.searchsorted(v_sorted, val)) / len(v_sorted) if len(v_sorted) > 0 else 0.5

            def _h_rank(val):
                return float(np.searchsorted(h_sorted, val)) / len(h_sorted) if len(h_sorted) > 0 else 0.5

            # Test 기간에서 B 조건(RSI<35 + MPE<10% + vol>=50% + MA200 + 온체인) 진입 탐지
            df_test    = df[test_start:test_end]
            p_test     = price_mpe[test_start:test_end]
            v_test     = vol_mpe[test_start:test_end]
            stoch_test = stoch_k[test_start:test_end]
            h_test     = h_onchain[test_start:test_end] if h_onchain is not None else None
            rsi_test   = rsi[test_start:test_end]
            ma_test    = ma200[test_start:test_end]
            btc_test   = btc_close[test_start:test_end]

            position    = 0
            entry_idx   = None
            entry_price = 0.0
            entry_feats = None
            entry_hour  = 0

            for i, idx in enumerate(df_test.index):
                price   = df_test["close"].loc[idx]
                rsi_val = rsi_test.loc[idx]   if idx in rsi_test.index else 50.0
                ma_val  = ma_test.loc[idx]    if idx in ma_test.index  else np.nan
                p_mpe   = p_test.loc[idx]     if idx in p_test.index   else np.nan
                v_mpe   = v_test.loc[idx]     if idx in v_test.index   else np.nan
                stoch   = stoch_test.loc[idx] if idx in stoch_test.index else np.nan
                oc_val  = h_test.loc[idx]     if (h_test is not None and idx in h_test.index) else np.nan
                btc_r24 = np.nan
                if idx in btc_test.index:
                    btc_past_idx = btc_test.index.get_loc(idx)
                    if btc_past_idx >= 24:
                        btc_r24 = (btc_test.iloc[btc_past_idx] /
                                   btc_test.iloc[btc_past_idx - 24] - 1) * 100

                # 조건 체크
                oc_ok  = True if (np.isnan(oc_val) or oc_thr is None) else (oc_val <= oc_thr)
                vol_ok = (not np.isnan(v_mpe) and not np.isnan(vol_thr_50) and v_mpe >= vol_thr_50)

                # 청산
                if position == 1:
                    held = i - entry_hour
                    if rsi_val > 50 or held >= MAX_HOLD_H:
                        pnl = (price - entry_price) / entry_price
                        equity_pnl = pnl * _kelly(entry_feats["price_mpe_rank_raw"])
                        actual_pnl_pct = pnl * 100

                        rec = dict(entry_feats)
                        rec["pnl_pct"] = actual_pnl_pct
                        rec["label"]   = 1 if actual_pnl_pct > 0 else 0
                        rec["sym"]     = sym
                        rec["period"]  = test_start.strftime("%Y-%m")
                        rec["held_h"]  = held
                        all_records.append(rec)
                        position = 0

                # 진입 (B 조건: RSI<35)
                if (position == 0
                        and rsi_val < RSI_THRESHOLD
                        and not np.isnan(p_mpe)
                        and not np.isnan(ma_val)
                        and p_mpe <= price_thr
                        and price > ma_val
                        and oc_ok
                        and vol_ok):

                    position    = 1
                    entry_price = price * (1 + FEE_RATE)
                    entry_hour  = i
                    entry_idx   = idx

                    # 8개 피처 기록 (진입 시점, look-ahead 없음)
                    entry_feats = {
                        "price_mpe_rank":    _p_rank(p_mpe),
                        "price_mpe_rank_raw": p_mpe,   # Kelly 계산용, SHAP에는 제외
                        "vol_mpe_rank":      _v_rank(v_mpe),
                        "rsi":               rsi_val,
                        "stoch_k":           stoch if not np.isnan(stoch) else 50.0,
                        "ma_dist_pct":       (price / ma_val - 1) * 100 if not np.isnan(ma_val) else 0.0,
                        "onchain_h_rank":    _h_rank(oc_val) if not np.isnan(oc_val) else 0.5,
                        "btc_return_24h":    btc_r24 if not np.isnan(btc_r24) else 0.0,
                        "vol_price_ratio":   v_mpe / (p_mpe + 1e-6),
                    }

            # 기간 끝에 미청산 포지션 강제 청산
            if position == 1 and entry_feats is not None:
                price_last = df_test["close"].iloc[-1]
                pnl = (price_last - entry_price) / entry_price
                actual_pnl_pct = pnl * 100
                rec = dict(entry_feats)
                rec["pnl_pct"] = actual_pnl_pct
                rec["label"]   = 1 if actual_pnl_pct > 0 else 0
                rec["sym"]     = sym
                rec["period"]  = test_start.strftime("%Y-%m")
                rec["held_h"]  = len(df_test) - entry_hour
                all_records.append(rec)

            cursor += test_delta

    df_trades = pd.DataFrame(all_records)
    # price_mpe_rank_raw는 Kelly 계산용이었으므로 SHAP 피처에서 제외
    if "price_mpe_rank_raw" in df_trades.columns:
        df_trades = df_trades.drop(columns=["price_mpe_rank_raw"])
    return df_trades


# ── SHAP 분석 ────────────────────────────────────────────────────────────────
def run_shap_analysis(df_trades: pd.DataFrame):
    feature_cols = [
        "price_mpe_rank", "vol_mpe_rank", "rsi", "stoch_k",
        "ma_dist_pct", "onchain_h_rank", "btc_return_24h", "vol_price_ratio",
    ]
    feat_labels_kr = {
        "price_mpe_rank":  "가격MPE 백분위",
        "vol_mpe_rank":    "볼륨MPE 백분위",
        "rsi":             "RSI 값",
        "stoch_k":         "Stochastic %K",
        "ma_dist_pct":     "MA200 거리(%)",
        "onchain_h_rank":  "온체인 엔트로피 백분위",
        "btc_return_24h":  "BTC 24H 수익률",
        "vol_price_ratio": "볼륨/가격 MPE 비율",
    }

    X = df_trades[feature_cols].copy()
    y = df_trades["label"].values

    n_samples  = len(df_trades)
    n_positive = y.sum()
    win_rate   = n_positive / n_samples * 100

    print(f"\n  데이터: {n_samples}건 (승리={int(n_positive)}, 패배={n_samples - int(n_positive)}, 승률={win_rate:.1f}%)")
    print(f"  피처: {feature_cols}")

    if n_samples < 20:
        print("  ⚠ 샘플 수 부족 (<20) → SHAP 분석 신뢰도 낮음")

    # LightGBM 학습 (분석 전용, 배포 없음)
    model = lgb.LGBMClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        min_child_samples=3,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    train_pred = model.predict_proba(X)[:, 1]
    train_auc  = ((train_pred[y == 1].mean() > train_pred[y == 0].mean()))
    print(f"  학습 완료 (분석 전용 모델, AUC 방향성={train_auc})")

    # SHAP 계산
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values가 list인 경우 (이진분류 positive class)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    # 피처 중요도 (|SHAP| 평균)
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":      feature_cols,
        "feature_kr":   [feat_labels_kr[f] for f in feature_cols],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n  [SHAP 피처 중요도 순위]")
    print(f"  {'순위':<4} {'피처':<22} {'|SHAP| 평균':>12} {'상대 중요도':>12}")
    print(f"  {'-'*52}")
    max_shap = importance_df["mean_abs_shap"].max()
    for rank, row in importance_df.iterrows():
        rel = row["mean_abs_shap"] / max_shap * 100
        bar = "█" * int(rel / 5)
        print(f"  {rank+1:<4} {row['feature_kr']:<22} {row['mean_abs_shap']:>12.4f} {rel:>10.1f}%  {bar}")

    return sv, X, y, importance_df, feat_labels_kr


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_shap_results(shap_values, X, importance_df, feat_labels_kr, df_trades):
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    feature_cols = X.columns.tolist()
    X_renamed    = X.rename(columns=feat_labels_kr)

    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.40)

    def _style(ax):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.6)
        ax.xaxis.label.set_color("#8b949e")

    # 1. 피처 중요도 바차트
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1)
    imp = importance_df.sort_values("mean_abs_shap")
    colors = ["#f78166" if i >= len(imp) - 3 else "#56d364" if i >= len(imp) - 5 else "#8b949e"
              for i in range(len(imp))]
    bars = ax1.barh(imp["feature_kr"], imp["mean_abs_shap"], color=colors, alpha=0.85)
    for bar, v in zip(bars, imp["mean_abs_shap"]):
        ax1.text(v + 0.001, bar.get_y() + bar.get_height()/2, f"{v:.4f}",
                 va="center", fontsize=8.5, color="#e6edf3")
    ax1.set_title("SHAP 피처 중요도 (|SHAP| 평균)", color="#e6edf3")
    ax1.set_xlabel("평균 |SHAP| 값", color="#8b949e")

    # 2. SHAP 값 분포 (bee swarm style → dot plot)
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2)
    top5_feats = importance_df.head(5)["feature"].tolist()[::-1]
    top5_kr    = [feat_labels_kr[f] for f in top5_feats]
    for yi, (feat, feat_kr) in enumerate(zip(top5_feats, top5_kr)):
        feat_idx = list(feature_cols).index(feat)
        sv_col   = shap_values[:, feat_idx]
        x_vals   = sv_col
        y_jitter = np.random.normal(yi, 0.08, len(x_vals))
        scatter  = ax2.scatter(x_vals, y_jitter, c=X[feat].values,
                               cmap="RdYlGn_r", alpha=0.6, s=20)
    ax2.axvline(0, color="#8b949e", linewidth=0.8)
    ax2.set_yticks(range(len(top5_kr)))
    ax2.set_yticklabels(top5_kr, color="#8b949e", fontsize=9)
    ax2.set_title("Top5 피처 SHAP 분포 (색=값 크기, 빨강=높음)", color="#e6edf3")
    ax2.set_xlabel("SHAP 값 (양수=승리 기여, 음수=패배 기여)", color="#8b949e")

    # 3. 승/패 거래별 피처 분포 비교 (top4)
    top4_feats = importance_df.head(4)["feature"].tolist()
    top4_kr    = [feat_labels_kr[f] for f in top4_feats]
    for subplot_i, (feat, feat_kr) in enumerate(zip(top4_feats, top4_kr)):
        row = 1 + subplot_i // 2
        col = subplot_i % 2
        ax  = fig.add_subplot(gs[row, col])
        _style(ax)

        win_vals  = df_trades[df_trades["label"] == 1][feat].values
        lose_vals = df_trades[df_trades["label"] == 0][feat].values

        bins = np.linspace(
            min(df_trades[feat].min(), 0),
            max(df_trades[feat].max(), 1),
            20
        )
        ax.hist(win_vals,  bins=bins, alpha=0.6, color="#56d364", label=f"승리 (n={len(win_vals)})")
        ax.hist(lose_vals, bins=bins, alpha=0.6, color="#f78166", label=f"패배 (n={len(lose_vals)})")
        ax.axvline(win_vals.mean(),  color="#56d364", linewidth=1.5, linestyle="--",
                   label=f"승리 avg={win_vals.mean():.3f}")
        ax.axvline(lose_vals.mean(), color="#f78166", linewidth=1.5, linestyle="--",
                   label=f"패배 avg={lose_vals.mean():.3f}")
        ax.set_title(f"{feat_kr} — 승/패 분포", color="#e6edf3")
        ax.legend(fontsize=7.5, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
        ax.set_xlabel(feat, color="#8b949e")
        ax.set_ylabel("거래수", color="#8b949e")

    fig.suptitle(
        "Exp26B: SHAP 피처 중요도 분석\n"
        "Exp26A-B 조건(RSI<35, 69건) — 어떤 신호가 실제 수익을 만들었나?",
        color="#e6edf3", fontsize=12, fontweight="bold", y=1.01,
    )

    path = RESULTS_DIR / "exp26b_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  차트 저장: {path}")


# ── 인사이트 출력 ─────────────────────────────────────────────────────────────
def print_insights(importance_df, df_trades):
    print("\n" + "=" * 70)
    print("  [Exp26B 핵심 인사이트 — Exp27 설계 방향]")
    print("=" * 70)

    top1 = importance_df.iloc[0]
    top2 = importance_df.iloc[1]
    top3 = importance_df.iloc[2]

    print(f"\n  ▶ 1위 피처: {top1['feature_kr']} ({top1['feature']})")
    print(f"     → 이 신호가 진입 결과에 가장 큰 영향")

    print(f"\n  ▶ 2위 피처: {top2['feature_kr']} ({top2['feature']})")
    print(f"     → 두 번째로 중요한 진입 타이밍 요소")

    print(f"\n  ▶ 3위 피처: {top3['feature_kr']} ({top3['feature']})")

    # 승/패 거래 비교
    wins  = df_trades[df_trades["label"] == 1]
    loses = df_trades[df_trades["label"] == 0]

    print(f"\n  ▶ 승리 vs 패배 핵심 차이:")
    for _, row in importance_df.head(5).iterrows():
        feat = row["feature"]
        w_avg = wins[feat].mean()
        l_avg = loses[feat].mean()
        diff_direction = "↑" if w_avg > l_avg else "↓"
        print(f"     {row['feature_kr']:22s}: 승리={w_avg:.3f}  패배={l_avg:.3f}  {diff_direction}")

    # Exp27 방향 제안
    print(f"\n  ▶ Exp27 설계 제안:")
    top_feat = importance_df.iloc[0]["feature"]
    if top_feat == "price_mpe_rank":
        print("     price_mpe가 1위 → MPE 임계값을 더 정교하게 (1%ile 진입에 집중)")
    elif top_feat == "vol_mpe_rank":
        print("     vol_mpe가 1위 → 볼륨 MPE 조건이 실제 핵심 → 볼륨 임계값 상향 (>=60%ile?)")
    elif top_feat == "rsi":
        print("     RSI가 1위 → RSI 임계값이 핵심 → RSI<25로 더 강하게 선별?")
    elif top_feat == "stoch_k":
        print("     Stochastic이 1위 → Stochastic<15로 더 강하게 선별?")
    elif top_feat in ("ma_dist_pct", "btc_return_24h"):
        print("     추세/레짐 피처가 1위 → 강세 레짐에서만 진입하는 조건 추가 검토")

    # 하위 피처 (제거 후보)
    bottom2 = importance_df.tail(2)
    print(f"\n  ▶ 중요도 하위 피처 (규칙에서 제거 후보):")
    for _, row in bottom2.iterrows():
        print(f"     {row['feature_kr']:22s}: |SHAP|={row['mean_abs_shap']:.4f} (낮음)")

    print("\n" + "=" * 70)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp26B: SHAP 피처 중요도 분석 (ML 트레이딩 없음, 순수 분석)")
    print(f"  코인: {', '.join(LABEL[s] for s in COINS)}")
    print(f"  기준: Exp26A-B (RSI<35 + MPE<10% + vol>=50%, 예상 69건)")
    print(f"  목적: '어떤 신호가 실제 수익을 만들었나?' 데이터 기반 확인")
    print("=" * 70)

    print("\n[온체인 데이터 로드...]")
    funding = collect_funding_rate("BTCUSDT", START, END)
    fg      = collect_fear_greed(START, END)

    print("\n[코인 데이터 + MPE + 기술지표 계산 (캐시 사용)...]")
    coin_data = {}
    for sym in COINS:
        df = collect(sym, "1h", START, END)

        price_mpe = rolling_mpe(df["close"], window=MPE_WINDOW,
                                m=MPE_M, scales=MPE_SCALES,
                                cache_key=f"{sym}_1h_{START}_{END}")

        vol_series = df["volume"].replace(0, np.nan).ffill()
        vol_mpe    = rolling_mpe(vol_series, window=MPE_WINDOW,
                                 m=MPE_M, scales=MPE_SCALES,
                                 cache_key=f"{sym}_1h_volume_{START}_{END}")

        stoch_k = compute_stochastic(df, k_period=14)

        h = combined_onchain_entropy(
            funding_entropy(funding, df.index),
            fear_greed_entropy(fg, df.index),
        )

        coin_data[sym] = {
            "df": df, "price_mpe": price_mpe,
            "vol_mpe": vol_mpe, "stoch_k": stoch_k, "h": h,
        }
        print(f"  [{LABEL[sym]:6s}] {len(df):,}봉 로드 완료")

    # 거래 수집 (피처 포함)
    print("\n[WF 거래 재수집 + 피처 추출 중...]")
    df_trades = collect_trades_with_features(coin_data)

    if df_trades.empty:
        print("  ⚠ 거래 데이터 없음. 종료.")
        return

    print(f"  총 {len(df_trades)}건 거래 수집")
    print(f"  코인별: {df_trades.groupby('sym')['sym'].count().to_dict()}")
    print(f"  구간별: {df_trades.groupby('period')['period'].count().to_dict()}")
    print(f"  전체 승률: {df_trades['label'].mean()*100:.1f}%")

    # 피처 분포 요약
    feature_cols = ["price_mpe_rank", "vol_mpe_rank", "rsi", "stoch_k",
                    "ma_dist_pct", "onchain_h_rank", "btc_return_24h", "vol_price_ratio"]
    print("\n  [피처 기초 통계]")
    print(f"  {'피처':<22} {'평균':>8} {'중앙값':>8} {'std':>8}")
    print(f"  {'-'*48}")
    for f in feature_cols:
        col = df_trades[f]
        print(f"  {f:<22} {col.mean():>8.3f} {col.median():>8.3f} {col.std():>8.3f}")

    # SHAP 분석
    print("\n[SHAP 분석 실행 중...]")
    shap_values, X, y, importance_df, feat_labels_kr = run_shap_analysis(df_trades)

    # 결과 저장
    RESULTS_DIR.mkdir(exist_ok=True)
    importance_df.to_csv(RESULTS_DIR / "exp26b_feature_importance.csv",
                         index=False, encoding="utf-8-sig")
    print(f"\n  피처 중요도 CSV 저장: results/exp26b_feature_importance.csv")

    # 인사이트 출력
    print_insights(importance_df, df_trades)

    # 시각화
    print("\n[시각화 생성...]")
    plot_shap_results(shap_values, X, importance_df, feat_labels_kr, df_trades)

    print("\nExp26B 완료!")


if __name__ == "__main__":
    main()
