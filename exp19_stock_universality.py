"""
실험 19: 주식 시장 범용성 검증

질문: MPE+RSI+MA200 신호가 암호화폐에만 작동하는가, 아니면 주식에서도 엣지가 있는가?

방법:
  - 데이터: Yahoo Finance 일봉 (2021-2025)
  - 대상: SPY(S&P500), NVDA, AAPL, QQQ(나스닥100)
  - 파라미터: 일봉 적합 버전
      * MPE window = 30봉 (30거래일 ≈ 6주, 암호화폐 168H와 다른 스케일이지만 자연스러운 중기 패턴)
      * MA200 = 200일 (표준 장기 추세선, 암호화폐와 동일 의미)
      * RSI 14일 (표준)
      * 최대 보유 = 10봉 (10거래일 = 2주)
      * Kelly 사이징 동일 (MPE 하위 1%→50%, 5%→30%, 10%→15%)
  - 온체인 없음 (주식에 없음 → 공정한 "코어 신호" 비교)

주의: MPE 파라미터(window=30)가 기본값(168)과 다르므로
  cache_key에 _w30_ 태그 필수! 기존 암호화폐 캐시와 충돌 방지.

비교 기준:
  - 암호화폐 5코인 코어 신호 (온체인 없이) 결과
  - 주식 4종목 결과

실행: py exp19_stock_universality.py
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
import yfinance as yf
from tqdm import tqdm

from src.data.binance_collector import collect as binance_collect
from src.entropy.calculators import rolling_mpe
from src.analysis.h4_backtest import FEE_RATE, ENTROPY_PCT

RESULTS_DIR = Path("results")
CACHE_DIR   = Path("data/cache")
START, END  = "2021-01-01", "2025-01-01"

# ── 주식 설정 ──────────────────────────────────────────────────────────────────
STOCK_TICKERS = ["SPY", "NVDA", "AAPL", "QQQ"]
STOCK_CFG = {
    "mpe_window": 30,     # 30거래일 ≈ 6주 (캐시 키에 w30 명시 필수)
    "ma_period":  200,    # 200일 MA (표준)
    "rsi_window": 14,
    "max_hold":   10,     # 10거래일 ≈ 2주
    "fee":        0.001,  # 주식 커미션 (ETF 기준, 슬리피지 포함)
    "annualize":  np.sqrt(252),
}

# ── 암호화폐 비교군 설정 (온체인 없이) ────────────────────────────────────────
CRYPTO_COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
CRYPTO_LABEL = {"BTCUSDT":"BTC","SOLUSDT":"SOL","AVAXUSDT":"AVAX","ADAUSDT":"ADA","DOTUSDT":"DOT"}
CRYPTO_CFG = {
    "mpe_window": 168,
    "ma_period":  200,
    "rsi_window": 14,
    "max_hold":   168,
    "fee":        FEE_RATE,
    "annualize":  np.sqrt(24 * 365),
}


def _setup_font():
    try:
        candidates = [f.fname for f in fm.fontManager.ttflist
                      if any(k in f.name for k in ["Malgun","NanumGothic","AppleGothic"])]
        if candidates:
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


# ── Yahoo Finance 데이터 수집 (캐시 포함) ──────────────────────────────────────
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """일봉 OHLCV 수집 및 로컬 캐시"""
    cache_path = CACHE_DIR / f"stock_{ticker}_1d_{start}_{end}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"  캐시 로드: {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"  다운로드: {ticker} 일봉 {start}~{end}")
    raw = yf.download(ticker, start=start, end=end, interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"{ticker} 데이터 없음")

    # MultiIndex 컬럼 처리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df


# ── 단일 종목 전략 실행 (온체인 없음) ─────────────────────────────────────────
def run_strategy_no_onchain(df: pd.DataFrame, mpe: pd.Series, cfg: dict) -> tuple:
    """
    MPE + RSI + MA200 전략 (온체인 필터 없음).
    주식/암호화폐 공용. cfg로 파라미터 주입.
    """
    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(cfg["rsi_window"]).mean()
    loss  = (-delta.clip(upper=0)).rolling(cfg["rsi_window"]).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-12))

    ma = df["close"].rolling(cfg["ma_period"]).mean()

    mpe_thresh = np.percentile(mpe.dropna(), ENTROPY_PCT)
    mpe_clean  = mpe.dropna()
    k_pct1  = np.percentile(mpe_clean, 1)
    k_pct5  = np.percentile(mpe_clean, 5)
    k_pct10 = np.percentile(mpe_clean, 10)

    def _kelly(v):
        if v <= k_pct1:  return 0.5
        if v <= k_pct5:  return 0.3
        if v <= k_pct10: return 0.15
        return 0.0

    equity      = 1.0
    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    entry_kfrac = 0.0
    trades      = []
    curve       = []
    fee         = cfg["fee"]

    for i in range(len(df)):
        price   = df["close"].iloc[i]
        rsi_val = rsi.iloc[i]
        ma_val  = ma.iloc[i]
        mpe_val = mpe.iloc[i] if i < len(mpe) else np.nan

        if position == 1:
            held = i - entry_bar
            if rsi_val > 50 or held >= cfg["max_hold"]:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl * entry_kfrac) * (1 - fee)
                trades.append({"pnl": pnl, "kfrac": entry_kfrac, "held": held})
                position = 0

        if position == 0 and not np.isnan(mpe_val) and not np.isnan(ma_val):
            k_frac = _kelly(mpe_val)
            if (rsi_val < 30
                    and mpe_val <= mpe_thresh
                    and price > ma_val
                    and k_frac > 0):
                position    = 1
                entry_price = price * (1 + fee)
                entry_bar   = i
                entry_kfrac = k_frac

        curve.append(equity)

    if position == 1:
        pnl = (df["close"].iloc[-1] - entry_price) / entry_price
        equity *= (1 + pnl * entry_kfrac) * (1 - fee)

    eq = pd.Series(curve, index=df.index)
    return eq, trades


# ── 포트폴리오 Sharpe ─────────────────────────────────────────────────────────
def portfolio_sharpe(eq_list: list, annualize: float) -> tuple:
    port = sum(e / len(eq_list) for e in eq_list)
    ret  = port.pct_change().dropna()
    s    = float(ret.mean() / (ret.std() + 1e-12) * annualize)
    return s, port


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Exp19: 주식 시장 범용성 검증 (MPE+RSI+MA200, 온체인 없음)")
    print("=" * 65)

    all_results = {}

    # ─ 1. 주식 4종목 ────────────────────────────────────────────────────────
    print("\n[주식 데이터 로드 및 전략 실행 (일봉, MPE window=30)...]")
    stock_eqs    = []
    stock_trades = []

    for ticker in STOCK_TICKERS:
        df  = load_stock_data(ticker, START, END)
        # 주의: window=30으로 cache_key에 _w30_ 포함 → 기존 캐시 오염 방지
        mpe = rolling_mpe(df["close"], window=STOCK_CFG["mpe_window"],
                          cache_key=f"{ticker}_1d_w30_{START}_{END}")

        eq, trades = run_strategy_no_onchain(df, mpe, STOCK_CFG)
        stock_eqs.append(eq)
        stock_trades.extend(trades)
        all_results[ticker] = {"eq": eq, "trades": trades}

        ret = eq.pct_change().dropna()
        s   = float(ret.mean() / (ret.std() + 1e-12) * STOCK_CFG["annualize"])
        avg_pnl = np.mean([t["pnl"]*100 for t in trades]) if trades else 0.0
        print(f"  [{ticker:6s}] Sharpe={s:+.3f}, 거래={len(trades)}, 평균PnL={avg_pnl:+.3f}%")

    s_stock, port_stock = portfolio_sharpe(stock_eqs, STOCK_CFG["annualize"])
    print(f"\n  [주식 포트폴리오] Sharpe={s_stock:+.4f}, 총거래={len(stock_trades)}건")

    # ─ 2. 암호화폐 비교군 (온체인 없이) ──────────────────────────────────────
    print("\n[암호화폐 비교군 (온체인 없음, MPE window=168)...]")
    crypto_eqs    = []
    crypto_trades = []

    for sym in CRYPTO_COINS:
        df  = binance_collect(sym, "1h", START, END)
        # 기존 캐시 사용 (window=168 고정, 파라미터 변경 없음)
        mpe = rolling_mpe(df["close"], window=CRYPTO_CFG["mpe_window"],
                          cache_key=f"{sym}_1h_{START}_{END}")

        eq, trades = run_strategy_no_onchain(df, mpe, CRYPTO_CFG)
        crypto_eqs.append(eq)
        crypto_trades.extend(trades)

        ret = eq.pct_change().dropna()
        s   = float(ret.mean() / (ret.std() + 1e-12) * CRYPTO_CFG["annualize"])
        avg_pnl = np.mean([t["pnl"]*100 for t in trades]) if trades else 0.0
        print(f"  [{CRYPTO_LABEL[sym]:6s}] Sharpe={s:+.3f}, 거래={len(trades)}, 평균PnL={avg_pnl:+.3f}%")

    s_crypto, port_crypto = portfolio_sharpe(crypto_eqs, CRYPTO_CFG["annualize"])
    print(f"\n  [암호화폐 포트폴리오(온체인 없음)] Sharpe={s_crypto:+.4f}, 총거래={len(crypto_trades)}건")

    # ─ 3. 비교 테이블 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("비교 요약")
    print("=" * 65)
    print(f"\n[주식 — 일봉, MPE w30, MA200, max_hold=10d]")
    print(f"{'종목':>8} {'Sharpe':>10} {'거래수':>8} {'평균PnL':>10}")
    print("-" * 42)
    for ticker in STOCK_TICKERS:
        d = all_results[ticker]
        trades = d["trades"]
        ret = d["eq"].pct_change().dropna()
        s   = float(ret.mean() / (ret.std() + 1e-12) * STOCK_CFG["annualize"])
        avg = np.mean([t["pnl"]*100 for t in trades]) if trades else 0.0
        print(f"  {ticker:>8} {s:>+10.3f} {len(trades):>8} {avg:>+10.3f}%")
    print(f"  {'포트':>8} {s_stock:>+10.4f} {len(stock_trades):>8}")

    print(f"\n[암호화폐 — 1H, MPE w168, MA200, 온체인 없음]")
    print(f"{'코인':>8} {'Sharpe':>10} {'거래수':>8}")
    print("-" * 35)
    for sym in CRYPTO_COINS:
        r, t = crypto_eqs[CRYPTO_COINS.index(sym)], all_results.get(sym)
    print(f"  {'포트':>8} {s_crypto:>+10.4f} {len(crypto_trades):>8}")

    # 범용성 평가
    positive_stocks = sum(1 for t in STOCK_TICKERS
                          if all_results[t]["trades"] and
                          np.mean([x["pnl"] for x in all_results[t]["trades"]]) > 0)

    print(f"\n  주식 포트폴리오 Sharpe: {s_stock:+.4f}")
    print(f"  암호화폐 포트폴리오 Sharpe (온체인 없음): {s_crypto:+.4f}")
    print(f"  양수 종목: {positive_stocks}/{len(STOCK_TICKERS)}")
    print()

    if s_stock > 0:
        print("  [결론] 주식에서도 양수 — MPE+RSI 코어 신호의 범용성 확인")
        if s_stock > 0.2:
            print("         Sharpe > 0.2 — 통계적으로 의미 있는 수준")
    else:
        print("  [결론] 주식에서 음수 — 신호가 암호화폐 특화일 가능성")

    # ─ 4. 시각화 ──────────────────────────────────────────────────────────
    _setup_font()
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # 주식
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22"); ax1.spines[:].set_color("#30363d")
    ax1.tick_params(colors="#8b949e")
    colors_s = ["#f7931a", "#00c896", "#e040fb", "#40a9ff"]
    for j, ticker in enumerate(STOCK_TICKERS):
        eq = all_results[ticker]["eq"]
        ax1.plot(eq.index, eq / eq.iloc[0], linewidth=0.9, color=colors_s[j],
                 alpha=0.75, label=ticker)
    norm_ps = port_stock / port_stock.iloc[0]
    ax1.plot(port_stock.index, norm_ps, linewidth=2.0, color="white",
             label=f"포트폴리오 ({s_stock:+.3f})")
    ax1.axhline(1.0, color="#30363d", linewidth=0.8)
    ax1.set_xlabel("날짜", color="#8b949e")
    ax1.set_ylabel("자산 배수", color="#8b949e")
    ax1.set_title(
        f"주식 (일봉, MPE w30)  |  Sharpe {s_stock:+.4f}  |  {len(stock_trades)}건",
        color="#e6edf3", fontsize=11
    )
    ax1.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax1.yaxis.grid(True, color="#21262d", linestyle="--")

    # 암호화폐 (온체인 없음)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22"); ax2.spines[:].set_color("#30363d")
    ax2.tick_params(colors="#8b949e")
    colors_c = ["#f7931a", "#00c896", "#e040fb", "#40a9ff", "#ff6b6b"]
    for j, (sym, eq) in enumerate(zip(CRYPTO_COINS, crypto_eqs)):
        ax2.plot(eq.index, eq / eq.iloc[0], linewidth=0.9, color=colors_c[j],
                 alpha=0.75, label=CRYPTO_LABEL[sym])
    norm_pc = port_crypto / port_crypto.iloc[0]
    ax2.plot(port_crypto.index, norm_pc, linewidth=2.0, color="white",
             label=f"포트폴리오 ({s_crypto:+.3f})")
    ax2.axhline(1.0, color="#30363d", linewidth=0.8)
    ax2.set_xlabel("날짜", color="#8b949e")
    ax2.set_ylabel("자산 배수", color="#8b949e")
    ax2.set_title(
        f"암호화폐 (1H, 온체인 없음)  |  Sharpe {s_crypto:+.4f}  |  {len(crypto_trades)}건",
        color="#e6edf3", fontsize=11
    )
    ax2.legend(fontsize=8.5, facecolor="#21262d", labelcolor="#e6edf3")
    ax2.yaxis.grid(True, color="#21262d", linestyle="--")

    universal = s_stock > 0 and s_crypto > 0
    fig.suptitle(
        f"Exp19: 주식 범용성  |  주식 {s_stock:+.3f}  vs  암호화폐(온체인X) {s_crypto:+.3f}"
        f"  |  {'범용 확인' if universal else '암호화폐 특화'}",
        color="#e6edf3", fontsize=12, fontweight="bold"
    )

    path = RESULTS_DIR / "exp19_stock_universality.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장: {path}")
    plt.show()

    # ─ 5. 최종 요약 ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("최종 범용성 평가")
    print("=" * 65)
    print(f"  주식 포트폴리오:         {s_stock:+.4f}  ({len(stock_trades)}건)")
    print(f"  암호화폐 (온체인 없음):  {s_crypto:+.4f}  ({len(crypto_trades)}건)")
    print()
    print(f"  온체인 기여도: {s_crypto - s_stock:+.4f} (암호화폐 전체 {0.804:.3f} - 온체인없음 {s_crypto:.3f} = {0.804-s_crypto:+.3f})")
    print()
    if s_stock > 0 and s_crypto > 0:
        print("  [최종] MPE+RSI 코어 신호 → 주식/암호화폐 모두 양수")
        print("         온체인은 암호화폐 특화 부스터. 코어 신호는 자산 불문")
    elif s_stock <= 0:
        print("  [최종] 주식에서 미작동 → 신호가 암호화폐 24/7 시장 특화")
        print("         주식은 개장시간 제한 + 낮은 변동성 → 다른 최적 스케일 가능성")


if __name__ == "__main__":
    main()
