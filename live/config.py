"""
라이브 트레이딩 설정.

API 키는 환경변수에서 읽는다 — 절대 코드에 직접 입력하지 말 것.
VPS: ~/.bashrc 또는 /etc/environment에 설정
로컬: .env 파일 또는 OS 환경변수
"""

import os

# ── Binance API ────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# ── 포트폴리오 ──────────────────────────────────────────────────────────────────
TOTAL_CAPITAL  = float(os.environ.get("TOTAL_CAPITAL", "100"))   # 총 자본 USDT
COINS = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT"]
COIN_CAPITAL   = TOTAL_CAPITAL / len(COINS)                       # 코인당 배분 USDT
MIN_NOTIONAL   = 11.0   # Binance Spot 최소 주문 금액 (10 USDT + 버퍼)

# ── 전략 파라미터 (백테스트와 동일, 절대 변경 금지) ──────────────────────────────
MPE_WINDOW   = 168      # MPE 롤링 윈도우 (1H 봉)
MA_PERIOD    = 200      # 이동평균 기간
RSI_WINDOW   = 14
MAX_HOLD_H   = 168      # 최대 보유 시간 (시간)
ENTROPY_PCT  = 10       # MPE 하위 X% 진입
ONCHAIN_PCT  = 40       # 온체인 엔트로피 하위 X% 진입
FEE_RATE     = 0.001    # 0.1%

# Kelly 분수
KELLY_PCT1   = 0.50     # MPE 하위 1% → 50%
KELLY_PCT5   = 0.30     # MPE 하위 5% → 30%
KELLY_PCT10  = 0.15     # MPE 하위 10% → 15%

# ── 데이터 ─────────────────────────────────────────────────────────────────────
WARMUP_BARS  = 400      # 계산 워밍업: MA200 + MPE168 + 여유 = 최소 370봉
INTERVAL     = "1h"

# ── 모드 ────────────────────────────────────────────────────────────────────────
# DRY_RUN=true 이면 신호 계산만 하고 실제 주문은 하지 않음 (첫 실행 검증용)
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

# ── 파일 경로 ───────────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR    = Path(__file__).parent
STATE_FILE  = BASE_DIR / "state.json"       # 포지션 상태 영속
LOG_DIR     = BASE_DIR / "logs"
MPE_HIST    = BASE_DIR / "mpe_history"      # 라이브 MPE 히스토리 (백분위 기준)
