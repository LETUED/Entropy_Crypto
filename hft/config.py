"""
HFT Maker 설정 — 세그먼트 5개, 스톱 2%
"""

SYMBOL = "BTCUSDT"

# ── 자본 / 세그먼트 ─────────────────────────────────────────────────────────
TOTAL_CAPITAL_USDT  = 10_000.0
N_SEGMENTS          = 5
CAPITAL_PER_SEGMENT = TOTAL_CAPITAL_USDT / N_SEGMENTS   # 2,000 USDT

STOP_LOSS_PCT       = 0.02     # 포지션 대비 2% 손실 → 강제 청산
MAKER_SPREAD_PCT    = 0.00003  # 중간가 기준 ±0.003% (총 스프레드 0.006%) — 실제 BTC 스프레드 수준
QUOTE_CAPITAL_RATIO = 0.5      # 세그먼트 자본의 50%를 호가 하나에 사용

# ── 엔트로피 신호 ────────────────────────────────────────────────────────────
ENTROPY_WINDOW    = 100   # Maker용 고정 윈도우
ENTROPY_THRESHOLD = 0.85  # Shannon entropy < 이 값 → 신호 (70% 이상 단방향)

# ── Volume-Adaptive Window (Taker 전용) ──────────────────────────────────────
ADAPTIVE_WINDOW_SECONDS = 30   # 항상 최근 30초 분량의 거래로 entropy 계산
ADAPTIVE_WINDOW_MIN     = 30   # 거래량 극히 적을 때 최소 윈도우
ADAPTIVE_WINDOW_MAX     = 300  # 거래량 극히 많을 때 최대 윈도우
VOLUME_GATE_MIN_TPS     = 1.5  # trades/sec 미만이면 신규 진입 차단

# ── OBI 신호 ────────────────────────────────────────────────────────────────
OBI_BOOK_LEVELS  = 10   # 오더북 상위 N 레벨
OBI_THRESHOLD    = 0.15  # |OBI| > 이 값 → 방향성 있음
OBI_ACTIVATE_MAX = 0.5   # |OBI| > 이 값 → 활성화 차단 (Adverse Selection 위험)

# ── 수수료 ──────────────────────────────────────────────────────────────────
MAKER_FEE = 0.0002   # 0.02% (Binance USDT-M 선물 기본)
TAKER_FEE = 0.0005   # 0.05% (시장가 주문)

# ── Taker 방향성 전략 ────────────────────────────────────────────────────────
TAKER_TARGET_PCT    = 0.002   # 0.2% 목표 (수수료 0.10% + 순수익 0.10%)
TAKER_STOP_PCT      = 0.001   # 0.1% 스톱
TAKER_MAX_HOLD_S    = 60      # 최대 보유 60초
TAKER_POSITION_PCT  = 0.15    # 총 자본의 15%를 포지션 하나에
TAKER_MAX_POSITIONS = 3       # 최대 동시 포지션

# ── 거래소 연결 ──────────────────────────────────────────────────────────────
WS_BASE   = "wss://stream.binance.com:9443/stream"
REST_BASE = "https://api.binance.com"
