# Entropy_Crypto 라이브 트레이딩 봇

전략: MPE<10% + RSI<30 + MA200 + 온체인<40% | 5코인 | Kelly 사이징 | Binance Spot

---

## VPS 배포 순서

### 1. 코드 업로드
```bash
# 로컬에서 VPS로 업로드 (또는 git clone)
scp -r /path/to/Entropy_Crypto user@vps-ip:~/
```

### 2. Python 패키지 설치
```bash
cd ~/Entropy_Crypto
pip install python-binance apscheduler pandas numpy scipy tqdm
```

### 3. 환경변수 설정 (~/.bashrc 추가)
```bash
export BINANCE_API_KEY=여기에_API_KEY
export BINANCE_API_SECRET=여기에_API_SECRET
export TOTAL_CAPITAL=100
# 처음엔 DRY_RUN=true로 확인, 확인 후 제거
export DRY_RUN=true
```
```bash
source ~/.bashrc
```

### 4. DRY RUN으로 신호 확인 (실제 주문 없음)
```bash
cd ~/Entropy_Crypto
python -m live.main
```
로그를 보고 5개 코인의 신호 상태 확인. 정상 작동 확인 후 다음 단계.

### 5. 실거래 활성화
```bash
# ~/.bashrc에서 DRY_RUN 라인 제거 또는
unset DRY_RUN
```

### 6. cron 등록 (매시간 정각 실행)
```bash
crontab -e
```
다음 라인 추가:
```
0 * * * * cd ~/Entropy_Crypto && python -m live.main >> ~/Entropy_Crypto/live/logs/cron.log 2>&1
```

---

## 파일 구조

```
live/
├── config.py          # 파라미터 (TOTAL_CAPITAL, WARMUP_BARS 등)
├── data_feed.py       # Binance API 데이터 수집
├── signal_engine.py   # MPE+RSI+MA200+온체인 신호 계산
├── portfolio.py       # 포지션 상태 (state.json)
├── executor.py        # 주문 실행 (매수/매도)
├── main.py            # 메인 사이클 (매시간 실행)
├── logger_setup.py    # 로그 설정
├── state.json         # 실행 중 생성 — 포지션 상태 저장
└── logs/
    └── 2026-05.log    # 월별 로그 파일
```

---

## 모니터링

```bash
# 최근 로그 확인
tail -f ~/Entropy_Crypto/live/logs/$(date +%Y-%m).log

# 현재 포지션 확인
cat ~/Entropy_Crypto/live/state.json
```

---

## 대기 자본 Earn 연동

봇이 실거래 활성화 상태일 때 자동으로:
- 포지션 없는 USDT → Binance Simple Earn Flexible 자동 예치
- 진입 신호 발생 → 필요 금액만 Earn에서 인출 → 매수
- 청산 후 → 회수 금액 즉시 재예치

**매 사이클 비교 로그 예시:**
```
──────────────────────────────────────────────────
수익 비교
  트레이딩 PnL  : +0.4500 USDT  (3건)
  대기자본 이자 : +0.1200 USDT  (89.00 USDT × APY 4.20%)
  합산 수익     : +0.5700 USDT
──────────────────────────────────────────────────
```

---

## 자본 확장 시 (100 → 10,000 USDT)

`config.py`의 TOTAL_CAPITAL을 변경하거나 환경변수 `TOTAL_CAPITAL=10000` 설정.  
$10,000에서는 Kelly 사이징이 제대로 작동:
- Kelly 50% × 2,000 USDT = 1,000 USDT/포지션
- Kelly 30% × 2,000 USDT = 600 USDT/포지션  
- Kelly 15% × 2,000 USDT = 300 USDT/포지션

---

## 주의사항

- API 키는 **절대 코드에 직접 입력 금지** — 환경변수만 사용
- Binance API 설정: **Spot 거래 권한만** 활성화, IP 화이트리스트 권장
- 봇이 실행 중인 동안 Binance에서 수동으로 해당 코인 거래 시 포지션 불일치 발생
- state.json 삭제 시 포지션 추적 초기화됨 — 수동 청산 후 삭제할 것
