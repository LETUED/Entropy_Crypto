# Entropy_Crypto — 온보딩 가이드

## 프로젝트 철학

> "잃지 않는 게 돈을 버는 거야"

라마누잔의 수학적 직관에서 영감받은 **엔트로피 기반 암호화폐 트레이딩 시스템**.  
목표: **과거에도, 미래에도 일정하게 통하는 타임리스 공식** — 과적합 없이.

핵심 아이디어: 엔트로피는 물리 법칙이기 때문에 특정 코인/기간에 핏되지 않는다.

---

## 전략 개요

**주신호**: Multiscale Permutation Entropy (MPE)  
**방향신호**: RSI < 30 (과매도) → 롱  
**청산**: RSI > 50 OR 최대 168H 보유

### 진입 조건 (전체)
1. RSI < 30 (과매도)
2. MPE < 하위 10% (저엔트로피 = 시장이 질서있는 상태)
3. 가격 > MA200 (상승 추세)
4. 온체인 엔트로피 < 40th percentile (자금조달비율 + 공포탐욕지수)

---

## 핵심 파라미터 (논문 기반 고정값 — 튜닝 금지)

```
m=3, tau=1, scales=[1,2,4,8]   # MPE 파라미터 (preprints.org/202511.1980)
window=168H                     # 롤링 윈도우 (7일, 1H 기준)
ENTROPY_PCT=10                  # MPE 하위 10%
MAX_HOLD_H=168                  # 최대 보유 168시간
MA_PERIOD=200                   # MA200
FEE_RATE=0.001                  # 0.1% 수수료
```

Kelly 포지션 사이징:
- MPE 하위 1% → 50%
- MPE 하위 5% → 30%
- MPE 하위 10% → 15%

---

## 파일 구조

```
Entropy_Crypto/
├── src/
│   ├── data/
│   │   ├── binance_collector.py      # OHLCV 수집 (캐시: data/cache/*.parquet)
│   │   └── onchain_collector.py      # 자금조달비율, 공포탐욕 수집
│   ├── entropy/
│   │   ├── calculators.py            # MPE, Delta MPE, BBW, ATR 계산
│   │   └── onchain_entropy.py        # 온체인 엔트로피 (H_funding, H_fg, H_combined)
│   └── analysis/
│       ├── h3_validation.py          # RSI 계산, 신호 생성
│       ├── h4_backtest.py            # 핵심 백테스트 엔진 (run_strategy, compute_metrics)
│       ├── walkforward.py            # Walk-forward 검증
│       ├── signal_quality.py         # 신호 품질 분석 (forward return edge)
│       ├── filter_diagnosis.py       # 필터 통과율 진단
│       ├── delta_mpe_backtest.py     # Delta MPE 전략 (레짐 전환 감지)
│       ├── multi_coin.py             # 멀티코인 포트폴리오 백테스트
│       └── visualizer.py            # 공통 시각화
├── run_walkforward.py                # Walk-forward + 신호품질 + 필터진단 실행
├── run_delta_mpe.py                  # Delta MPE 전략 실행
└── results/                          # 차트 저장 디렉토리
```

---

## 실행 방법

```bash
# Walk-forward 검증 (핵심)
py run_walkforward.py

# Delta MPE 전략 실험
py run_delta_mpe.py
```

데이터는 `data/cache/`에 자동 캐싱 — 재실행 시 빠름.  
MPE 계산은 수분 소요 (35,000개 봉 × 롤링 계산).

---

## 검증 아키텍처

### Walk-forward (run_walkforward.py)
- 학습 12개월 → 임계값 계산
- 테스트 6개월 → out-of-sample 실행
- 6개월씩 이동, 2022~2024 총 6개 윈도우
- 판단: 모든 구간 Sharpe > 0 → 타임리스

### 신호 품질 분석
- 진입 후 24H/48H/72H/168H 수익률 측정
- 랜덤 기준선 대비 edge 계산
- **핵심 발견**: 168H edge +22.9%p — 신호는 맞지만 조기진입

### 필터 진단
- 각 조건 독립 통과율 측정
- 병목 파악: RSI<30 + MA200 동시 충족이 구조적으로 드묾

---

## 실험 결과 요약 (2021~2025, BTCUSDT 1H)

### 신호 타이밍 문제
| 시점 | Edge (vs 랜덤) |
|------|---------------|
| 24H | -28.3%p (너무 이른 진입) |
| 168H | **+22.9%p** (신호 자체는 유효) |

→ **해결**: 청산 로직을 24H 고정 → RSI>50 OR 168H max로 변경

### Delta MPE 실험 결과
| 전략 | 신호수 | Sharpe |
|------|--------|--------|
| Delta+RSI (MA없음) | 654 | -0.561 |
| Delta+RSI+MA200 | 90 | -0.590 |
| Delta+RSI+MA+온체인 | ~30 | +0.479 |
| **기존 MPE level 전략** | **~16** | **+0.436** |

→ **결론**: Delta MPE는 신호 수를 늘리지만 노이즈도 같이 증가.  
기존 MPE level 필터가 더 선별적으로 유효한 진입 포착.

### 근본 문제
RSI<30 (가격 급락) + MA200 (상승추세) 동시 충족 = 구조적으로 희귀  
→ 4년간 진입 ~16번 (너무 적음)

---

## 현재 논의 중인 다음 방향

**옵션 A**: MA200 필터 제거 — RSI<30 + MPE<10%만, 하락장도 커버 (short 포함)  
**옵션 B**: 멀티코인 — BTC/ETH/SOL 포트폴리오, 총 진입 기회 증가

---

## 참고 논문

| 논문 | 핵심 기여 |
|------|----------|
| Multiscale Permutation Entropy (preprints.org/202511.1980) | m=3, tau=1, scales=[1,2,4,8] 파라미터 근거 |
| Thermodynamic Analysis (2023) | Delta Entropy가 분산 41~57% 설명 |
| Hidden Order in Trades (2025) | 엔트로피-수익률 관계 실증 |
| MPE vs GARCH (2025) | 암호화폐 변동성 예측에서 MPE 우위 |
| Permutation Transition Entropy (2020) | 레짐 전환 감지 |

---

## 주의사항

- **파라미터 튜닝 금지**: m, tau, scales는 논문 기반 고정값. 변경 시 과적합.
- **엔트로피 신호 = 타이밍 필터** (방향 예측 아님) — RSI가 방향 결정
- 리스크 관리(SL/TP)는 별도 모듈로 분리 예정 — 신호 품질과 혼동 금지
- BTC 단일 코인 결과이므로 일반화 전 멀티코인 검증 필요
