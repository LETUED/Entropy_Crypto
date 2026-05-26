#!/bin/bash
# VPS 배포 스크립트 (Ubuntu/Debian 기준)
# 사용법: bash deploy_vps.sh

set -e

PROJECT_DIR="/home/$USER/Entropy_Crypto"
PYTHON="python3"

echo "=== Entropy_Crypto 라이브 트레이딩 VPS 배포 ==="

# 1. 의존성 설치
echo "[1/4] 패키지 설치..."
pip install python-binance apscheduler yfinance pandas numpy tqdm --quiet

# 2. 환경변수 설정 확인
echo "[2/4] 환경변수 확인..."
if [ -z "$BINANCE_API_KEY" ]; then
    echo "⚠️  BINANCE_API_KEY 미설정!"
    echo "    ~/.bashrc에 다음 추가 후 source ~/.bashrc:"
    echo "    export BINANCE_API_KEY=your_key"
    echo "    export BINANCE_API_SECRET=your_secret"
    echo "    export TOTAL_CAPITAL=100"
fi

# 3. 드라이런 (API 연결 테스트)
echo "[3/4] API 연결 테스트..."
cd "$PROJECT_DIR"
$PYTHON -c "
from binance.client import Client
import os
c = Client(os.environ.get('BINANCE_API_KEY',''), os.environ.get('BINANCE_API_SECRET',''))
info = c.get_account()
print('Binance 연결 OK | 계정 타입:', info.get('accountType'))
print('USDT 잔액:', next((b['free'] for b in info['balances'] if b['asset']=='USDT'), '0'))
"

# 4. cron 등록 (매시간 정각 실행)
echo "[4/4] cron 등록..."
CRON_JOB="0 * * * * cd $PROJECT_DIR && $PYTHON -m live.main >> $PROJECT_DIR/live/logs/cron.log 2>&1"
(crontab -l 2>/dev/null | grep -v "live.main"; echo "$CRON_JOB") | crontab -

echo ""
echo "=== 배포 완료 ==="
echo "로그 위치: $PROJECT_DIR/live/logs/"
echo "상태 파일: $PROJECT_DIR/live/state.json"
echo "수동 실행: cd $PROJECT_DIR && python -m live.main"
echo ""
echo "cron 확인: crontab -l"
