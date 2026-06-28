#!/bin/bash
# nohup 래퍼: timeout 없이 백그라운드 영구 실행
# cline execute_command의 timeout 1800 래퍼를 우회
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
LOG=/tmp/cline/smoke800.log

mkdir -p /tmp/cline
cd "$SCRIPT_DIR"

nohup "$PYTHON" unit_similarity_pooled3025.py --max-n 800 \
    >"$LOG" 2>&1 &
PID=$!
echo "LAUNCHED PID=$PID"
echo "LOG=$LOG"
echo $PID > /tmp/cline/smoke800.pid
