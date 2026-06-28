#!/bin/bash
#SBATCH --job-name=subjctrl
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
# SUBJECT 단독 검정 (LEVEL 통제). 순수 CPU(numpy) → GPU 미요청(--gres 없음).
# --exclusive 금지. 본인 JOBID만 관리. resume 불가, 한 번에 완주.
# 사용: sbatch run_subjctrl_cpu.sh [smoke|full]   (기본 full)
set -euo pipefail
trap 'echo "[exit $? @ $(date)]"' EXIT

MODE="${1:-full}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface

PYTHON=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
ANALYSIS=/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/analysis
cd "$ANALYSIS"

# import/경로 가드 (빠른 smoke: pilot당 40개, perm 50)
echo "[guard $(date)] smoke import/path check"
"$PYTHON" subject_controlled_test.py --max-n 40 --n-perm 50

if [ "$MODE" = "smoke" ]; then
  echo "[done $(date)] smoke only (MODE=smoke)"
  exit 0
fi

echo "[start $(date)] FULL subject-controlled test (pooled N=3025, n-perm=2000)"
"$PYTHON" subject_controlled_test.py --n-perm 2000
echo "[done $(date)] outputs -> $ANALYSIS/REPORT_subject_controlled_*.md , subjctrl_artifacts.npz"
