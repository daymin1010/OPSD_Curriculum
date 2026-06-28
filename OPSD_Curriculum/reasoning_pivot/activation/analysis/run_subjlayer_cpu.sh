#!/bin/bash
#SBATCH --job-name=subjlayer3025
#SBATCH --partition=l40sq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
# 순수 CPU(numpy/sklearn) 분석. GPU 미요청(--gres 없음) → 동료 GPU 작업 비방해.
# --exclusive 미사용. 본인 JOBID 만 squeue -j / scancel 로 관리할 것.
# 전체 N=3025 (max-n 미지정). resume 불가, 한 번에 완주.
set -euo pipefail
trap 'echo "[exit $? @ $(date)]"' EXIT

# BLAS 멀티스레딩 (8 cpus)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface

PYTHON=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
ANALYSIS=/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/analysis
cd "$ANALYSIS"

echo "[start $(date)] subject_layer_resolved.py  full N=3025 (no --max-n)"
"$PYTHON" subject_layer_resolved.py --n-perm 1000 --pca-comps 150 --probe-pca 100
echo "[done $(date)] outputs -> $ANALYSIS/REPORT_subject_controlled_<date>.md + ${ANALYSIS}/subjlayer_*.png"
