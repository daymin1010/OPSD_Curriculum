#!/bin/bash
#SBATCH --job-name=unitsim_full3025
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
# 순수 CPU(numpy SVD) 분석. GPU 미요청(--gres 없음) → 동료 GPU 작업 비방해.
# 전체 N=3025 (max-n 미지정 → union/중복 게이트 작동). resume 불가, 한 번에 완주.
set -euo pipefail
trap 'echo "[exit $? @ $(date)]"' EXIT

# BLAS 멀티스레딩으로 SVD 가속 (8 cpus)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface

PYTHON=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
ANALYSIS=/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/analysis
cd "$ANALYSIS"

echo "[start $(date)] full N=3025 unit similarity (no --max-n)"
"$PYTHON" unit_similarity_pooled3025.py
echo "[done $(date)] outputs -> $ANALYSIS/outputs/unit_similarity_pooled3025/"
