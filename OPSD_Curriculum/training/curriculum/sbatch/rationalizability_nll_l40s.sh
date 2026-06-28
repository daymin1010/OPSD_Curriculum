#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name rationaliz_nll
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=01:30:00
#
# H1 (1단계): base Qwen3-8B forced-decode NLL of reference solution (generation 0)
# -> unit별 NLL vs g·level 상관. L40S x1.

set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
ANALYSIS=$REPO/src/OPSD_Curriculum/reasoning_pivot/activation/analysis

source $REPO/miniforge3/etc/profile.d/conda.sh
conda activate $REPO/envs/opsd
export HF_HOME=$REPO/cache/huggingface
export HF_HUB_OFFLINE=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NODE_CACHE=/dev/shm/jimin_2782_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}
export TEMP=$TMPDIR; export TMP=$TMPDIR
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$ANALYSIS"
python rationalizability_nll.py
echo "=== DONE $(date) ==="
