#!/bin/bash
#SBATCH --job-name=Z2b_smoke
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# PILOT-2 smoke: validate extract pipeline on smoke_v2 (~10 samples), 1 GPU.
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/pilot2/smoke_v2.parquet
OUTDIR=$ACT/outputs/pilot2/smoke

cd "$BASE"
echo "[INFO] node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

CUDA_VISIBLE_DEVICES=0 $PY "$ACT/extract_thinking_pilot.py" \
    --rank 0 --world-size 1 \
    --chunk-id -1 \
    --samples-parquet "$SAMPLES" \
    --output-dir "$OUTDIR" \
    --spec-name thinking_8k_v2 \
    --max-new-tokens 8192
echo "[INFO] smoke done $(date)"
ls -la "$OUTDIR/shifts" | head -n 20
