#!/bin/bash
#SBATCH --job-name=Z4b_p2_chunk
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-03
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# pilot2 (replication, Qwen3-4B, H200 single-GPU) chunk runner. Usage:
#   sbatch --job-name=Z4b_p2_c0 run_pilot2_chunk_qwen3_4b_h200.sh 0   (... 0..3)
# Each chunk = ~354 samples. WORLD=1.
# 8B 산출물(outputs/pilot2/) 는 절대 건드리지 않음. 분리된 OUTDIR + spec_name.
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

CHUNK="${1:?usage: run_pilot2_chunk_qwen3_4b_h200.sh <chunk_id 0..3>}"

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/pilot2/unit30_v2_samples.parquet
OUTDIR=$ACT/outputs/pilot2_qwen3_4b   # separate from 8B outputs/pilot2

mkdir -p "$OUTDIR"
cd "$BASE"
echo "[INFO] node=$(hostname) chunk=$CHUNK date=$(date) model=Qwen/Qwen3-4B"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

WORLD=1
$PY "$ACT/extract_thinking_pilot.py" \
    --rank 0 --world-size "$WORLD" \
    --chunk-id "$CHUNK" \
    --samples-parquet "$SAMPLES" \
    --output-dir "$OUTDIR" \
    --model-id Qwen/Qwen3-4B \
    --spec-name thinking_8k_v2_4b \
    --max-new-tokens 8192

echo "[INFO] chunk $CHUNK done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
