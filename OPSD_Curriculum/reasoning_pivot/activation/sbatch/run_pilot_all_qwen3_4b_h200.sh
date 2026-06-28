#!/bin/bash
#SBATCH --job-name=Z4b_p1_all
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-03
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# pilot1 (Qwen3-4B, H200 single-GPU) ALL-in-one runner. WORLD=1, chunk-id=-1 → all rows.
# Resume-safe: shifts/{id}.pt already present are skipped → 중간에 cancel/L40S 이전 가능.
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/unit30_samples.parquet
OUTDIR=$ACT/outputs/pilot_qwen3_4b

mkdir -p "$OUTDIR"
cd "$BASE"
echo "[INFO] node=$(hostname) ALL date=$(date) model=Qwen/Qwen3-4B"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

$PY "$ACT/extract_thinking_pilot.py" \
    --rank 0 --world-size 1 \
    --chunk-id -1 \
    --samples-parquet "$SAMPLES" \
    --output-dir "$OUTDIR" \
    --model-id Qwen/Qwen3-4B \
    --spec-name thinking_8k_v1_4b \
    --max-new-tokens 8192

echo "[INFO] pilot1 all done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
