#!/bin/bash
#SBATCH --job-name=Z4b_p1_smoke
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-03
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# Smoke: pilot1 chunk0, --limit 5, Qwen3-4B on H200 single GPU.
# Verifies model load, <think> token logic, dA shape, NaN/inf, wall-time/sample.
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/unit30_samples.parquet
OUTDIR=$ACT/outputs/pilot_qwen3_4b_smoke

mkdir -p "$OUTDIR"
cd "$BASE"
echo "[INFO] SMOKE node=$(hostname) date=$(date) model=Qwen/Qwen3-4B limit=5"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

$PY "$ACT/extract_thinking_pilot.py" \
    --rank 0 --world-size 1 \
    --chunk-id 0 \
    --samples-parquet "$SAMPLES" \
    --output-dir "$OUTDIR" \
    --model-id Qwen/Qwen3-4B \
    --spec-name thinking_8k_v1_4b_smoke \
    --limit 5 \
    --max-new-tokens 8192

echo "[INFO] SMOKE done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
