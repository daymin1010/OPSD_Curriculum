#!/bin/bash
#SBATCH --job-name=Z2_pilot_smoke
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# NEW-CODE validation for extract_thinking_pilot.py on smoke2 (~12 samples):
#   - truncation t_k path (ok_truncated)
#   - is_correct scoring (math_verify)
#   - rich meta schema + resume/skip-existing
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/smoke2_samples.parquet
OUTDIR=$ACT/outputs/smoke2

cd "$BASE"
echo "[INFO] node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

WORLD=2
for r in $(seq 0 $((WORLD-1))); do
  echo "[INFO] launching rank $r on CUDA_VISIBLE_DEVICES=$r"
  CUDA_VISIBLE_DEVICES=$r $PY "$ACT/extract_thinking_pilot.py" \
      --rank "$r" --world-size "$WORLD" \
      --chunk-id -1 \
      --samples-parquet "$SAMPLES" \
      --output-dir "$OUTDIR" \
      --spec-name thinking_8k_v1 \
      --max-new-tokens 8192 &
  sleep 8
done
wait
echo "[INFO] smoke done $(date)"
ls -la "$OUTDIR/shifts" | head -n 20
