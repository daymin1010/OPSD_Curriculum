#!/bin/bash
#SBATCH --job-name=Z1_think_smoke
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

# ── env: keep shared caches clean ───────────────────────────────────────────
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/smoke_samples.parquet
OUTDIR=$ACT/outputs

cd "$BASE"

echo "[INFO] node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

WORLD=4

# ── launch WORLD shards, one per visible GPU ────────────────────────────────
# Stagger rank launches to avoid concurrent CUDA-context init storm
# (previously caused cudaErrorDevicesUnavailable on 2/4 ranks).
for r in $(seq 0 $((WORLD-1))); do
  echo "[INFO] launching rank $r on CUDA_VISIBLE_DEVICES=$r"
  CUDA_VISIBLE_DEVICES=$r $PY "$ACT/extract_thinking_smoke.py" \
      --rank "$r" --world-size "$WORLD" \
      --samples-parquet "$SAMPLES" \
      --output-dir "$OUTDIR" \
      --max-new-tokens 8192 &
  sleep 8
done
wait
echo "[INFO] all shards done; merging + report"

# ── merge + report (CPU) ────────────────────────────────────────────────────
$PY "$ACT/make_smoke_report.py"

echo "[INFO] FINISHED $(date)"
