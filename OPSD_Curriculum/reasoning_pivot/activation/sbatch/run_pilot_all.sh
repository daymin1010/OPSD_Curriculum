#!/bin/bash
#SBATCH --job-name=Z2_pilot_all
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# Single-job full-pilot runner (no chunk split). 2x L40s, WORLD=2.
#   --chunk-id omitted  -> default -1  -> ALL rows (1,608) processed.
#   rank0/rank1 each take ~804 samples via df.iloc[rank::world].
# Resume-safe: shifts/{problem_id}.pt already present are skipped, so a kill /
# time-limit cut can simply be re-submitted to finish the remainder.
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
OUTDIR=$ACT/outputs/pilot         # shared shifts dir

cd "$BASE"
echo "[INFO] node=$(hostname) date=$(date) (full pilot, chunk=ALL)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

WORLD=2
for r in $(seq 0 $((WORLD-1))); do
  echo "[INFO] launching rank $r (ALL rows) on CUDA_VISIBLE_DEVICES=$r"
  CUDA_VISIBLE_DEVICES=$r $PY "$ACT/extract_thinking_pilot.py" \
      --rank "$r" --world-size "$WORLD" \
      --samples-parquet "$SAMPLES" \
      --output-dir "$OUTDIR" \
      --spec-name thinking_8k_v1 \
      --max-new-tokens 8192 &
  sleep 8
done
wait
echo "[INFO] full pilot done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
