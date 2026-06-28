#!/bin/bash
#SBATCH --job-name=Z2b_pilot2_chunk
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# PILOT-2 (replication, disjoint from original pilot) chunk runner. Usage:
#   sbatch --job-name=Z2b_c0 run_pilot2_chunk.sh 0
#   sbatch --job-name=Z2b_c1 run_pilot2_chunk.sh 1   ... (0..3)
# Each chunk = ~354 samples (chunk_id round-robin balanced). 2x L40s, WORLD=2.
# Resume-safe: shifts/{id}.pt already present are skipped, so a time-limit cut
# can simply be re-submitted to finish the remainder.
# Wall time 23h (prev 10h got cut). Output dir = outputs/pilot2 (NEVER touches pilot/).
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

CHUNK="${1:?usage: run_pilot2_chunk.sh <chunk_id 0..3>}"

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/pilot2/unit30_v2_samples.parquet
OUTDIR=$ACT/outputs/pilot2         # shared shifts dir across all chunks (NEW)

cd "$BASE"
echo "[INFO] node=$(hostname) chunk=$CHUNK date=$(date)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true

WORLD=2
for r in $(seq 0 $((WORLD-1))); do
  echo "[INFO] launching rank $r (chunk $CHUNK) on CUDA_VISIBLE_DEVICES=$r"
  CUDA_VISIBLE_DEVICES=$r $PY "$ACT/extract_thinking_pilot.py" \
      --rank "$r" --world-size "$WORLD" \
      --chunk-id "$CHUNK" \
      --samples-parquet "$SAMPLES" \
      --output-dir "$OUTDIR" \
      --spec-name thinking_8k_v2 \
      --max-new-tokens 8192 &
  sleep 8
done
wait
echo "[INFO] chunk $CHUNK done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
