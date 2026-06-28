#!/bin/bash
#SBATCH --job-name=Z2b_pilot2_all
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# PILOT-2 single-job full runner (NO chunk split). GPU 2개만 사용해 큐 경쟁 방지.
#   --chunk-id 생략 -> 전체 1417 rows 처리. rank0/rank1 이 df.iloc[rank::world] 로 분담.
# Resume-safe: shifts/{problem_id}.pt 가 이미 있으면 스킵. time-limit 잘려도 재제출하면 이어서 완료.
# Output dir = outputs/pilot2 (기존 4-chunk 산출물과 동일 위치 → 재활용). pilot/ 는 절대 안 건드림.
set -euo pipefail
trap 'echo "[exit] $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
export TOKENIZERS_PARALLELISM=false

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
BASE=/scratch/lami2026/personal/jimin_2782
ACT=$BASE/src/OPSD_Curriculum/reasoning_pivot/activation
SAMPLES=$ACT/outputs/pilot2/unit30_v2_samples.parquet
OUTDIR=$ACT/outputs/pilot2         # shared shifts dir (resume from 4-chunk run)

cd "$BASE"
echo "[INFO] node=$(hostname) date=$(date) (pilot2 full, chunk=ALL, GPU=2)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv || true
echo "[INFO] existing shifts (resume base): $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"

WORLD=2
for r in $(seq 0 $((WORLD-1))); do
  echo "[INFO] launching rank $r (ALL rows) on CUDA_VISIBLE_DEVICES=$r"
  CUDA_VISIBLE_DEVICES=$r $PY "$ACT/extract_thinking_pilot.py" \
      --rank "$r" --world-size "$WORLD" \
      --samples-parquet "$SAMPLES" \
      --output-dir "$OUTDIR" \
      --spec-name thinking_8k_v2 \
      --max-new-tokens 8192 &
  sleep 8
done
wait
echo "[INFO] pilot2 full done $(date)"
echo "[INFO] shifts count: $(ls "$OUTDIR/shifts" 2>/dev/null | wc -l)"
