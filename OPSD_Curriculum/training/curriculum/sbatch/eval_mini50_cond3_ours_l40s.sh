#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name eval_m50_ours
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=10:00:00
#
# EVAL mini50 cond3_ours — 5 checkpoints (step 10,20,30,40,50).
# Qwen3-8B + LoRA(r=64) on L40S x2 (TP=2), non-thinking mode.
# val_n=3 for aime24/aime25/hmmt25, val_n=1 for math500, temp=1.0.

set -euo pipefail

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py

BASE_MODEL=Qwen/Qwen3-8B
CKPT_BASE=$REPO/checkpoints/opsd_curriculum/mini50_8b/mini50_cond3_ours_h200
OUTDIR=$REPO/outputs/eval_opsd_curriculum/mini50_cond3_ours_nonthink

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
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
export NCCL_P2P_DISABLE=1

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$OUTDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$OPSD_SRC/eval"

run_eval() {
  local step="$1" ds="$2" valn="$3"
  local ckpt="$CKPT_BASE/checkpoint-${step}"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] EVAL mini50_cond3_ours step=$step  dataset=$ds  val_n=$valn  thinking=OFF  TP=2"
  echo "=================================================================="
  if [ ! -d "$ckpt" ]; then
    echo "[SKIP] checkpoint $ckpt does not exist yet"
    return 0
  fi
  python "$EVAL" \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$ckpt" \
    --dataset "$ds" \
    --val_n "$valn" \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --no_thinking \
    --output_file "$OUTDIR/${ds}_mini50_cond3_ours_step${step}_nonthink_valn${valn}.json"
  echo "[$(date +%H:%M:%S)] DONE step=$step dataset=$ds"
}

for STEP in 10 20 30 40 50; do
  echo "########## CHECKPOINT-$STEP START ##########"
  run_eval "$STEP" aime24 3
  run_eval "$STEP" aime25 3
  run_eval "$STEP" hmmt25 3
  run_eval "$STEP" math500 1
  echo "########## CHECKPOINT-$STEP DONE ##########"
done

echo "ALL EVAL DONE"