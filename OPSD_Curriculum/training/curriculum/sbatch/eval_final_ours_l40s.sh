#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name eval_fin_ours
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=04:00:00
#
# EVAL final checkpoints — cond3_ours (ours): mini50-step50, mini100-step100, q4-step225
# non-thinking, TP=2, val_n=3/3/3/1, temp=1.0. ~40min per ckpt × 3 = ~2h.

set -euo pipefail

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py

BASE_MODEL=Qwen/Qwen3-8B

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

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$OPSD_SRC/eval"

run_eval() {
  local ckpt="$1" label="$2" outdir="$3"
  if [ ! -d "$ckpt" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $label — checkpoint $ckpt not found"
    return 0
  fi
  mkdir -p "$outdir"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] EVAL $label  thinking=OFF  TP=2"
  echo "=== Adapter: $ckpt ==="
  echo "=================================================================="
  for ds_valn in "aime24 3" "aime25 3" "hmmt25 3" "math500 1"; do
    local ds=$(echo $ds_valn | cut -d' ' -f1)
    local valn=$(echo $ds_valn | cut -d' ' -f2)
    echo "--- $label dataset=$ds val_n=$valn ---"
    python "$EVAL" \
      --base_model "$BASE_MODEL" \
      --checkpoint_dir "$ckpt" \
      --dataset "$ds" \
      --val_n "$valn" \
      --temperature 1.0 \
      --tensor_parallel_size 2 \
      --gpu_memory_utilization 0.9 \
      --no_thinking \
      --output_file "$outdir/${ds}_${label}_nonthink_valn${valn}.json"
    echo "[$(date +%H:%M:%S)] DONE $label dataset=$ds"
  done
}

# mini50 final (step 50)
run_eval "$REPO/checkpoints/opsd_curriculum/mini50_8b/mini50_cond3_ours_h200/checkpoint-50" \
  "mini50_ours_step50" \
  "$REPO/outputs/eval_opsd_curriculum/mini50_cond3_ours_nonthink"

# mini100 final (step 100)
run_eval "$REPO/checkpoints/opsd_curriculum/mini100_8b/mini100_cond3_ours_h200/checkpoint-100" \
  "mini100_ours_step100" \
  "$REPO/outputs/eval_opsd_curriculum/mini100_cond3_ours_nonthink"

# q4 final (step 225)
run_eval "$REPO/checkpoints/opsd_curriculum/quarter_8b/quarter_cond3_ours_q4_h200/checkpoint-225" \
  "q4_ours_step225" \
  "$REPO/outputs/eval_opsd_curriculum/quarter_cond3_ours_nonthink"

echo "ALL FINAL EVAL DONE (ours)"