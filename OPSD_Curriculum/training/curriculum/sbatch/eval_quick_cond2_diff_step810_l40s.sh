#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name qeval_diff810
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=04:00:00
#
# QUICK EVAL cond2_diff checkpoint-810 — Qwen3-8B + LoRA(r=64) on L40S x2 (TP=2).
# non-thinking mode. aime24 SKIPPED (already done with val_n=12).
# val_n=3 for aime25/hmmt25, val_n=1 for math500, temp=1.0.

set -euo pipefail

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py

BASE_MODEL=Qwen/Qwen3-8B
CKPT=$REPO/eval_ckpts/cond2_diff_step810
OUTDIR=$REPO/outputs/eval_opsd_curriculum/cond2_diff_step810_nonthink

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
echo "=== Adapter dir contents ==="; ls -1 "$CKPT" | head

cd "$OPSD_SRC/eval"

run_eval() {
  local ds="$1" valn="$2"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] EVAL cond2_diff_step810  dataset=$ds  val_n=$valn  temp=1.0  thinking=OFF  TP=2"
  echo "=================================================================="
  python "$EVAL" \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CKPT" \
    --dataset "$ds" \
    --val_n "$valn" \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --no_thinking \
    --output_file "$OUTDIR/${ds}_cond2_diff_step810_nonthink_valn${valn}.json"
  echo "[$(date +%H:%M:%S)] DONE dataset=$ds"
}

# aime24 skipped — already completed with val_n=12
run_eval aime25 3
run_eval hmmt25 3
run_eval math500 1

echo "ALL EVAL DONE"