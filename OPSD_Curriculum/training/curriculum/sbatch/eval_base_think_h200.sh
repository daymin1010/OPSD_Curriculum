#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name qeval_base_think
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=24:00:00
#
# EVAL base Qwen3 (no adapter) — THINKING mode (원저 run_eval.sh 정합: --no_thinking 없음).
# 인자: 4b | 8b. AIME24/25 + HMMT25, val_n=12, temp=1.0, TP=2. RESUME(있는 json skip).

set -euo pipefail

SIZE="${1:?usage: eval_base_think_h200.sh 4b|8b}"
case "$SIZE" in
  4b) BASE_MODEL=Qwen/Qwen3-4B ;;
  8b) BASE_MODEL=Qwen/Qwen3-8B ;;
  *) echo "unknown size: $SIZE"; exit 1 ;;
esac

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
OUTDIR=$REPO/outputs/eval_opsd_curriculum/base_qwen3_${SIZE}_think

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

cd "$OPSD_SRC/eval"

for ds_valn in "aime24 12" "aime25 12" "hmmt25 12"; do
  set -- $ds_valn; ds=$1; valn=$2
  OUT="$OUTDIR/${ds}_base_qwen3_${SIZE}_think_valn${valn}.json"
  if [ -s "$OUT" ]; then echo "[SKIP] $OUT 존재"; continue; fi
  echo "[$(date +%H:%M:%S)] EVAL base_qwen3_${SIZE}  dataset=$ds  val_n=$valn  thinking=ON  TP=2"
  python "$EVAL" \
    --base_model "$BASE_MODEL" \
    --dataset "$ds" \
    --val_n "$valn" \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --output_file "$OUT"
  echo "[$(date +%H:%M:%S)] DONE dataset=$ds"
done
echo "ALL EVAL DONE base_${SIZE} think"
