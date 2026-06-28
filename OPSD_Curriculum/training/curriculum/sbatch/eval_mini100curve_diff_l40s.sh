#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name eval_m100c_diff
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=08:00:00
#
# mini100 ours step-curve — checkpoints 40, 80 (기존 100과 합쳐 곡선). TP=2, non-thinking.
# AIME24/25/HMMT25 (val_n=3) + MATH500 (val_n=1).

set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-8B
CKPT_BASE=$REPO/checkpoints/opsd_curriculum/mini100_8b/mini100_cond2_diff_h200
OUTDIR=$REPO/outputs/eval_opsd_curriculum/mini100_cond2_diff_nonthink
ARM=mini100_diff

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
export TEMP=$TMPDIR; export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
export NCCL_P2P_DISABLE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$OUTDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$OPSD_SRC/eval"

for STEP in 40 80; do
  CKPT="$CKPT_BASE/checkpoint-${STEP}"
  if [ ! -d "$CKPT" ]; then echo "[SKIP] $CKPT not found"; continue; fi
  for ds_valn in "aime24 3" "aime25 3" "hmmt25 3" "math500 1"; do
    ds=$(echo $ds_valn | cut -d' ' -f1); valn=$(echo $ds_valn | cut -d' ' -f2)
    echo "[$(date +%H:%M:%S)] EVAL ${ARM} step=${STEP} dataset=${ds} val_n=${valn} TP=2 non-thinking"
    python "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
      --dataset "$ds" --val_n "$valn" --temperature 1.0 \
      --tensor_parallel_size 2 --gpu_memory_utilization 0.9 --no_thinking \
      --output_file "$OUTDIR/${ds}_${ARM}_step${STEP}_nonthink_valn${valn}.json"
    echo "[$(date +%H:%M:%S)] DONE ${ARM} step=${STEP} dataset=${ds}"
  done
done
echo "ALL mini100-curve diff DONE"
