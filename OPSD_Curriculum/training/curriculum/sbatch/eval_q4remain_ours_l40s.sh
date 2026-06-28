#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name eval_q4rem_ours
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=02:30:00
#
# EVAL q4 ours step225 — REMAINING datasets only (hmmt25, math500); the prior
# eval_final job TIMEOUT'd (4h) before reaching these. TP=2, non-thinking.

set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-8B
CKPT=$REPO/checkpoints/opsd_curriculum/quarter_8b/quarter_cond3_ours_q4_h200/checkpoint-225
OUTDIR=$REPO/outputs/eval_opsd_curriculum/quarter_cond3_ours_nonthink
LABEL=q4_ours_step225

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
cd "$OPSD_SRC/eval"

if [ ! -d "$CKPT" ]; then echo "[SKIP] $CKPT not found"; exit 0; fi
for ds_valn in "hmmt25 3" "math500 1"; do
  ds=$(echo $ds_valn | cut -d' ' -f1); valn=$(echo $ds_valn | cut -d' ' -f2)
  echo "[$(date +%H:%M:%S)] EVAL $LABEL dataset=$ds val_n=$valn thinking=OFF TP=2"
  python "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
    --dataset "$ds" --val_n "$valn" --temperature 1.0 \
    --tensor_parallel_size 2 --gpu_memory_utilization 0.9 --no_thinking \
    --output_file "$OUTDIR/${ds}_${LABEL}_nonthink_valn${valn}.json"
  echo "[$(date +%H:%M:%S)] DONE $LABEL dataset=$ds"
done
echo "ALL Q4-REMAIN EVAL DONE (ours)"
