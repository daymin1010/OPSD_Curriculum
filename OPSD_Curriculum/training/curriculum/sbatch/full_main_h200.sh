#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --job-name opsd_cur_full_main_h200
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=18:00:00
# ============================================================
# FULL ③ main (diff x cluster, 16-stage) — Qwen3-8B on H200 (141GB) ×2 (ws=2).
# B_glob=32 via pd2*ga8*ws2. gpu_mem_util=0.6. T=480 (arm-common).
# attach_gold=true (monitor-only rollout_acc). stage_distinct>1 abort guard
# self-checks the 16-stage schedule integrity. Checkpoints every 30 steps
# (save_total_limit=3) to /scratch; resume auto-detected (18h backstop).
# Submitted via afterok on the diff-only run.
# ============================================================
set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
STAGES=$REPO/src/OPSD_Curriculum/training/stages
ROW=$REPO/src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet

echo "=== job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
source $REPO/miniforge3/etc/profile.d/conda.sh
conda activate $REPO/envs/opsd

export HF_HOME=$REPO/cache/huggingface
export WANDB_PROJECT=OPSD_Curriculum
export WANDB_DIR=$REPO/wandb
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export NODE_CACHE=/dev/shm/jimin_2782_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs" "$WANDB_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

GPU_LOG="$REPO/runs/gpu_mem_${SLURM_JOB_NAME}.${SLURM_JOB_ID}.log"
( while true; do date +%s | tr -d '\n'; echo -n " "; \
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr '\n' ';'; \
  echo ""; sleep 15; done ) > "$GPU_LOG" 2>/dev/null &
GPU_LOGGER_PID=$!
trap 'kill $GPU_LOGGER_PID 2>/dev/null || true; rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

if [ ! -f "$ROW" ]; then echo "[phase0] building join table"; python curriculum_schedule.py phase0; fi

accelerate launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12958 \
    train_opsd_curriculum.py \
    --config configs/full_8b_h200.yaml \
    --arm main \
    --stages_json $STAGES/stages_arm3_excludeOther.json \
    --curriculum_T 480 \
    --run_config full_main_h200
echo "=== DONE $(date) ; gpu log: $GPU_LOG ==="
