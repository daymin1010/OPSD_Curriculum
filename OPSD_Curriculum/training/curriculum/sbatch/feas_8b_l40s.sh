#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --job-name opsd_cur_feas_8b_l40s
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=03:00:00
# ============================================================
# 8B FEASIBILITY — Qwen3-8B on L40S (46GB) ×4 (ws=4). *** CORE FEASIBILITY GOAL ***
# Can 8B colocate-train on L40S? B_glob=32 via pd1*ga8*ws4. gpu_mem_util=0.45.
# Short probe T=8 (diff-only) to surface OOM ceiling + measure step time.
# If OOM: lower vllm_gpu_memory_utilization further (0.4/0.35), keep pd1.
# Background GPU-mem logger writes to runs/. Run AFTER 1.7B smoke gate passes.
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

# Background GPU memory logger (feasibility evidence)
GPU_LOG="$REPO/runs/gpu_mem_${SLURM_JOB_NAME}.${SLURM_JOB_ID}.log"
( while true; do date +%s | tr -d '\n'; echo -n " "; \
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr '\n' ';'; \
  echo ""; sleep 15; done ) > "$GPU_LOG" 2>/dev/null &
GPU_LOGGER_PID=$!
trap 'kill $GPU_LOGGER_PID 2>/dev/null || true' EXIT

if [ ! -f "$ROW" ]; then echo "[phase0] building join table"; python curriculum_schedule.py phase0; fi

accelerate launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12954 \
    train_opsd_curriculum.py \
    --config configs/full_8b_l40s.yaml \
    --arm diffonly \
    --stages_json $STAGES/stages_diffonly_setA.json \
    --curriculum_T 8 \
    --run_config feas_8b_l40s
echo "=== DONE $(date) ; gpu log: $GPU_LOG ==="
