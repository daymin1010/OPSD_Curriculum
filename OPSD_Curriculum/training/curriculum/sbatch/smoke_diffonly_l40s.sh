#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --job-name opsd_cur_smoke_diff_l40s
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=03:00:00
# ============================================================
# SMOKE — Qwen3-1.7B, diff-only arm, L40S ×4 (ws=4). B_glob=32 (pd2*ga4*ws4; OOM fix 87059).

# T=8 (4 stages × 2 opt-steps; T % 4 == 0). GATE: curriculum/stage_respected==1.0
# for ALL steps. reward_proxy option C: loss + completion length only.
# Submit FIRST (L40S priority). NO --exclusive.
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
# .so dirs must stay on an exec-enabled fs (/dev/shm). The DeepSpeed cpu_adam
# JIT build (gcc/as .s/.o temps in TMPDIR) is bulky and was filling RAM-backed
# /dev/shm -> "No space left on device". Keep the final loadable .so on /dev/shm
# (exec), but send the bulky assembler scratch (TMPDIR) to roomy scratch disk.
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs" "$WANDB_DIR"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR"' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

# Phase-0 join (idempotent) — only if row table missing
if [ ! -f "$ROW" ]; then echo "[phase0] building join table"; python curriculum_schedule.py phase0; fi

accelerate launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 4 \
    --main_process_port 12950 \
    train_opsd_curriculum.py \
    --config configs/smoke_1p7b_l40s.yaml \
    --arm diffonly \
    --stages_json $STAGES/stages_diffonly_setA.json \
    --curriculum_T 8 \
    --run_config smoke_diffonly_l40s
echo "=== DONE $(date) ==="
