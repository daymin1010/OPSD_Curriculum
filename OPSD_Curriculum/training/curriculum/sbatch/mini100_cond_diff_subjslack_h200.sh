#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --job-name opsd_mini100_cond_diff_subjslack_h200
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=06:00:00
# ============================================================
# mini100 diff — subjslack (level_backbone_residual_subject_slack, alpha=2.0)
# Qwen3-8B, H200 x2, B_glob=32 (pd2*ga8*ws2). manifest: mini100, arm=diff.
# diff & ours share IDENTICAL universe; only stage assignment differs.
# ============================================================
set -euo pipefail

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
STAGES=$REPO/src/OPSD_Curriculum/training/stages_subjslack_20260624
ROW=$REPO/src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
DIFF_JSON=$STAGES/stages_cond2_diff_mini100.json
OURS_JSON=$STAGES/stages_cond3_ours_subjslack_mini100.json
ARM_JSON=$DIFF_JSON
RUN_CONFIG=mini100_cond_diff_subjslack_h200

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

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs" "$WANDB_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

GPU_LOG="$REPO/runs/gpu_mem_${SLURM_JOB_NAME}.${SLURM_JOB_ID}.log"
( while true; do date +%s | tr -d '\n'; echo -n " "; \
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr '\n' ';'; \
  echo ""; sleep 15; done ) > "$GPU_LOG" 2>/dev/null &
GPU_LOGGER_PID=$!
trap 'kill $GPU_LOGGER_PID 2>/dev/null || true; rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

if [ ! -f "$ROW" ]; then echo "[mini100] missing row table: $ROW" >&2; exit 2; fi

# fairness gate: diff & ours must share identical universe (subset -> expect none)
python verify_schedule_manifest_once.py \
  --diff_json "$DIFF_JSON" \
  --ours_json "$OURS_JSON" \
  --row_table "$ROW" \
  --B_glob 32 \
  --seed 42 \
  --within_stage_order shuffle \
  --curriculum_passes 1 \
  --expect_universe none

"$REPO/envs/opsd/bin/python" -m accelerate.commands.launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12993 \
    train_opsd_curriculum_manifest_once.py \
    --config configs/mini100_8b_h200_subjslack.yaml \
    --arm cond_diff_mini100_subjslack \
    --stages_json "$ARM_JSON" \
    --within_stage_order shuffle \
    --tail_policy partial \
    --curriculum_passes 1 \
    --run_config "$RUN_CONFIG"

echo "=== DONE $(date) ; gpu log: $GPU_LOG ==="
