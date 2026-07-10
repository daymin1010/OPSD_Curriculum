#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --job-name cliff8b_mainphase
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=7-00:00:00
# ============================================================
# Main-phase 8B 스케일 재현 (H200, C-03 x2). train_cliff4b_h200.sh 기반.
#   사용: sbatch --job-name cliff4b_<ARM> train_mainphase_h200.sh <ARM>
#   ARM ∈ {benchsubj_k1, benchsubj_k2, benchsubj_k3}
#   - 매니페스트: mainphase_20260709/stages_${ARM}.json (벤치정렬 + 난이도재배분)
#   - config: full_4b_main.yaml (context OFF/1024, fixed teacher) — output_dir 내장
#   - ★ --allow_duplicate_pids True: 하드 k배 복제 반영(k=1엔 무영향)
#   - run_config = cliff4b_${ARM} → eval_cliff4b_h200.sh <ARM> 그대로 재사용.
# ============================================================
set -euo pipefail
ARM="${1:?ARM required: benchsubj_k1|benchsubj_k2|benchsubj_k3}"
CONFIG="${CONFIG:-configs/full_8b_h200.yaml}"

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
STAGES=$REPO/src/OPSD_Curriculum/training/mainphase_20260709
ROW=$REPO/src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
ARM_JSON=$STAGES/stages_${ARM}.json
RUN_CONFIG=cliff8b_${ARM}
[ -f "$ARM_JSON" ] || { echo "[ERR] manifest 없음: $ARM_JSON" >&2; exit 2; }

echo "=== job=$SLURM_JOB_ID arm=$ARM node=$(hostname) allow_dup=True $(date) ==="
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
export TEMP=$TMPDIR; export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs" "$WANDB_DIR"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"
[ -f "$ROW" ] || { echo "[ERR] row table 없음: $ROW" >&2; exit 2; }

PORT=$((13100 + SLURM_JOB_ID % 300))
"$REPO/envs/opsd/bin/python" -m accelerate.commands.launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 8 \
    --main_process_port $PORT \
    train_opsd_curriculum_manifest_once.py \
    --config "$CONFIG" \
    --arm "$ARM" \
    --stages_json "$ARM_JSON" \
    --within_stage_order shuffle \
    --tail_policy partial \
    --curriculum_passes 1 \
    --allow_duplicate_pids True \
    --run_config "$RUN_CONFIG"
echo "=== DONE arm=$ARM $(date) ==="
