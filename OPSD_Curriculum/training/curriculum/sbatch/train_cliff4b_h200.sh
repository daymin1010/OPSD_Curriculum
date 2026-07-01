#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --job-name cliff4b
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=24:00:00
# ============================================================
# 4B 커리큘럼 학습 (cliff 실험) — arm을 인자로.
#   사용: sbatch --job-name cliff4b_<ARM> train_cliff4b_h200.sh <ARM> [SEED]
#   ARM ∈ {shuffle, diff, cliff_P, cliff_subjgeo, cliff_subjrand_s0, cliff_subjrand_s1}
#         (subj_V1/subj_shuf = 구 subject-primary, 중간선으로 대체)
#   SEED(옵션): 학습/스케줄 seed. 생략 시 42. 지정 시 run_config에 _s<SEED> 접미사.
# Qwen3-4B, H200 x2, B_glob=32. clean universe 28,743.
# context_scaling ON(full_4b_cliff.yaml): stage별 생성 max_new_tokens 1024→4096 램프.
# ============================================================
set -euo pipefail
ARM="${1:?ARM required: shuffle|diff|cliff_P|cliff_subjgeo|cliff_subjrand_s0|cliff_subjrand_s1}"
SEED="${2:-}"   # 옵션: 생략 시 기본 seed(42), run_config 접미사 없음

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
STAGES=$REPO/src/OPSD_Curriculum/training/stages_cliff4b_20260630
ROW=$REPO/src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
ARM_JSON=$STAGES/stages_${ARM}.json
if [ -n "$SEED" ]; then RUN_CONFIG=cliff4b_${ARM}_s${SEED}; SEED_ARGS="--seed $SEED --curriculum_seed $SEED";
else RUN_CONFIG=cliff4b_${ARM}; SEED_ARGS=""; fi
[ -f "$ARM_JSON" ] || { echo "[ERR] manifest 없음: $ARM_JSON" >&2; exit 2; }

echo "=== job=$SLURM_JOB_ID arm=$ARM node=$(hostname) $(date) ==="
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

PORT=$((13100 + SLURM_JOB_ID % 300))   # 같은 노드 동시 실행 시 포트 충돌 방지
"$REPO/envs/opsd/bin/python" -m accelerate.commands.launch \
    --config_file $OPSD_SRC/accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 8 \
    --main_process_port $PORT \
    train_opsd_curriculum_manifest_once.py \
    --config configs/full_4b_cliff.yaml \
    --arm "$ARM" \
    --stages_json "$ARM_JSON" \
    --within_stage_order shuffle \
    --tail_policy partial \
    --curriculum_passes 1 \
    $SEED_ARGS \
    --run_config "$RUN_CONFIG"

echo "=== DONE arm=$ARM $(date) ==="
