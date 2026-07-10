#!/bin/bash
# ============================================================
# Main-phase 학습 — H100, 직접 실행(SLURM 아님). train_cliff4b.sh 기반.
#   사용: REPO=/path/to/src ./train_mainphase.sh <ARM>
#   ARM ∈ {benchsubj_k1, benchsubj_k2, benchsubj_k3}
#   - 매니페스트: mainphase_20260709/stages_${ARM}.json (벤치정렬 + 난이도재배분)
#   - config: full_4b_main.yaml (context OFF/1024, fixed teacher)
#   - ★ --allow_duplicate_pids True: 하드 문제 k배 복제가 학습에 반영되게(k=1엔 무영향)
#   - run_config = cliff4b_${ARM} → 기존 eval_cliff4b.sh <ARM> 그대로 재사용.
#   ⚠️ 2장/arm 고정(B_glob=32). 2 arm 병렬 시 CUDA_VISIBLE_DEVICES=0,1 / 2,3 + PORT 다르게.
# ============================================================
set -euo pipefail
ARM="${1:?ARM required: benchsubj_k1|benchsubj_k2|benchsubj_k3|... (stages_<ARM>.json)}"
SEED="${2:-}"                                     # 옵션: seed 재현용. 지정 시 run_config에 _s<SEED>
CONFIG="${CONFIG:-configs/full_4b_main.yaml}"     # context OFF/1024/fixed teacher
: "${REPO:?REPO env 필요 (OPSD_Curriculum 상위 경로)}"
: "${ENV_PY:=python}"
NPROC="${NPROC:-2}"

OPSD_SRC=$REPO/OPSD_Curriculum/training/opsd_src
CUR=$REPO/OPSD_Curriculum/training/curriculum
STAGES=$REPO/OPSD_Curriculum/training/mainphase_20260709
ROW=$REPO/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
ARM_JSON=$STAGES/stages_${ARM}.json
if [ -n "$SEED" ]; then RUN_CONFIG=cliff4b_${ARM}_s${SEED}; SEED_ARGS="--seed $SEED --curriculum_seed $SEED";
else RUN_CONFIG=cliff4b_${ARM}; SEED_ARGS=""; fi   # eval 호환: eval_cliff4b.sh <ARM[_s<SEED>]>
WORK="${WORK:-$REPO/_run}"

[ -f "$ARM_JSON" ] || { echo "[ERR] manifest 없음: $ARM_JSON" >&2; exit 2; }
[ -f "$ROW" ]      || { echo "[ERR] row table 없음: $ROW" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export WANDB_PROJECT=OPSD_Curriculum
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$WORK/wandb"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export NODE_CACHE="$WORK/cache/node_${RUN_CONFIG}"   # seed 병렬 run 캐시 충돌 방지 (ARM 동일해도 분리)
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR="$WORK/cache/tmp_${RUN_CONFIG}"; export TEMP=$TMPDIR TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$WANDB_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

ACCEL_CONFIG="${ACCEL_CONFIG:-$REPO/OPSD_Curriculum/training/h100_port/accelerate_h100.yaml}"
echo "=== [MAINPHASE] arm=$ARM config=$CONFIG allow_dup=True $(date) ==="
"$ENV_PY" -m accelerate.commands.launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NPROC" \
    --gradient_accumulation_steps 8 \
    --main_process_port "${PORT:-13100}" \
    train_opsd_curriculum_manifest_once.py \
    --config "$CONFIG" \
    --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.3}" \
    --output_dir "$WORK/checkpoints/full_4b_cliff" \
    --arm "$ARM" \
    --stages_json "$ARM_JSON" \
    --within_stage_order shuffle \
    --tail_policy partial \
    --curriculum_passes 1 \
    --allow_duplicate_pids True \
    $SEED_ARGS \
    --run_config "$RUN_CONFIG"
echo "=== [MAINPHASE] DONE arm=$ARM $(date) ==="
