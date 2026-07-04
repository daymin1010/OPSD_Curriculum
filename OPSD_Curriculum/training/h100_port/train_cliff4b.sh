#!/bin/bash
# ============================================================
# 4B 커리큘럼 학습 — H100 4장, 직접 실행(SLURM 아님). 포터블.
#   사용: REPO=/path/to/repo_root ./train_cliff4b.sh <ARM> [SEED]
#   ARM ∈ {shuffle, diff, cliff_P, cliff_subjgeo, cliff_subjrand_s0, cliff_subjrand_s1}
#   SEED(옵션): 학습/스케줄 seed. 생략 시 42. 지정 시 run_config에 _s<SEED> 접미사.
#   context_scaling은 full_4b_cliff.yaml(context_scaling: true)에서 자동 ON.
#   REPO = OPSD_Curriculum의 상위(= src 또는 repo root). 모델/데이터 세팅은 SETUP.md 참조.
# ============================================================
set -euo pipefail
ARM="${1:?ARM required: shuffle|diff|cliff_P|cliff_subjgeo|cliff_subjrand_s0|cliff_subjrand_s1|diff5|diff5_subj}"
SEED="${2:-}"                # 옵션: 생략 시 기본 seed(42), run_config 접미사 없음
CONFIG="${CONFIG:-configs/full_4b_cliff.yaml}"   # env override (teacher-update: configs/full_4b_diff5_ema.yaml)
RUN_TAG="${RUN_TAG:-}"       # run_config/체크포인트 접미사 (예: _ema → fixed와 구분)
: "${REPO:?REPO env 필요 (OPSD_Curriculum 상위 경로). 예: export REPO=\$HOME/opsd}"
: "${ENV_PY:=python}"        # conda 환경 python (SETUP.md대로 만들고 activate 후 실행)
# ⚠️ 반드시 2장/arm: B_glob=32 = per_device 2 × grad_accum 8 × world 2 (메인서버와 동일).
# 4장으로 돌리면 world=4 → B_glob=64 → 커리큘럼 모니터가 abort함.
# 2장씩 2 arm 병렬: 각각 CUDA_VISIBLE_DEVICES=0,1 / 2,3 + PORT 다르게.
NPROC="${NPROC:-2}"          # 2장/arm 고정

OPSD_SRC=$REPO/OPSD_Curriculum/training/opsd_src
CUR=$REPO/OPSD_Curriculum/training/curriculum
STAGES=$REPO/OPSD_Curriculum/training/stages_cliff4b_20260630
ROW=$REPO/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
ARM_JSON=$STAGES/stages_${ARM}.json
if [ -n "$SEED" ]; then RUN_CONFIG=cliff4b_${ARM}${RUN_TAG}_s${SEED}; SEED_ARGS="--seed $SEED --curriculum_seed $SEED";
else RUN_CONFIG=cliff4b_${ARM}${RUN_TAG}; SEED_ARGS=""; fi
WORK="${WORK:-$REPO/_run}"   # 체크포인트/캐시 출력 루트 (config output_dir도 여기 기준 권장)

[ -f "$ARM_JSON" ] || { echo "[ERR] manifest 없음: $ARM_JSON" >&2; exit 2; }
[ -f "$ROW" ]      || { echo "[ERR] row table 없음: $ROW (SETUP.md의 데이터 전송 확인)" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # H100 0-3번 사용
export HF_HOME="${HF_HOME:-$WORK/hf}"
export WANDB_PROJECT=OPSD_Curriculum
export WANDB_MODE="${WANDB_MODE:-offline}"   # 외부 wandb 미사용 기본
export WANDB_DIR="$WORK/wandb"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export NODE_CACHE="$WORK/cache/node_${ARM}"
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR="$WORK/cache/tmp_${ARM}"; export TEMP=$TMPDIR TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$WANDB_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

# H100 accelerate 설정: opsd_src의 cpu-offload 대신 offload 끈 버전(cpu_adam 컴파일 회피).
ACCEL_CONFIG="${ACCEL_CONFIG:-$REPO/OPSD_Curriculum/training/h100_port/accelerate_h100.yaml}"
echo "=== [H100] arm=$ARM nproc=$NPROC accel=$ACCEL_CONFIG $(date) ==="
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
    $SEED_ARGS \
    --run_config "$RUN_CONFIG"
echo "=== [H100] DONE arm=$ARM $(date) ==="
