#!/bin/bash
# 4B 커리큘럼 eval — H100, 직접 실행. 포터블.
#   사용: REPO=/path ./eval_cliff4b.sh <ARM> ["100 400 650"]
#   기본: 최종 체크포인트(예: 899)만 자동 감지해 eval. 중간 커브 원하면 인자로 전달.
set -euo pipefail
ARM="${1:?ARM required}"; STEPS="${2:-}"
: "${REPO:?REPO env 필요}"; : "${ENV_PY:=python}"
OPSD_SRC=$REPO/OPSD_Curriculum/training/opsd_src
CUR=$REPO/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-4B
WORK="${WORK:-$REPO/_run}"
CKPT_BASE="${CKPT_BASE:-$WORK/checkpoints/full_4b_cliff/${RUN_PREFIX:-cliff4b}_${ARM}}"
# 최종 체크포인트 자동 감지(899 등) → STEPS에 없으면 추가
FINAL=$(ls -d "$CKPT_BASE"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
[ -n "$FINAL" ] && case " $STEPS " in *" $FINAL "*) ;; *) STEPS="$STEPS $FINAL";; esac
OUTDIR="${OUTDIR:-$WORK/outputs/eval/${RUN_PREFIX:-cliff4b}_${ARM}_nonthink}"
TP="${TP:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$WORK/hf}"; export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export NODE_CACHE="$WORK/cache/eval_${ARM}"
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm TRITON_CACHE_DIR=$NODE_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR="$WORK/cache/etmp_${ARM}"; export TEMP=$TMPDIR TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}; export NCCL_P2P_DISABLE=1
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$OUTDIR"
cd "$OPSD_SRC/eval"
for STEP in $STEPS; do
  CKPT="$CKPT_BASE/checkpoint-${STEP}"
  [ -d "$CKPT" ] || { echo "[SKIP] $CKPT"; continue; }
  for dv in "aime24 12" "aime25 12" "hmmt25 12" "math500 1" "minerva 1"; do
    ds=$(echo $dv|cut -d' ' -f1); vn=$(echo $dv|cut -d' ' -f2)
    OUT="$OUTDIR/${ds}_${RUN_PREFIX:-cliff4b}_${ARM}_step${STEP}_nonthink_valn${vn}.json"
    [ -f "$OUT" ] && { echo "[SKIP] $(basename $OUT)"; continue; }
    "$ENV_PY" "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
      --dataset "$ds" --val_n "$vn" --temperature 1.0 \
      --tensor_parallel_size "$TP" --gpu_memory_utilization 0.9 --no_thinking --output_file "$OUT"
  done
done
echo "ALL eval $ARM DONE"
