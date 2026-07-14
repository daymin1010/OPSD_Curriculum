#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --job-name eval_think8b_h200
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=48:00:00
# 8B 커리큘럼 eval (mainphase 스케일 재현) — arm 인자. step 커브 (기본 100·400·650·900).
#   사용: sbatch --job-name think8b_<ARM> eval_think8b_h200.sh <ARM> ["900"]
# AIME24/25 + HMMT25 (val_n=12), THINKING mode (--no_thinking 없음, 원저 run_eval.sh 정합), TP=2. RESUME(있는 json skip).
set -euo pipefail
ARM="${1:?ARM required}"; STEPS="${2:-}"   # 기본: 최종 체크포인트만(아래 자동감지). 중간 커브 원하면 "100 400 650" 처럼 인자로.
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-8B
CKPT_BASE="${CKPT_BASE:-$REPO/checkpoints/opsd_curriculum/full_8b/${RUN_PREFIX:-cliff8b}_${ARM}}"
# 최종 체크포인트 자동 감지(899 등) → STEPS에 없으면 추가 (900 하드코딩 금지)
FINAL=$(ls -d "$CKPT_BASE"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
[ -n "$FINAL" ] && case " $STEPS " in *" $FINAL "*) ;; *) STEPS="$STEPS $FINAL";; esac
OUTDIR="${OUTDIR:-$REPO/outputs/eval_opsd_curriculum/${RUN_PREFIX:-cliff8b}_${ARM}_think}"

source $REPO/miniforge3/etc/profile.d/conda.sh
conda activate $REPO/envs/opsd
export HF_HOME=$REPO/cache/huggingface; export HF_HUB_OFFLINE=0
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export NODE_CACHE=/dev/shm/jimin_2782_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}; export TEMP=$TMPDIR TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}; export NCCL_P2P_DISABLE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$OUTDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT
cd "$OPSD_SRC/eval"

for STEP in $STEPS; do
  CKPT="$CKPT_BASE/checkpoint-${STEP}"
  [ -d "$CKPT" ] || { echo "[SKIP] $CKPT 없음"; continue; }
  for ds_valn in "aime24 12" "aime25 12" "hmmt25 12"; do
    ds=$(echo $ds_valn|cut -d' ' -f1); valn=$(echo $ds_valn|cut -d' ' -f2)
    OUT="$OUTDIR/${ds}_${RUN_PREFIX:-cliff8b}_${ARM}_step${STEP}_think_valn${valn}.json"
    [ -f "$OUT" ] && { echo "[RESUME-SKIP] $(basename $OUT)"; continue; }
    echo "[$(date +%H:%M:%S)] EVAL $ARM step=$STEP $ds val_n=$valn"
    python "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
      --dataset "$ds" --val_n "$valn" --temperature 1.0 \
      --tensor_parallel_size 2 --gpu_memory_utilization 0.9 \
      --output_file "$OUT"
  done
done
echo "ALL eval $ARM DONE"
