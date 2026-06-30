#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --job-name eval_fullc_cond5
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=20:00:00
#
# full subjslack COND5 (diffmatched seed0) step-curve — checkpoints 100·400·650·900. TP=2, non-thinking.
# AIME24/25 + HMMT25 (val_n=12, pass@12) + MATH500 (val_n=1, pass@1)  ← ours/diff step900 헤드라인과 동일 config.
# load-bearing 비교: ours vs cond5 (난이도 매칭 + level내 subject 랜덤) → subject 기하 효과 입증.
# RESUME: 이미 존재하는 결과 json은 skip → timeout/중단 후 재제출하면 남은 것만 이어서.
# 제출: sbatch --dependency=afterok:94217 eval_fullcurve_cond5_l40s.sh   (cond5 학습 94217 끝나면 자동 시작)

set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-8B
CKPT_BASE=$REPO/checkpoints/opsd_curriculum/full_8b_subjslack/full_cond_cond5seed0_subjslack_h200
OUTDIR=$REPO/outputs/eval_opsd_curriculum/full_subjslack_cond_cond5seed0_nonthink
ARM=full_cond5seed0
# step900(load-bearing: ours vs cond5)을 맨 먼저 → 20h 안에 핵심 결과 우선 확보. 나머지는 시간 남으면 채움.
STEPS="900 650 400 100"

source $REPO/miniforge3/etc/profile.d/conda.sh
conda activate $REPO/envs/opsd
export HF_HOME=$REPO/cache/huggingface
export HF_HUB_OFFLINE=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NODE_CACHE=/dev/shm/jimin_2782_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${SLURM_JOB_ID}
export TEMP=$TMPDIR; export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${PYTHONPATH:-}
export NCCL_P2P_DISABLE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$OUTDIR" "$REPO/runs"
trap 'rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$OPSD_SRC/eval"

for STEP in $STEPS; do
  CKPT="$CKPT_BASE/checkpoint-${STEP}"
  if [ ! -d "$CKPT" ]; then echo "[SKIP] $CKPT not found"; continue; fi
  for ds_valn in "aime24 12" "aime25 12" "hmmt25 12" "math500 1"; do
    ds=$(echo $ds_valn | cut -d' ' -f1); valn=$(echo $ds_valn | cut -d' ' -f2)
    OUT="$OUTDIR/${ds}_${ARM}_step${STEP}_nonthink_valn${valn}.json"
    if [ -f "$OUT" ]; then echo "[$(date +%H:%M:%S)] [RESUME-SKIP] exists: $(basename $OUT)"; continue; fi
    echo "[$(date +%H:%M:%S)] EVAL ${ARM} step=${STEP} dataset=${ds} val_n=${valn} TP=2 non-thinking"
    python "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
      --dataset "$ds" --val_n "$valn" --temperature 1.0 \
      --tensor_parallel_size 2 --gpu_memory_utilization 0.9 --no_thinking \
      --output_file "$OUT"
    echo "[$(date +%H:%M:%S)] DONE ${ARM} step=${STEP} dataset=${ds}"
  done
done
echo "ALL full-curve ${ARM} DONE"
