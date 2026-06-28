#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --job-name eval_p12m_diff
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=18:00:00
#
# full subjslack DIFF — MATH-500 only, step900, pass@12 (val_n=12, temp 1.0). TP=2, non-thinking.
# RESUME: 결과 json 존재 시 skip.
set -euo pipefail
REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
EVAL=$OPSD_SRC/eval/evaluate_math.py
BASE_MODEL=Qwen/Qwen3-8B
CKPT_BASE=$REPO/checkpoints/opsd_curriculum/full_8b_subjslack/full_cond_diff_subjslack_h200
OUTDIR=$REPO/outputs/eval_opsd_curriculum/full_subjslack_cond_diff_nonthink
ARM=full_diff
STEP=900
VALN=12
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
CKPT="$CKPT_BASE/checkpoint-${STEP}"
if [ ! -d "$CKPT" ]; then echo "[FATAL] $CKPT not found" >&2; exit 2; fi
OUT="$OUTDIR/math500_${ARM}_step${STEP}_nonthink_valn${VALN}.json"
if [ -f "$OUT" ]; then echo "[RESUME-SKIP] exists: $(basename $OUT)"; exit 0; fi
echo "[$(date +%H:%M:%S)] EVAL ${ARM} step=${STEP} dataset=math500 val_n=${VALN} TP=2 non-thinking"
python "$EVAL" --base_model "$BASE_MODEL" --checkpoint_dir "$CKPT" \
  --dataset math500 --val_n "$VALN" --temperature 1.0 \
  --tensor_parallel_size 2 --gpu_memory_utilization 0.9 --no_thinking \
  --output_file "$OUT"
echo "[$(date +%H:%M:%S)] DONE ${ARM} math500 pass@12"
