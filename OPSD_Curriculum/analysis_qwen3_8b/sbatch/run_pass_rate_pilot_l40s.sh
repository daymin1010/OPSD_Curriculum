#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --job-name qwen3_pilot_l40s
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=08:00:00

# ============================================================
# Qwen3-8B Pass Rate — Full Pilot Run (L40s 4×, tp=4)
# 2,666 samples × 8 rollouts each
#
# 사전조건: smoke test 성공 확인 후 제출
# 예상 wallclock: 2~5시간 (smoke test 결과로 캘리브레이션)
#
# 제출:
#   sbatch sbatch/run_pass_rate_pilot_l40s.sh
# ============================================================

set -euo pipefail

echo "=== Qwen3-8B Pass Rate — Full Pilot (L40s 4×) ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Time    : $(date)"

# ── Conda / env
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new

# ── GPU setup
export HF_HOME=/home/lami2026/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || true   # MPS daemon

# ── vLLM: V0 엔진 강제 (V1은 torchinductor 사용 → /tmp noexec에서 .so mmap 실패)
export VLLM_USE_V1=0
# ── vLLM multiproc: spawn 방식 강제 (fork → CUDA re-init 에러 방지)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONNOUSERSITE=1

# ── torchinductor/triton 캐시를 /scratch로 이전 (/tmp noexec 회피)
export TORCHINDUCTOR_CACHE_DIR=/scratch/lami2026/personal/jimin_2782/.torchinductor_cache
export TRITON_CACHE_DIR=/scratch/lami2026/personal/jimin_2782/.triton_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
rm -rf /tmp/torchinductor_lami2026/ 2>/dev/null || true

# ── CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Script dir
SCRIPT_DIR="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b"
cd "$SCRIPT_DIR"

echo "=== Starting full pilot run (n=2666, tp=4) ==="
python pass_rate_measurement.py \
    --output_path outputs/pass_rate_pilot_2666.parquet \
    --n_rollouts 8 \
    --max_tokens 4096 \
    --tp_size 4 \
    --gpu_mem_util 0.85 \
    --temperature 1.0 \
    --top_p 0.95

echo ""
echo "=== Full run complete — running verification checklist ==="
python compute_summary.py \
    --input_path outputs/pass_rate_pilot_2666.parquet \
    --output_txt outputs/pass_rate_summary.txt

echo ""
echo "=== Summary saved to outputs/pass_rate_summary.txt ==="
echo "=== DONE ==="
echo "Time: $(date)"
