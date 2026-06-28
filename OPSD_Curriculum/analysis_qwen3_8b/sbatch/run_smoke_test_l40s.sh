#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --job-name qwen3_smoke_l40s
#SBATCH --partition=l40sq
#SBATCH -w iREMB-C-07
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=00:30:00

# ============================================================
# Qwen3-8B Pass Rate — Smoke Test (L40s 4×, tp=4)
#
# 목적: 10 sample smoke test
#   - vLLM tp=4 호환성 확인 (Qwen3-8B: 32 heads, 8 KV heads → tp=4 OK)
#   - enable_thinking=False 효과 확인 (응답에 <think> 없는지)
#   - answer extraction OK (math_verify)
#   - wallclock 추정 (× 266.6 = 2,666 full)
#
# 제출:
#   sbatch sbatch/run_smoke_test_l40s.sh
# ============================================================

set -euo pipefail

echo "=== Qwen3-8B Pass Rate Smoke Test (L40s 4×) ==="
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
# compute node의 /tmp 캐시 정리 (노드에서 직접 실행됨 — 로그인 노드 아님)
rm -rf /tmp/torchinductor_lami2026/ 2>/dev/null || true

# ── CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Script dir
SCRIPT_DIR="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b"
cd "$SCRIPT_DIR"

echo "=== Starting smoke test (n=10, tp=4) ==="
python pass_rate_measurement.py \
    --n_samples 10 \
    --output_path outputs/smoke_test_l40s.parquet \
    --n_rollouts 8 \
    --max_tokens 4096 \
    --tp_size 4 \
    --gpu_mem_util 0.85 \
    --temperature 1.0 \
    --top_p 0.95

echo ""
echo "=== Smoke test complete — running summary ==="
python compute_summary.py \
    --input_path outputs/smoke_test_l40s.parquet \
    --output_txt outputs/smoke_test_l40s_summary.txt

echo ""
echo "=== First row of parquet ==="
python -c "
import pandas as pd
df = pd.read_parquet('outputs/smoke_test_l40s.parquet')
row = df.iloc[0]
print('sample_id  :', row['sample_id'])
print('gt         :', row['ground_truth'])
print('pass_rate  :', row['pass_rate'])
print('pass_count :', row['pass_count'])
print('subject    :', row['subject'])
print('level      :', row['level'])
print('trunc_count:', row['truncation_count'])
print('mean_len   :', row['mean_response_length'])
print()
print('Response[0] (first 300 chars):')
print(row['raw_responses'][0][:300])
print()
print('All', len(df), 'rows:')
print(df[['sample_id','pass_rate','pass_count','subject','level']].to_string())
"

echo ""
echo "=== DONE ==="
echo "Time: $(date)"
