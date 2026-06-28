#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --job-name qwen3_smoke_h200
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time=00:30:00

# ============================================================
# Qwen3-8B Pass Rate — Smoke Test (H200 2×, tp=2)
#
# 목적: 10 sample smoke test
#   - vLLM tp=2 호환성 확인 (Qwen3-8B: 32 heads, 8 KV heads → tp=2 OK)
#   - enable_thinking=False 효과 확인 (응답에 <think> 없는지)
#   - answer extraction OK (math_verify)
#   - wallclock 추정 (× 266.6 = 2,666 full)
#
# 제출:
#   sbatch sbatch/run_smoke_test_h200.sh
# ============================================================

set -euo pipefail

echo "=== Qwen3-8B Pass Rate Smoke Test (H200 2×) ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Time    : $(date)"

# ── Conda / env
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new

# ── GPU setup
export HF_HOME=/home/lami2026/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || true   # MPS daemon (배려용)

# ── vLLM multiproc: spawn 방식 강제 (fork → CUDA re-init 에러 방지)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONNOUSERSITE=1

# ── CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Script dir
SCRIPT_DIR="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b"
cd "$SCRIPT_DIR"

echo "=== Starting smoke test (n=10, tp=2) ==="
python pass_rate_measurement.py \
    --n_samples 10 \
    --output_path outputs/smoke_test_h200.parquet \
    --n_rollouts 8 \
    --max_tokens 4096 \
    --tp_size 2 \
    --gpu_mem_util 0.85 \
    --temperature 1.0 \
    --top_p 0.95

echo ""
echo "=== Smoke test complete — running summary ==="
python compute_summary.py \
    --input_path outputs/smoke_test_h200.parquet \
    --output_txt outputs/smoke_test_h200_summary.txt

echo ""
echo "=== First row of parquet ==="
python -c "
import pandas as pd
df = pd.read_parquet('outputs/smoke_test_h200.parquet')
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
