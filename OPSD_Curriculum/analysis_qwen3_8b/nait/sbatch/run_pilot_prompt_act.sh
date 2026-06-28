#!/bin/bash
#SBATCH --job-name=nait_pilot_prompt_act
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_pilot_prompt_act.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_pilot_prompt_act.%j.%N.err

# ── Full pilot 2,666 prompt-only activation extraction (single job) ─────────
# Smoke showed ~0.04s/sample after warmup → ~7 min for full 2666.
# Reserve 45 min just in case.

set -euo pipefail

WORKSPACE="/scratch/lami2026/personal/jimin_2782"
SCRIPT="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/nait/extract_prompt_activation.py"
OUTPUT_DIR="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act"

echo "======================================================================"
echo "  Pilot 2666 — Qwen3-8B prompt-only activation"
echo "  Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "======================================================================"

set +u
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new
set -u

export HF_HOME=/home/lami2026/.cache/huggingface
export PYTHONNOUSERSITE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || echo "[INFO] MPS daemon already running or skipped"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" \
    --model-id "Qwen/Qwen3-8B" \
    --output-dir "${OUTPUT_DIR}" \
    --chunk-id 0 \
    --num-chunks 1 \
    --resume \
    --device cuda

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "  Pilot finished | exit ${EXIT_CODE} | $(date)"
echo "======================================================================"

# Quick summary
python - <<PYEOF
import json, torch
from pathlib import Path

out_dir = Path("${OUTPUT_DIR}")
meta = [json.loads(l) for l in open(out_dir/"prompt_activation_metadata.jsonl") if l.strip()]
ok    = [r for r in meta if r.get("status","").startswith("ok")]
err   = [r for r in meta if r.get("status","") == "error"]
pts   = sorted(out_dir.glob("*.pt"))
print(f"[SUMMARY] metadata rows: {len(meta)}  ok={len(ok)}  err={len(err)}")
print(f"[SUMMARY] .pt files: {len(pts)} (expected 2666)")
if pts:
    d = torch.load(pts[0], map_location="cpu", weights_only=False)
    print(f"[SUMMARY] shape={tuple(d['prompt_act'].shape)} dtype={d['prompt_act'].dtype}")
PYEOF

exit ${EXIT_CODE}
