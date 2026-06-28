#!/bin/bash
#SBATCH --job-name=nait_smoke_qwen3
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_smoke_qwen3.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_smoke_qwen3.%j.%N.err

# ── Smoke test: 10 samples, L40S × 4 (allocated, code uses 1), 30 min walltime ─
# Purpose:
#   - Verify non-reasoning mode (no <think> tags in prompt)
#   - Verify hooks fire on all 36 layers
#   - Verify Δ𝒜 shape: {0..35: Tensor(22528,)} bfloat16
#   - Measure per-sample wallclock → extrapolate for pilot run
#   - Check GPU memory margin on L40S (48 GB)
#
# Submit from workspace root:
#   sbatch src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_smoke_test.sh

set -euo pipefail

WORKSPACE="/scratch/lami2026/personal/jimin_2782"
SCRIPT="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b.py"
OUTPUT_DIR="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/smoke_shifts"

echo "======================================================================"
echo "  NAIT Activation Shift Smoke Test — Qwen3-8B (non-reasoning)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "======================================================================"

# ── Environment ────────────────────────────────────────────────────────────
# 주의: /etc/bashrc 에 unbound var 있어서 set -u 와 충돌 → conda.sh 직접 source
set +u
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new
set -u

export HF_HOME=/home/lami2026/.cache/huggingface
export PYTHONNOUSERSITE=1

echo "[ENV] Python: $(which python)"
echo "[ENV] CUDA devices: ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── MPS daemon (배려 있게 사용) ────────────────────────────────────────────
unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || echo "[INFO] MPS daemon already running or skipped"

# ── Smoke run: 10 samples, chunk 0/1 ──────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" \
    --model-id "Qwen/Qwen3-8B" \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens 4096 \
    --num-samples 10 \
    --chunk-id 0 \
    --num-chunks 1 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "  Smoke test finished | exit code: ${EXIT_CODE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Date: $(date)"
echo "======================================================================"

# ── Quick validation ───────────────────────────────────────────────────────
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "[VALIDATE] Checking smoke outputs..."

    python - <<'PYEOF'
import sys, json, torch
from pathlib import Path

out_dir = Path("/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/smoke_shifts")
meta_path = out_dir / "shifts_metadata.jsonl"

# 1. Metadata row count
if not meta_path.exists():
    print("[FAIL] shifts_metadata.jsonl not found!")
    sys.exit(1)

records = [json.loads(l) for l in open(meta_path) if l.strip()]
ok_records = [r for r in records if r.get("status", "").startswith("ok")]
print(f"[CHECK] metadata rows: {len(records)} total, {len(ok_records)} ok")
print(f"  → is_trunc count: {sum(1 for r in ok_records if r.get('is_trunc', False))}/{len(ok_records)}")

# 2. Load a .pt and check structure
pt_files = sorted(out_dir.glob("*.pt"))
print(f"[CHECK] .pt files found: {len(pt_files)}")

if pt_files:
    d = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    shifts = d["shifts"]
    num_layers = len(shifts)
    sample_layer = shifts[0]
    print(f"[CHECK] num layers in shifts dict: {num_layers} (expected 36)")
    print(f"[CHECK] layer-0 tensor shape: {tuple(sample_layer.shape)} (expected (22528,))")
    print(f"[CHECK] layer-0 tensor dtype: {sample_layer.dtype} (expected bfloat16)")
    print(f"[CHECK] is_trunc field in .pt: {'is_trunc' in d}")
    print(f"[CHECK] generated_text (first 120 chars): {d.get('generated_text','')[:120]!r}")
    # Check for <think> in generated_text
    gen_text = d.get("generated_text", "")
    if "<think>" in gen_text or "</think>" in gen_text:
        print("[FAIL] <think> tag found in generated_text!")
        sys.exit(1)
    else:
        print("[CHECK] No <think> tags in generated_text ✓")

print("[CHECK] Smoke validation PASSED ✓")
PYEOF

fi

exit ${EXIT_CODE}
