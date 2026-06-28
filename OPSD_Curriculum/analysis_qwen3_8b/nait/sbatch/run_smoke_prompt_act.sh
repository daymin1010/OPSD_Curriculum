#!/bin/bash
#SBATCH --job-name=nait_smoke_prompt_act
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_smoke_prompt_act.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_smoke_prompt_act.%j.%N.err

# ── Smoke test: 10 sample prompt-only activation extraction ─────────────────
# Verify:
#   - shape (36, intermediate_size) bfloat16
#   - no NaN
#   - prompt has <|im_start|>assistant, no non-empty <think>
#   - all 36 layers captured
#
# Submit from workspace root:
#   sbatch src/OPSD_Curriculum/analysis_qwen3_8b/nait/sbatch/run_smoke_prompt_act.sh

set -euo pipefail

WORKSPACE="/scratch/lami2026/personal/jimin_2782"
SCRIPT="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/nait/extract_prompt_activation.py"
OUTPUT_DIR="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act_smoke"

echo "======================================================================"
echo "  Smoke — Qwen3-8B prompt-only activation extraction"
echo "  Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "======================================================================"

# ── Environment ────────────────────────────────────────────────────────────
set +u
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new
set -u

export HF_HOME=/home/lami2026/.cache/huggingface
export PYTHONNOUSERSITE=1

echo "[ENV] Python: $(which python)"
echo "[ENV] CUDA devices: ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || echo "[INFO] MPS daemon already running or skipped"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" \
    --model-id "Qwen/Qwen3-8B" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples 10 \
    --chunk-id 0 \
    --num-chunks 1 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "  Smoke finished | exit ${EXIT_CODE} | $(date)"
echo "======================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "[VALIDATE] Checking smoke outputs..."
    python - <<PYEOF
import sys, json, torch
from pathlib import Path

out_dir = Path("${OUTPUT_DIR}")
meta_path = out_dir / "prompt_activation_metadata.jsonl"

if not meta_path.exists():
    print("[FAIL] metadata.jsonl not found!")
    sys.exit(1)

records = [json.loads(l) for l in open(meta_path) if l.strip()]
ok = [r for r in records if r.get("status", "").startswith("ok")]
print(f"[CHECK] metadata: {len(records)} total, {len(ok)} ok")

pt_files = sorted(out_dir.glob("*.pt"))
print(f"[CHECK] .pt files: {len(pt_files)} (expected 10)")
if len(pt_files) == 0:
    print("[FAIL] no .pt files!")
    sys.exit(1)

d = torch.load(pt_files[0], map_location="cpu", weights_only=False)
act = d["prompt_act"]
print(f"[CHECK] prompt_act shape: {tuple(act.shape)} (expected (36, ~12288 or ~22528))")
print(f"[CHECK] prompt_act dtype: {act.dtype} (expected bfloat16)")
print(f"[CHECK] prompt_len: {d['prompt_len']}  last_token_idx: {d['last_token_idx']}")
print(f"[CHECK] subject: {d.get('subject','')}  level: {d.get('level','')}")

f = act.float()
nans = torch.isnan(f).sum().item()
infs = torch.isinf(f).sum().item()
print(f"[CHECK] NaN={nans}  Inf={infs}")
if nans or infs:
    print("[FAIL] NaN/Inf detected!")
    sys.exit(1)

# Per-layer norm sanity
norms = f.norm(dim=1)
print(f"[CHECK] per-layer norm  min={norms.min():.2f}  max={norms.max():.2f}  mean={norms.mean():.2f}")
if (norms == 0).any():
    print("[FAIL] zero-norm layer detected!")
    sys.exit(1)

# Re-construct prompt and check chat-template content
import sys as _s
_s.path.insert(0, "${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/nait")
from extract_prompt_activation import build_prompt, load_train_lookup
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
lookup = load_train_lookup()
sid = d["id"]
if sid in lookup:
    p = build_prompt(tok, lookup[sid])
    print(f"[CHECK] prompt[:200] = {p[:200]!r}")
    assert "<|im_start|>user" in p, "no <|im_start|>user"
    assert "<|im_start|>assistant" in p, "no <|im_start|>assistant"
    import re
    m = re.search(r"<think>(.*?)</think>", p, flags=re.DOTALL)
    if m is not None:
        inner = m.group(1).strip()
        if inner:
            print(f"[FAIL] non-empty <think>: {inner!r}")
            sys.exit(1)
        else:
            print("[CHECK] <think></think> empty ✓")
    else:
        print("[CHECK] no <think> block found ✓")

# Cross-sample shape consistency
for pf in pt_files[1:5]:
    dd = torch.load(pf, map_location="cpu", weights_only=False)
    assert dd["prompt_act"].shape == act.shape, f"shape mismatch in {pf}"
print("[CHECK] cross-sample shape consistent ✓")

print("\n[CHECK] Smoke validation PASSED ✓")
PYEOF
fi

exit ${EXIT_CODE}
