#!/bin/bash
#SBATCH --job-name=opsdcl_envchk
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# =============================================================================
# OPSD-Curriculum  GATE 2  — environment / resource discovery (polite, 1 GPU).
#   - Discovers GPU count/type on iREMB-C-03 & iREMB-C-07 via scontrol (NO alloc)
#   - Checks for conda env `opsd`
#   - Probes our analysis env (verl_new) for torch/CUDA/flash-attn/trl GOLD
#   - Confirms Qwen/Qwen3-8B is cached in our HF_HOME (no download attempt)
# Nothing is trained, downloaded, or written outside our personal dir.
# Individual probes are non-fatal (set +e) so the report is always complete.
# =============================================================================
set -uo pipefail
trap 'echo "[exit] rc=$? $(date)"' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1          # never hit the network during a discovery job
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BASE=/scratch/lami2026/personal/jimin_2782
PY=$BASE/envs/verl_new/bin/python

echo "============================================================"
echo "[INFO] host=$(hostname)  date=$(date)"
echo "============================================================"

echo; echo "### 1. Allocated GPU (this job) ###"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

echo; echo "### 2. Full node inventory (scontrol, no allocation) ###"
for N in iREMB-C-03 iREMB-C-07; do
  echo "--- $N ---"
  scontrol show node "$N" 2>/dev/null | grep -E "NodeName|CfgTRES|AllocTRES|Gres=|State=|RealMemory|CPUTot" || echo "  (scontrol unavailable)"
done

echo; echo "### 3. Partitions ###"
sinfo -o "%P %a %l %D %N %G" 2>/dev/null || echo "  (sinfo unavailable)"

echo; echo "### 4. conda env list (looking for 'opsd') ###"
if command -v conda >/dev/null 2>&1; then
  conda env list || true
else
  echo "  conda not on PATH; checking common envs dir"
  ls -d "$BASE"/envs/* 2>/dev/null || true
fi

echo; echo "### 5. analysis env (verl_new) library probe ###"
if [ -x "$PY" ]; then
  "$PY" - <<'PYEOF'
def safe(label, fn):
    try:
        print(f"  [OK]   {label}: {fn()}")
    except Exception as e:
        print(f"  [MISS] {label}: {type(e).__name__}: {e}")

import importlib
safe("torch", lambda: __import__("torch").__version__)
def _cuda():
    import torch
    return f"available={torch.cuda.is_available()} n={torch.cuda.device_count()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'}"
safe("cuda", _cuda)
safe("transformers", lambda: __import__("transformers").__version__)
safe("trl", lambda: __import__("trl").__version__)
def _gold():
    import importlib
    importlib.import_module("trl.experimental.gold")
    return "trl.experimental.gold import OK"
safe("trl GOLD", _gold)
safe("peft", lambda: __import__("peft").__version__)
safe("datasets", lambda: __import__("datasets").__version__)
safe("vllm", lambda: __import__("vllm").__version__)
safe("flash_attn", lambda: __import__("flash_attn").__version__)
safe("math_verify", lambda: __import__("math_verify").__version__ if hasattr(__import__("math_verify"),"__version__") else "imported")
safe("accelerate", lambda: __import__("accelerate").__version__)
PYEOF
else
  echo "  [MISS] $PY not executable"
fi

echo; echo "### 6. Qwen/Qwen3-8B cache presence (offline, no download) ###"
SNAP="$HF_HUB_CACHE/models--Qwen--Qwen3-8B"
if [ -d "$SNAP" ]; then
  echo "  [OK] cache dir exists: $SNAP"
  du -sh "$SNAP" 2>/dev/null || true
  find "$SNAP/snapshots" -maxdepth 2 -name "config.json" 2>/dev/null | head
else
  echo "  [MISS] $SNAP not found — will need (offline) staging before 8B run"
  echo "  (existing Qwen caches:)"
  ls -d "$HF_HUB_CACHE"/models--Qwen--* 2>/dev/null || echo "    none"
fi

echo; echo "### 7. OPSD dataset cache (siyanzhao/Openthoughts_math_30k_opsd) ###"
ls -d "$HF_HOME"/datasets/siyanzhao___* 2>/dev/null || \
ls -d "$HF_HUB_CACHE"/datasets--siyanzhao--* 2>/dev/null || \
echo "  [MISS] OPSD dataset not cached"

echo; echo "[INFO] env_check done $(date)"
