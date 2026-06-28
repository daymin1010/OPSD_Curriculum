#!/bin/bash
# =============================================================================
# OPSD-Curriculum  —  dedicated training env setup (personal dir ONLY).
#   - Installs Miniforge to a PERSONAL prefix (no shared conda, no ~/.bashrc edit)
#   - Creates env `opsd` at envs/opsd from opsd_src/environment.yml pinned stack
#     (trl==0.26.0, torch==2.8.0, vllm==0.11.0, transformers==4.57.1, ...)
#   - Adds flash-attn==2.8.3 (prebuilt wheel; matches handoff)
#   - Does NOT touch envs/verl_new (analysis env) or any shared files.
# Run:  bash src/OPSD_Curriculum/training/setup_opsd_env.sh  > runs/setup_opsd_env.log 2>&1 &
# Idempotent-ish: skips Miniforge install if prefix already present.
# =============================================================================
set -euo pipefail

BASE=/scratch/lami2026/personal/jimin_2782
FORGE=$BASE/miniforge3
ENV_PREFIX=$BASE/envs/opsd
ENV_YML=$BASE/src/OPSD_Curriculum/training/opsd_src/environment.yml
REQ_TXT=$BASE/src/OPSD_Curriculum/training/opsd_src/requirements_opsd.txt


export HOME_SAVE="${HOME:-}"
# Keep conda from writing to shared HOME: point its config/pkgs into personal dir.
export CONDA_PKGS_DIRS=$BASE/cache/conda_pkgs
export PIP_CACHE_DIR=$BASE/cache/pip
# Hermetic env: ignore the shared ~/.local site-packages so pip installs ALL deps
# (incl. transitive python-dateutil/pytz/tzdata/six) INTO envs/opsd, not user-site.
# Must also be exported by training run wrappers so runtime imports stay isolated.
export PYTHONNOUSERSITE=1

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

echo "==== [1] Miniforge ===="
if [ ! -x "$FORGE/bin/conda" ]; then
  cd "$BASE"
  INSTALLER=Miniforge3-Linux-x86_64.sh
  if [ ! -f "$INSTALLER" ]; then
    echo "[dl] fetching Miniforge installer ..."
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER" -O "$INSTALLER"
  fi
  echo "[install] -> $FORGE (batch, no PATH/bashrc modification)"
  bash "$INSTALLER" -b -p "$FORGE"
  rm -f "$INSTALLER"
else
  echo "[skip] Miniforge already at $FORGE"
fi

# Use conda WITHOUT activating into the shared shell (no `conda init`).
export PATH="$FORGE/bin:$PATH"
source "$FORGE/etc/profile.d/conda.sh"

# Use an ISOLATED condarc so conda ignores the shared ~/.condarc
# (`channels: defaults` -> dead internal Nexus mirror iremb-nr-s:8081).
export CONDARC=$BASE/src/OPSD_Curriculum/training/opsd_src/condarc_opsd.yaml
echo "[INFO] CONDARC=$CONDARC"


echo "==== [2] Create bare env (python=3.10 + pip) at $ENV_PREFIX ===="
# Why `conda create` (NOT `conda env create`): only `conda create` accepts
# `--override-channels -c conda-forge`, which FULLY bypasses the channels merged
# from /etc/conda/condarc (the dead internal Nexus mirror iremb-nr-s:8081).
# `conda env create -f env.yml` merges all condarc channels -> CONNECTION FAILED.
# conda provides ONLY python+pip; the heavy stack is pip-installed in step [2b].
if [ -d "$ENV_PREFIX/conda-meta" ]; then
  echo "[skip] env already exists at $ENV_PREFIX"
else
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.10 pip
fi

PY=$ENV_PREFIX/bin/python
PIPBIN=$ENV_PREFIX/bin/pip

echo "==== [2b] pip install pinned stack (PyPI; no pip.conf overrides present) ===="
# Idempotent: pip skips already-satisfied pins on re-run. Upgrade pip first.
"$PY" -m pip install -U pip
"$PIPBIN" install -r "$REQ_TXT"

echo "==== [3] flash-attn 2.8.3 (PREBUILT wheel; auto-detect ABI) ===="
# IMPORTANT: do NOT let flash-attn failure kill the whole script — the core env
# is already complete. Source-building flash-attn needs nvcc + 20-40min compile,
# and the default `pip install flash-attn` uses build isolation (no torch in the
# overlay env) -> `ModuleNotFoundError: No module named 'torch'`. So we fetch the
# matching PREBUILT wheel from GitHub releases based on the env's torch/cuda/abi.
FA_VER=2.8.3
if "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "[skip] flash-attn present: $("$PY" -c 'import flash_attn;print(flash_attn.__version__)')"
else
  # Detect: torch major.minor (e.g. 2.8), cuda major (12), cxx11abi (TRUE/FALSE), cpXY
  read -r FA_TORCH FA_CU FA_ABI FA_CP < <("$PY" - <<'PYEOF'
import torch, sys
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])   # 2.8
cu = (torch.version.cuda or "12.0").split(".")[0]               # 12
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(tv, cu, abi, cp)
PYEOF
)
  FA_WHL="flash_attn-${FA_VER}+cu${FA_CU}torch${FA_TORCH}cxx11abi${FA_ABI}-${FA_CP}-${FA_CP}-linux_x86_64.whl"
  FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHL}"
  echo "[flash-attn] torch=$FA_TORCH cu=$FA_CU abi=$FA_ABI py=$FA_CP"
  echo "[flash-attn] wheel: $FA_WHL"
  if "$PIPBIN" install "$FA_URL"; then
    echo "[flash-attn] installed from prebuilt wheel"
  else
    echo "[flash-attn][WARN] prebuilt wheel failed; trying source build (--no-build-isolation)"
    "$PIPBIN" install ninja packaging wheel setuptools psutil || true
    "$PIPBIN" install --no-build-isolation "flash-attn==${FA_VER}" \
      || echo "[flash-attn][WARN] source build also failed — env is otherwise usable; revisit later"
  fi
fi


echo "==== [4] Verify ===="
"$PY" - <<'PYEOF'
def chk(label, fn):
    try: print(f"  [OK]   {label}: {fn()}")
    except Exception as e: print(f"  [MISS] {label}: {type(e).__name__}: {e}")
chk("torch", lambda: __import__("torch").__version__)
chk("transformers", lambda: __import__("transformers").__version__)
chk("trl", lambda: __import__("trl").__version__)
def _gold():
    import importlib; importlib.import_module("trl.experimental.gold"); return "import OK"
chk("trl.experimental.gold", _gold)
chk("vllm", lambda: __import__("vllm").__version__)
chk("peft", lambda: __import__("peft").__version__)
chk("accelerate", lambda: __import__("accelerate").__version__)
chk("deepspeed", lambda: __import__("deepspeed").__version__)
chk("datasets", lambda: __import__("datasets").__version__)
chk("flash_attn", lambda: __import__("flash_attn").__version__)
chk("math_verify", lambda: "imported" if __import__("math_verify") else "?")
chk("wandb", lambda: __import__("wandb").__version__)
PYEOF

echo "==== DONE: opsd env at $ENV_PREFIX ===="
