#!/bin/bash
#SBATCH --job-name=opsd_envsetup
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

# =============================================================================
# OPSD-Curriculum  —  run the dedicated training env setup on a COMPUTE node.
#   Why sbatch on C-07: login/dev node could not reach package mirrors; compute
#   nodes have proven egress (HF model/dataset downloads succeeded there) and
#   user confirms pip(PyPI) works there. Policy: sbatch only, C-03/07 only,
#   no --exclusive. Everything written under personal dir.
#   The actual logic lives in setup_opsd_env.sh (idempotent, resumable).
# =============================================================================
set -uo pipefail
echo "[INFO] host=$(hostname)  date=$(date)"
echo "[INFO] running setup_opsd_env.sh (conda-forge override, pip stack)"

bash /scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/training/setup_opsd_env.sh

rc=$?
echo "[INFO] setup_opsd_env.sh exit rc=$rc  date=$(date)"
exit $rc
