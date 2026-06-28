#!/bin/bash
#SBATCH --job-name=nait_pilot_qwen3
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_pilot_qwen3.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-nait_pilot_qwen3.%j.%N.err

# ── Pilot run: 2,666 samples, 4-way data-parallel (single job, 4 GPUs) ───────
#
# This single sbatch allocates 4 L40S GPUs and spawns 4 python processes
# concurrently — one per GPU — each handling ~666 samples (chunk 0..3).
# All chunks share the same OUTPUT_DIR (per-sample .pt, no collision).
#
# Submit:
#   sbatch src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_pilot.sh

set -euo pipefail

WORKSPACE="/scratch/lami2026/personal/jimin_2782"
SCRIPT="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b.py"
OUTPUT_DIR="${WORKSPACE}/src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts"
LOG_DIR="${WORKSPACE}/runs/nait_pilot_qwen3_${SLURM_JOB_ID}"
NUM_CHUNKS=4

echo "======================================================================"
echo "  NAIT Activation Shift Pilot — Qwen3-8B (4-way parallel)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "======================================================================"

# ── Environment ────────────────────────────────────────────────────────────
set +u
source /home/lami2026/archive/gusrl/miniconda3/etc/profile.d/conda.sh
conda activate verl_new
set -u
export HF_HOME=/home/lami2026/.cache/huggingface
export PYTHONNOUSERSITE=1

echo "[ENV] Python: $(which python)"
echo "[ENV] CUDA visible (job-level): ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ── MPS daemon ────────────────────────────────────────────────────────────
unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || echo "[INFO] MPS daemon already running or skipped"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Spawn 4 chunks in parallel, one per GPU ───────────────────────────────
PIDS=()
for CHUNK_ID in 0 1 2 3; do
    CHUNK_LOG="${LOG_DIR}/chunk${CHUNK_ID}.log"
    echo "[SPAWN] chunk=${CHUNK_ID}  GPU=${CHUNK_ID}  log=${CHUNK_LOG}"

    # Each process sees ONLY one GPU → device cuda:0 inside python maps to physical GPU CHUNK_ID
    CUDA_VISIBLE_DEVICES=${CHUNK_ID} \
    python "${SCRIPT}" \
        --model-id "Qwen/Qwen3-8B" \
        --output-dir "${OUTPUT_DIR}" \
        --max-new-tokens 4096 \
        --chunk-id "${CHUNK_ID}" \
        --num-chunks "${NUM_CHUNKS}" \
        --device cuda \
        --resume \
        > "${CHUNK_LOG}" 2>&1 &
    PIDS+=($!)
    # small stagger so model-download / FS contention is gentler
    sleep 5
done

echo "[PARENT] All 4 chunks spawned. PIDs: ${PIDS[@]}"
echo "[PARENT] Waiting for all chunks to complete..."

# Wait for all chunks. Collect each exit code.
EXIT_OVERALL=0
for i in 0 1 2 3; do
    if wait "${PIDS[$i]}"; then
        echo "[PARENT] chunk ${i} (pid ${PIDS[$i]}): OK"
    else
        rc=$?
        echo "[PARENT] chunk ${i} (pid ${PIDS[$i]}): FAILED (rc=${rc})"
        EXIT_OVERALL=1
    fi
done

echo ""
echo "======================================================================"
echo "  Pilot finished | overall exit code: ${EXIT_OVERALL}"
echo "  Date: $(date)"
echo "======================================================================"

# ── Quick validation ──────────────────────────────────────────────────────
echo "[VALIDATE] .pt file count in output dir:"
ls "${OUTPUT_DIR}"/*.pt 2>/dev/null | wc -l || echo "0 .pt files found"

echo "[VALIDATE] metadata row count:"
wc -l < "${OUTPUT_DIR}/shifts_metadata.jsonl" 2>/dev/null || echo "no metadata file"

echo "[VALIDATE] per-chunk log tails:"
for i in 0 1 2 3; do
    echo "--- chunk ${i} (last 8 lines) ---"
    tail -8 "${LOG_DIR}/chunk${i}.log" 2>&1 || true
done

exit ${EXIT_OVERALL}
