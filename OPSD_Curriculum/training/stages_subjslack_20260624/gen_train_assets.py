#!/usr/bin/env python3
"""
gen_train_assets.py — generate configs (4) + sbatch (8) for subjslack training.
Templates copied from curriculum/configs/quarter_8b_h200.yaml and
curriculum/sbatch/quarter_cond3_ours_q4_h200.sh (only the differing fields vary).
diff & ours share config; only stages_json / arm / port differ.
"""
from pathlib import Path

REPO = Path("/scratch/lami2026/personal/jimin_2782")
CUR = REPO / "src/OPSD_Curriculum/training/curriculum"
SD = REPO / "src/OPSD_Curriculum/training/stages_subjslack_20260624"
CFG_DIR = CUR / "configs"
SB_DIR = CUR / "sbatch"

# scale -> (manifest_suffix, save_steps, save_total_limit, time)
SCALES = {
    "mini50":  ("mini50",  10,  8, "06:00:00"),
    "mini100": ("mini100", 20,  8, "06:00:00"),
    "mini150": ("mini150", 15, 12, "09:00:00"),
    "quarter": ("q4",      10, 30, "12:00:00"),
}
# (scale, arm) -> main_process_port  (unique; avoid existing 12973/12981-84)
PORTS = {
    ("mini50", "diff"): 12991, ("mini50", "ours"): 12992,
    ("mini100", "diff"): 12993, ("mini100", "ours"): 12994,
    ("mini150", "diff"): 12995, ("mini150", "ours"): 12996,
    ("quarter", "diff"): 12997, ("quarter", "ours"): 12998,
}

CONFIG_TMPL = """# ============================================================
# {scale} subjslack config — Qwen3-8B on H200 (x2, ws=2).
# curriculum = level_backbone_residual_subject_slack (alpha=2.0).
# save_steps={save_steps}, save_total_limit={limit}.
# ============================================================
model_name_or_path: Qwen/Qwen3-8B
torch_dtype: bfloat16
attn_implementation: flash_attention_2
use_peft: true
lora_r: 64
lora_alpha: 128
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

learning_rate: 5.0e-6
max_grad_norm: 0.1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
gradient_checkpointing: true
num_train_epochs: 1
max_completion_length: 1024
max_length: 20000
save_strategy: steps
save_steps: {save_steps}
save_total_limit: {limit}
load_best_model_at_end: false
eval_strategy: "no"
logging_steps: 2
beta: 0
lmbda: 1
temperature: 1.1
top_p: 0.95
top_k: 20
use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.6
vllm_tensor_parallel_size: 1
output_dir: /scratch/lami2026/personal/jimin_2782/checkpoints/opsd_curriculum/{scale}_8b_subjslack
wandb_project: OPSD_Curriculum

fixed_teacher: true
jsd_token_clip: 0.06
reason_first: false
student_thinking: false
teacher_thinking: true
curriculum_B_glob: 32
attach_gold: true
"""

SBATCH_TMPL = """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --job-name opsd_{scale}_cond_{arm}_subjslack_h200
#SBATCH --partition=h200q
#SBATCH -w iREMB-C-03
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err
#SBATCH --time={time}
# ============================================================
# {scale} {arm} — subjslack (level_backbone_residual_subject_slack, alpha=2.0)
# Qwen3-8B, H200 x2, B_glob=32 (pd2*ga8*ws2). manifest: {mfsuffix}, arm={arm}.
# diff & ours share IDENTICAL universe; only stage assignment differs.
# ============================================================
set -euo pipefail

REPO=/scratch/lami2026/personal/jimin_2782
OPSD_SRC=$REPO/src/OPSD_Curriculum/training/opsd_src
CUR=$REPO/src/OPSD_Curriculum/training/curriculum
STAGES=$REPO/src/OPSD_Curriculum/training/stages_subjslack_20260624
ROW=$REPO/src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet
DIFF_JSON=$STAGES/stages_cond2_diff_{mfsuffix}.json
OURS_JSON=$STAGES/stages_cond3_ours_subjslack_{mfsuffix}.json
ARM_JSON={arm_json}
RUN_CONFIG={scale}_cond_{arm}_subjslack_h200

echo "=== job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
source $REPO/miniforge3/etc/profile.d/conda.sh
conda activate $REPO/envs/opsd

export HF_HOME=$REPO/cache/huggingface
export WANDB_PROJECT=OPSD_Curriculum
export WANDB_DIR=$REPO/wandb
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export NODE_CACHE=/dev/shm/jimin_2782_${{SLURM_JOB_ID}}
export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/inductor
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export VLLM_CACHE_ROOT=$NODE_CACHE/vllm
export TORCH_EXTENSIONS_DIR=$NODE_CACHE/torch_ext
export TMPDIR=$REPO/cache/tmp/jimin_2782_${{SLURM_JOB_ID}}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PYTHONPATH=$OPSD_SRC:$CUR:${{PYTHONPATH:-}}

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \\
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$REPO/runs" "$WANDB_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$CUR"

GPU_LOG="$REPO/runs/gpu_mem_${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}.log"
( while true; do date +%s | tr -d '\\n'; echo -n " "; \\
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr '\\n' ';'; \\
  echo ""; sleep 15; done ) > "$GPU_LOG" 2>/dev/null &
GPU_LOGGER_PID=$!
trap 'kill $GPU_LOGGER_PID 2>/dev/null || true; rm -rf "$NODE_CACHE" "$TMPDIR" 2>/dev/null || true' EXIT

if [ ! -f "$ROW" ]; then echo "[{scale}] missing row table: $ROW" >&2; exit 2; fi

# fairness gate: diff & ours must share identical universe (subset -> expect none)
python verify_schedule_manifest_once.py \\
  --diff_json "$DIFF_JSON" \\
  --ours_json "$OURS_JSON" \\
  --row_table "$ROW" \\
  --B_glob 32 \\
  --seed 42 \\
  --within_stage_order shuffle \\
  --curriculum_passes 1 \\
  --expect_universe none

"$REPO/envs/opsd/bin/python" -m accelerate.commands.launch \\
    --config_file $OPSD_SRC/accelerate.yaml \\
    --num_processes 2 \\
    --gradient_accumulation_steps 8 \\
    --main_process_port {port} \\
    train_opsd_curriculum_manifest_once.py \\
    --config configs/{scale}_8b_h200_subjslack.yaml \\
    --arm cond_{arm}_{scale}_subjslack \\
    --stages_json "$ARM_JSON" \\
    --within_stage_order shuffle \\
    --tail_policy partial \\
    --curriculum_passes 1 \\
    --run_config "$RUN_CONFIG"

echo "=== DONE $(date) ; gpu log: $GPU_LOG ==="
"""


def main():
    made = []
    for scale, (mfsuffix, save_steps, limit, time) in SCALES.items():
        (CFG_DIR / f"{scale}_8b_h200_subjslack.yaml").write_text(
            CONFIG_TMPL.format(scale=scale, save_steps=save_steps, limit=limit))
        made.append(f"configs/{scale}_8b_h200_subjslack.yaml")
        for arm in ("diff", "ours"):
            arm_json = "$DIFF_JSON" if arm == "diff" else "$OURS_JSON"
            txt = SBATCH_TMPL.format(scale=scale, arm=arm, time=time, mfsuffix=mfsuffix,
                                     arm_json=arm_json, port=PORTS[(scale, arm)])
            fn = SB_DIR / f"{scale}_cond_{arm}_subjslack_h200.sh"
            fn.write_text(txt)
            made.append(f"sbatch/{scale}_cond_{arm}_subjslack_h200.sh")
    print("generated:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
