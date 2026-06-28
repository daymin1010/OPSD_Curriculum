# NAIT Activation Shift Extraction — Qwen3-8B (non-reasoning mode)

## Overview

**Track B** of the OPSD self-distillation curriculum analysis.  
Extracts NAIT activation shifts Δ𝒜 = 𝒜(t_K) − 𝒜(t_1) for all 2,666 pilot samples using **Qwen3-8B in non-reasoning mode** (non-thinking, greedy decode).

This is a faithful replication of the 4.6_Task2 NAIT protocol on a new base model:
- Original: `DeepSeek-R1-Distill-Qwen-1.5B`, 28 layers, intermediate_size=8960
- This script: `Qwen/Qwen3-8B`, 36 layers, intermediate_size≈22528

---

## Key Protocol Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Input sequence | `prompt + y_hat` (model-generated) | 4.6 protocol; student-mode measurement |
| Generation | Greedy (`do_sample=False`) | Deterministic single trajectory (NAIT requirement) |
| Chat template | `enable_thinking=False`, no system message | Non-reasoning mode; mirrors 4.6 no-system-msg setup |
| Hook location | `layer.mlp.down_proj` pre-hook, all layers | 4.6 protocol exactly |
| t_1 | `prompt_len` (first generated token index) | 4.6 definition |
| t_K | `seq_len - 1` (last token, EOS inclusive) | 4.6 definition |
| dtype | `bfloat16` | Qwen3-8B native dtype |
| max_new_tokens | 4096 | Task spec |
| is_trunc | `num_generated >= max_new_tokens AND last_token != EOS` | Explicit (4.6 had implicit) |

**Do NOT change** y_hat → y*, Greedy → Sampling, or hook location without explicit user approval.

---

## File Structure

```
activation/
├── extract_activation_shifts_qwen3_8b.py   # Main extraction script
├── sbatch/
│   ├── run_smoke_test.sh                   # 10 samples, L40S×1, 30min
│   └── run_pilot.sh                        # 2,666 samples, array 0-3 (4 chunks)
├── outputs/
│   ├── smoke_shifts/                       # Smoke test outputs
│   │   ├── {id}.pt
│   │   ├── shifts_metadata.jsonl
│   │   └── shifts_checkpoint_chunk0.json
│   └── shifts/                             # Pilot outputs (2,666 × .pt)
│       ├── {sample_id}.pt
│       ├── shifts_metadata.jsonl
│       └── shifts_checkpoint_chunk{0..3}.json
└── README.md  (this file)
```

---

## Pilot ID Universe

Identical to Track A (`pass_rate_measurement.py`):

```python
from _nait_common import BASE_DIR as NAIT_BASE, load_metadata, resolve_shift_dirs
dirs = resolve_shift_dirs(None) + [NAIT_BASE / "activation/full_shifts_l7l8"]
df   = load_metadata(dirs)   # → 2,666 rows, deduplicated by id
```

Pilot IDs = `extra_info['index']` in `train_L2.parquet`.

---

## Output Schema

### Per-sample `.pt` file (`{sample_id}.pt`)

```python
{
  "id":                   str,          # = extra_info['index']
  "subject":              str,          # GPT label
  "level":                int,          # GPT label (1..8)
  "shifts":               {             # NAIT Δ𝒜
      layer_idx (int): Tensor(intermediate_size,)  # bfloat16, CPU
      ...                               # 36 layers (0..35)
  },
  "t1_idx":               int,          # prompt_len (first gen token)
  "tK_idx":               int,          # seq_len - 1 (last token)
  "num_generated_tokens": int,
  "is_trunc":             bool,         # num_gen >= 4096 AND last != EOS
  "generated_text":       str,          # decoded response (y_hat)
  "gen_time_sec":         float,
  "forward_time_sec":     float,
  "total_time_sec":       float,
}
```

### `shifts_metadata.jsonl`

One JSON line per sample:
```json
{"id": "12345", "subject": "Algebra", "level": 3, "t1_idx": 118, "tK_idx": 972,
 "num_generated_tokens": 855, "is_trunc": false, "shift_file": "/.../12345.pt",
 "chunk_id": 0, "num_chunks": 4, "gen_time_sec": 45.2, "forward_time_sec": 12.1,
 "total_time_sec": 57.3, "status": "ok"}
```

---

## Storage Estimate

- Per sample: 36 layers × 22528 × 2 bytes (bfloat16) ≈ 1.55 MB
- 2,666 samples: ≈ **4.1 GB** (shift tensors only)
- Including generated_text + metadata: ≈ **4.5–5 GB** total

Check available space: `df -h /scratch/lami2026/`

---

## Procedure

### Step 1 — Smoke Test

```bash
sbatch src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_smoke_test.sh
```

Check output in `runs/slurm-nait_smoke_qwen3.<JOB_ID>.iREMB-C-07.out`.

**Minimum pass criteria:**
- `.pt` files: 10
- `shifts_metadata.jsonl` rows: 10 ok
- `num layers in shifts dict` = 36
- `layer-0 tensor shape` = (22528,)
- `layer-0 tensor dtype` = bfloat16
- `No <think> tags` in generated_text
- Per-sample wallclock: reported in summary

### Step 2 — Update Pilot Walltime

After smoke test, check summary output for:
```
Recommended sbatch walltime per chunk: Xh Ym Zs (×1.5 safety margin)
```

Edit `#SBATCH --time=` in `sbatch/run_pilot.sh` accordingly.

### Step 3 — Submit Pilot

```bash
# Independent (after Track A completes and L40S is free):
sbatch src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_pilot.sh

# With Track A dependency:
sbatch --dependency=afterok:<TRACK_A_JOB_ID> \
       src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_pilot.sh
```

This launches a SLURM array job (4 tasks: chunk 0, 1, 2, 3) — each on L40S×1.

### Step 4 — Verify Completion

```bash
# All 4 chunks done?
ls outputs/shifts/*.pt | wc -l      # → 2666
wc -l < outputs/shifts/shifts_metadata.jsonl  # → ≥ 2666

# Quick schema check
python -c "
import torch
from pathlib import Path
d = torch.load(sorted(Path('outputs/shifts').glob('*.pt'))[0], weights_only=False)
print('layers:', len(d['shifts']))
print('shape:', tuple(d['shifts'][0].shape))
print('dtype:', d['shifts'][0].dtype)
print('is_trunc:', d['is_trunc'])
"
```

---

## Notes

- **4.6_Task2/ is READ-ONLY**. This script imports `_nait_common.py` from there but never modifies anything.
- The `_nait_common.py` has `NUM_LAYERS=28` hardcoded for downstream analysis. The new Qwen3-8B analysis scripts will need `NUM_LAYERS=36` — handle separately in `_nait_common_qwen3_8b.py`.
- Chunks write to same `outputs/shifts/` dir. Per-sample `.pt` filenames are unique (by sample_id), so no collisions.
- `--resume` flag in pilot run allows safe restart after preemption.

---

*Created: 2026-05-21 | Track B of OPSD Curriculum Analysis*
