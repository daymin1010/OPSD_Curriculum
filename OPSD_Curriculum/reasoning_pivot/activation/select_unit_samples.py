#!/usr/bin/env python3
"""
select_unit_samples.py
======================
Phase Z-2 PILOT sample selection.

unit = (subject, level) cell. Take up to K=30 rows per non-empty cell from the
FULL labeled universe (29,434), deterministic (seed=42). Assign a balanced
chunk_id in {0..NUM_CHUNKS-1} so each sbatch chunk is difficulty/subject-balanced
and roughly equal size.

Outputs (CPU only; safe from login shell):
  outputs/unit30_samples.parquet   -- full pilot set with chunk_id
  outputs/smoke2_samples.parquet   -- ~12 rows (new-code validation), --smoke

Columns kept (incl. ground-truth `answer` for is_correct scoring downstream).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

BASE = Path("/scratch/lami2026/personal/jimin_2782")
FULL = BASE / "src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet"
OUT_DIR = BASE / "src/OPSD_Curriculum/reasoning_pivot/activation/outputs"

K_PER_CELL = 30
NUM_CHUNKS = 4
SEED = 42

KEEP_COLS = [
    "problem_id", "row_index", "source", "subject", "subject_raw", "level",
    "problem_text", "problem_char_len", "problem_qwen_tok_len",
    "r1_cot_token_count", "solution_char_len",
    "answer", "correct",   # answer = ground-truth final answer; correct = R1 solution correctness
]


def select_full() -> pd.DataFrame:
    df = pd.read_parquet(FULL)
    print(f"[INFO] full universe: {df.shape}")
    keep = [c for c in KEEP_COLS if c in df.columns]

    picks = []
    cells = []
    for (subj, lv), grp in df.groupby(["subject", "level"], sort=True):
        n = min(K_PER_CELL, len(grp))
        chosen = grp.sample(n=n, random_state=SEED)
        picks.append(chosen)
        cells.append((subj, int(lv), len(grp), n))

    out = pd.concat(picks, ignore_index=True)[keep].copy()

    # ── balanced chunk assignment ───────────────────────────────────────────
    # Deterministic shuffle, then round-robin chunk_id so each chunk is balanced
    # across cells and roughly equal in size.
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out["chunk_id"] = [i % NUM_CHUNKS for i in range(len(out))]

    assert out["problem_id"].is_unique, "duplicate problem_id in pilot set!"

    # ── report ────────────────────────────────────────────────────────────
    print(f"\n[CELLS] {len(cells)} non-empty (subject x level) cells")
    short = [(s, l, full_n, k) for (s, l, full_n, k) in cells if k < K_PER_CELL]
    print(f"[CELLS] cells with < {K_PER_CELL} available: {len(short)}")
    for s, l, full_n, k in short:
        print(f"        {s:25s} L{l}: full={full_n:5d} -> took {k}")
    print(f"\n[TOTAL] selected = {len(out)} samples")
    print("[CHUNK] sizes:", out["chunk_id"].value_counts().sort_index().to_dict())
    print("\n[SUBJECT x LEVEL] selected counts:")
    print(pd.crosstab(out["subject"], out["level"]))
    return out


def select_smoke() -> pd.DataFrame:
    """~12 rows for NEW-code validation: cover levels 1..8 + extra hard (L6/7/8)
    so the truncation t_k path is exercised. Includes ground-truth `answer`."""
    df = pd.read_parquet(FULL)
    keep = [c for c in KEEP_COLS if c in df.columns]
    picks = []
    plan = [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 5]  # hard-heavy tail
    used = set()
    for lv in plan:
        sub = df[(df["level"] == lv) & (~df["problem_id"].isin(used))]
        if len(sub) == 0:
            continue
        chosen = sub.sample(n=1, random_state=SEED + lv + len(used))
        used.add(chosen.iloc[0]["problem_id"])
        picks.append(chosen)
    out = pd.concat(picks, ignore_index=True)[keep].copy()
    out["chunk_id"] = 0
    print(f"[SMOKE2] selected {len(out)} samples")
    print(out[["problem_id", "subject", "level", "problem_qwen_tok_len",
               "r1_cot_token_count", "answer"]].to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="also write smoke2_samples.parquet (~12 rows)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        smoke = select_smoke()
        sp = OUT_DIR / "smoke2_samples.parquet"
        smoke.to_parquet(sp, index=False)
        print(f"\n[OK] wrote {len(smoke)} -> {sp}")

    out = select_full()
    op = OUT_DIR / "unit30_samples.parquet"
    out.to_parquet(op, index=False)
    print(f"\n[OK] wrote {len(out)} -> {op}")


if __name__ == "__main__":
    main()
