#!/usr/bin/env python3
"""
select_unit_samples_v2.py
=========================
Phase Z-2 PILOT-2 (replication) sample selection.

Same spec as `select_unit_samples.py` (unit = (subject, level) cell, K=30 per
non-empty cell, balanced chunk_id), BUT this draws a set that is **disjoint**
from the original ~1541 pilot. We load the original pilot's problem_ids, exclude
them from the full universe, then sample K=30 per cell with a NEW seed (43).

Goal: a fully independent replication set (~1500) so pilot ∪ pilot2 ≈ 3,000
unique problems — matching the hand-off "pilot 3,000" target.

Outputs (CPU only; safe from login shell):
  outputs/pilot2/unit30_v2_samples.parquet  -- full pilot2 set with chunk_id
  outputs/pilot2/smoke_v2.parquet           -- ~12 rows (new-code validation), --smoke

Columns kept (incl. ground-truth `answer` for is_correct scoring downstream).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

BASE = Path("/scratch/lami2026/personal/jimin_2782")
FULL = BASE / "src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet"
ACT_OUT = BASE / "src/OPSD_Curriculum/reasoning_pivot/activation/outputs"
ORIG_SAMPLES = ACT_OUT / "unit30_samples.parquet"   # original ~1541 pilot (READ ONLY)
OUT_DIR = ACT_OUT / "pilot2"                          # NEW dir; never touches pilot/

K_PER_CELL = 30
NUM_CHUNKS = 4
SEED = 43   # NEW seed (orig used 42) so even overlapping cells draw fresh rows

KEEP_COLS = [
    "problem_id", "row_index", "source", "subject", "subject_raw", "level",
    "problem_text", "problem_char_len", "problem_qwen_tok_len",
    "r1_cot_token_count", "solution_char_len",
    "answer", "correct",   # answer = ground-truth final answer; correct = R1 solution correctness
]


def load_excluded() -> set:
    """problem_ids already used in the original pilot (to guarantee disjointness)."""
    if not ORIG_SAMPLES.exists():
        raise FileNotFoundError(f"original pilot not found: {ORIG_SAMPLES}")
    orig = pd.read_parquet(ORIG_SAMPLES, columns=["problem_id"])
    ids = set(orig["problem_id"].tolist())
    print(f"[INFO] original pilot problem_ids to exclude: {len(ids)}")
    return ids


def select_full(excluded: set) -> pd.DataFrame:
    df = pd.read_parquet(FULL)
    print(f"[INFO] full universe: {df.shape}")
    before = len(df)
    df = df[~df["problem_id"].isin(excluded)].copy()
    print(f"[INFO] after excluding original pilot: {len(df)} (removed {before - len(df)})")
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
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out["chunk_id"] = [i % NUM_CHUNKS for i in range(len(out))]

    assert out["problem_id"].is_unique, "duplicate problem_id in pilot2 set!"
    # hard guarantee: zero intersection with original pilot
    inter = set(out["problem_id"]).intersection(excluded)
    assert len(inter) == 0, f"DISJOINT VIOLATION: {len(inter)} ids overlap original pilot!"

    # ── report ────────────────────────────────────────────────────────────
    print(f"\n[CELLS] {len(cells)} non-empty (subject x level) cells")
    short = [(s, l, full_n, k) for (s, l, full_n, k) in cells if k < K_PER_CELL]
    print(f"[CELLS] cells with < {K_PER_CELL} available (after exclusion): {len(short)}")
    for s, l, full_n, k in short:
        print(f"        {s:25s} L{l}: avail={full_n:5d} -> took {k}")
    print(f"\n[TOTAL] selected = {len(out)} samples (DISJOINT from original {len(excluded)})")
    print("[CHUNK] sizes:", out["chunk_id"].value_counts().sort_index().to_dict())
    print("\n[SUBJECT x LEVEL] selected counts:")
    print(pd.crosstab(out["subject"], out["level"]))
    return out


def select_smoke(excluded: set) -> pd.DataFrame:
    """~12 rows for NEW-code validation: cover levels 1..8 + extra hard (L6/7/8),
    excluding original pilot ids. Includes ground-truth `answer`."""
    df = pd.read_parquet(FULL)
    df = df[~df["problem_id"].isin(excluded)].copy()
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
    print(f"[SMOKE_V2] selected {len(out)} samples")
    print(out[["problem_id", "subject", "level", "problem_qwen_tok_len",
               "r1_cot_token_count", "answer"]].to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="also write smoke_v2.parquet (~12 rows)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded = load_excluded()

    if args.smoke:
        smoke = select_smoke(excluded)
        sp = OUT_DIR / "smoke_v2.parquet"
        smoke.to_parquet(sp, index=False)
        print(f"\n[OK] wrote {len(smoke)} -> {sp}")

    out = select_full(excluded)
    op = OUT_DIR / "unit30_v2_samples.parquet"
    out.to_parquet(op, index=False)
    print(f"\n[OK] wrote {len(out)} -> {op}")


if __name__ == "__main__":
    main()
