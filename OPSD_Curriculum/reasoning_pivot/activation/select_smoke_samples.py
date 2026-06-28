#!/usr/bin/env python3
"""
select_smoke_samples.py
=======================
Phase Z-1 smoke: deterministically select 7 samples from the pilot universe,
one each at level 1, 2, 4, 5, 6, 7, 8 (difficulty-mixed, hard-heavy for
truncation stress test).

Output: outputs/smoke_samples.parquet  (problem_id, level, problem_text, ...)

CPU only — safe to run from a login shell.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path("/scratch/lami2026/personal/jimin_2782")
PILOT = BASE / "src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet"
OUT_DIR = BASE / "src/OPSD_Curriculum/reasoning_pivot/activation/outputs"
OUT = OUT_DIR / "smoke_samples.parquet"

LEVELS = [1, 2, 4, 5, 6, 7, 8]
SEED = 42

KEEP_COLS = [
    "problem_id", "row_index", "source", "subject", "level",
    "problem_text", "problem_char_len", "problem_qwen_tok_len",
    "r1_cot_token_count", "correct",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PILOT)
    print(f"[INFO] pilot universe: {df.shape}")

    picks = []
    for lv in LEVELS:
        sub = df[df["level"] == lv]
        if len(sub) == 0:
            print(f"[WARN] level {lv}: no rows, skipping")
            continue
        # deterministic shuffle within level, take first
        chosen = sub.sample(n=1, random_state=SEED)
        picks.append(chosen)
        row = chosen.iloc[0]
        print(f"[PICK] level={lv} id={row['problem_id']} "
              f"subject={row['subject']} qtok={row['problem_qwen_tok_len']} "
              f"r1_cot={row['r1_cot_token_count']}")

    out = pd.concat(picks, ignore_index=True)
    keep = [c for c in KEEP_COLS if c in out.columns]
    out = out[keep].reset_index(drop=True)

    assert out["problem_id"].is_unique, "duplicate problem_id in smoke set!"
    assert len(out) == len([lv for lv in LEVELS]), "missing some level pick"

    out.to_parquet(OUT, index=False)
    print(f"\n[OK] wrote {len(out)} samples -> {OUT}")
    print(out[["problem_id", "level", "subject", "problem_qwen_tok_len",
               "r1_cot_token_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
