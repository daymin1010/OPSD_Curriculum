#!/usr/bin/env python3
"""
verify_schedule.py
==================
Regression gate for the OPSD curriculum data layer. Run AFTER `phase0`.

Checks, in order (any failure -> exit 1):
  1. Phase-0 join meta (`outputs/join_setA_meta.json`) matches the frozen
     Phase-0 reference EXACTLY (n_total, n_setA, match_rate, 16 main cells,
     4 diff-only stages). This catches silent data drift (e.g. a missing shard,
     a relabel, a join-key regression).
  2. Schedule dry-run for BOTH arms (main 16-stage, diffonly 4-stage):
     - schedule length == T * B_glob
     - length a multiple of B_glob (no optimizer step straddles a stage)
     - every stage pool non-empty
     - per-stage budget == round(T/num_stages)
     - reports cycles (pool re-use factor) per stage.

CPU only. No model load. Safe to run on a login node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import curriculum_schedule as cs

# ----------------------------------------------------------------------------
# Frozen Phase-0 reference (computed 2026-06-18; opsd full 29,434 / Set-A 28,771)
# These are the numbers that downstream curriculum claims depend on.
# ----------------------------------------------------------------------------
REF = {
    "n_total_opsd": 29434,
    "n_setA": 28771,
    "n_unmatched": 0,
    "n_other": 663,
    "match_rate": 1.0,
    "main_cells": {
        "D1|C1": 252,  "D1|C2": 1201, "D1|C3": 3703, "D1|C4": 393,
        "D2|C1": 1999, "D2|C2": 3684, "D2|C3": 4084, "D2|C4": 2114,
        "D3|C1": 1955, "D3|C2": 3504, "D3|C3": 1860, "D3|C4": 2259,
        "D4|C1": 278,  "D4|C2": 906,  "D4|C3": 79,   "D4|C4": 500,
    },
    "diff_cells": {"D1": 5549, "D2": 11881, "D3": 9578, "D4": 1763},
}

OUT_DIR = cs.OUT_DIR
META_PATH = OUT_DIR / "join_setA_meta.json"
ROW_TABLE = OUT_DIR / "join_setA_rows.parquet"
STAGES_MAIN = cs.STAGES_DIR / "stages_arm3_excludeOther.json"
STAGES_DIFF = cs.STAGES_DIR / "stages_diffonly_setA.json"


def _fail(msg: str):
    print(f"[verify] FAIL: {msg}", flush=True)
    sys.exit(1)


def check_phase0_meta() -> dict:
    if not META_PATH.exists():
        _fail(f"missing {META_PATH} — run `python curriculum_schedule.py phase0` first")
    meta = json.loads(META_PATH.read_text())

    if not meta.get("gate_pass"):
        _fail("phase0 gate_pass=False")

    scalar_checks = [
        ("n_total_opsd", meta["n_total_opsd"], REF["n_total_opsd"]),
        ("n_setA", meta["n_setA"], REF["n_setA"]),
        ("n_unmatched", meta["n_unmatched"], REF["n_unmatched"]),
        ("n_other", meta["n_other"], REF["n_other"]),
        ("match_rate", round(meta["join_info"]["match_rate"], 6), REF["match_rate"]),
    ]
    for name, got, exp in scalar_checks:
        if got != exp:
            _fail(f"{name}={got} != reference {exp}")
        print(f"[verify] OK  {name} = {got}", flush=True)

    for cell, exp in REF["main_cells"].items():
        got = meta["main_cells"].get(cell)
        if got != exp:
            _fail(f"main_cell {cell}={got} != reference {exp}")
    print(f"[verify] OK  16 main cells match reference "
          f"(min={min(meta['main_cells'].values())}, "
          f"empty={sum(1 for v in meta['main_cells'].values() if v == 0)})", flush=True)

    for d, exp in REF["diff_cells"].items():
        got = meta["diff_cells"].get(d)
        if got != exp:
            _fail(f"diff_cell {d}={got} != reference {exp}")
    print("[verify] OK  4 diff-only stage counts match reference", flush=True)

    return meta


def check_schedule(arm: str, stages_json: Path, T: int, B_glob: int, seed: int = 42):
    if not stages_json.exists():
        _fail(f"missing stages JSON {stages_json}")
    if not ROW_TABLE.exists():
        _fail(f"missing row table {ROW_TABLE} — run phase0 first")

    sched, meta = cs.build_schedule(stages_json, ROW_TABLE, arm, T, B_glob, seed)
    exp_len = T * B_glob
    if len(sched) != exp_len:
        _fail(f"[{arm}] schedule len {len(sched)} != T*B_glob {exp_len}")
    if len(sched) % B_glob != 0:
        _fail(f"[{arm}] schedule len {len(sched)} not a multiple of B_glob {B_glob}")

    num_stages = meta["num_stages"]
    budget = meta["budget_per_stage"]
    exp_budget = round(T / num_stages)
    if budget != exp_budget:
        _fail(f"[{arm}] budget {budget} != round(T/num_stages) {exp_budget}")

    empties = [s["stage_index"] for s in meta["stages"] if s["pool_size"] == 0]
    if empties:
        _fail(f"[{arm}] empty stage pools: {empties}")

    max_idx = max(sched)
    if max_idx >= REF["n_total_opsd"]:
        _fail(f"[{arm}] schedule index {max_idx} out of range (n={REF['n_total_opsd']})")

    print(f"[verify] OK  arm={arm}: stages={num_stages} budget/stage={budget} "
          f"len={len(sched)} (T*B_glob={exp_len})", flush=True)
    for s in meta["stages"]:
        print(f"[verify]      stage {s['stage_index']:2d} "
              f"[{s.get('difficulty')}|{s.get('subject_cluster')}] "
              f"pool={s['pool_size']:5d} slots={s['slots']:6d} cycles={s['cycles']}",
              flush=True)
    return meta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=480, help="schedule steps for dry-run")
    ap.add_argument("--B_glob", type=int, default=32,
                    help="global batch for dry-run (OPSD effective batch = pd*ga*ws; "
                         "1.7B: 4*2*4=32, 8B: 2*4*4=32 — both arms ws=4)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("VERIFY 1/2 — Phase-0 join meta vs frozen reference")
    print("=" * 70)
    check_phase0_meta()

    print("\n" + "=" * 70)
    print(f"VERIFY 2/2 — schedule dry-run (T={args.T}, B_glob={args.B_glob})")
    print("=" * 70)
    check_schedule("diffonly", STAGES_DIFF, args.T, args.B_glob, args.seed)
    print()
    check_schedule("main", STAGES_MAIN, args.T, args.B_glob, args.seed)

    print("\n[verify] ALL CHECKS PASSED ✓", flush=True)


if __name__ == "__main__":
    main()
