#!/usr/bin/env python3
"""
verify_schedule_manifest_once.py
================================

Standalone gate for the 2026-06-22 manifest-once OPSD curriculum experiment.

This file is intentionally separate from `verify_schedule.py` to avoid mixing
the legacy v1 cycling/padded schedule checks with the new fair 1-pass checks.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from curriculum_schedule import OUT_DIR
from curriculum_schedule_manifest_once import (
    build_schedule_from_stage_manifest,
    load_stage_manifest,
)


REF_SET_A_N = 28771
REF_SET_A_T_B32 = 900
REF_FULL_N = 29434
REF_FULL_T_B32 = 920


def _fail(msg: str):
    print(f"[verify-manifest-once] FAIL: {msg}", flush=True)
    sys.exit(1)


def collect_problem_ids(manifest: dict) -> list[str]:
    ids: list[str] = []
    for st in manifest["stages"]:
        ids.extend(str(x) for x in st.get("problem_ids", []))
    return ids


def check_single_arm(label: str, stages_json: Path, row_table: Path, B_glob: int, seed: int,
                     within_stage_order: str, curriculum_passes: int) -> dict:
    if not stages_json.exists():
        _fail(f"[{label}] missing stages_json: {stages_json}")
    if not row_table.exists():
        _fail(f"missing row_table: {row_table}")

    manifest = load_stage_manifest(stages_json)
    ids = collect_problem_ids(manifest)
    cnt = Counter(ids)
    dup = [pid for pid, c in cnt.items() if c > 1]
    if dup:
        _fail(f"[{label}] duplicate problem_ids={len(dup)} examples={dup[:5]}")

    rows = pd.read_parquet(row_table, columns=["problem_id", "opsd_index", "in_setA"])
    by_pid = rows.set_index("problem_id")
    missing = [pid for pid in ids if pid not in by_pid.index]
    if missing:
        _fail(f"[{label}] missing problem_ids in row_table={len(missing)} examples={missing[:5]}")

    sub = by_pid.loc[ids]
    if manifest.get("include_other") is False:
        non_setA = sub.loc[~sub["in_setA"].astype(bool)]
        if len(non_setA):
            _fail(f"[{label}] include_other=false but non-Set-A rows={len(non_setA)} examples={list(non_setA.index[:5])}")

    schedule, stage_per_pos, expected, meta = build_schedule_from_stage_manifest(
        stages_json,
        row_table,
        B_glob=B_glob,
        seed=seed,
        within_stage_order=within_stage_order,
        tail_policy="partial",
        curriculum_passes=curriculum_passes,
    )

    exp_N = len(ids) * curriculum_passes
    exp_T = math.ceil(exp_N / B_glob)
    if len(schedule) != exp_N:
        _fail(f"[{label}] schedule len {len(schedule)} != expected {exp_N}")
    if len(stage_per_pos) != len(schedule):
        _fail(f"[{label}] stage_per_pos len {len(stage_per_pos)} != schedule len {len(schedule)}")
    if len(expected) != exp_T or int(meta["T"]) != exp_T:
        _fail(f"[{label}] T mismatch expected={exp_T} meta={meta['T']} counters={len(expected)}")

    print(
        f"[verify-manifest-once] OK  {label}: spec={manifest.get('spec')} "
        f"include_other={manifest.get('include_other')} stages={len(manifest['stages'])} "
        f"N/pass={len(ids)} N/train={len(schedule)} T={meta['T']} tail={meta['tail_size']} "
        f"order={within_stage_order} passes={curriculum_passes}",
        flush=True,
    )
    for st in meta["stages"]:
        print(
            f"[verify-manifest-once]      stage={st['stage_index']} order={st['order_index']} "
            f"pool={st['pool_size']} first_step={st['first_step']} last_step={st['last_step']}",
            flush=True,
        )
    boundary = sum(1 for c in expected if len(c) > 1)
    print(f"[verify-manifest-once]      boundary_mixed_steps={boundary}", flush=True)
    return {"manifest": manifest, "ids": ids, "meta": meta, "expected": expected}


def main():
    ap = argparse.ArgumentParser(description="Verify diff/ours manifest-once OPSD schedules")
    ap.add_argument("--diff_json", required=True)
    ap.add_argument("--ours_json", required=True)
    ap.add_argument("--row_table", default=str(OUT_DIR / "join_setA_rows.parquet"))
    ap.add_argument("--B_glob", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--within_stage_order", choices=["shuffle", "manifest"], default="shuffle")
    ap.add_argument("--curriculum_passes", type=int, default=1)
    ap.add_argument("--expect_universe", choices=["setA", "full", "auto", "none"], default="setA")
    args = ap.parse_args()

    diff = check_single_arm(
        "cond2_diff", Path(args.diff_json), Path(args.row_table), args.B_glob,
        args.seed, args.within_stage_order, args.curriculum_passes,
    )
    ours = check_single_arm(
        "cond3_ours_C2", Path(args.ours_json), Path(args.row_table), args.B_glob,
        args.seed, args.within_stage_order, args.curriculum_passes,
    )

    diff_set = set(diff["ids"])
    ours_set = set(ours["ids"])
    if diff_set != ours_set:
        _fail(
            f"diff/ours problem_id set mismatch: diff_minus_ours={len(diff_set - ours_set)} "
            f"ours_minus_diff={len(ours_set - diff_set)}"
        )

    N = len(diff_set)
    T = int(diff["meta"]["T"])
    if int(ours["meta"]["T"]) != T:
        _fail(f"T mismatch: diff={T} ours={ours['meta']['T']}")

    if args.expect_universe == "setA":
        if N != REF_SET_A_N or T != REF_SET_A_T_B32 * args.curriculum_passes:
            _fail(f"expected Set-A N={REF_SET_A_N}, T={REF_SET_A_T_B32 * args.curriculum_passes}; got N={N}, T={T}")
    elif args.expect_universe == "full":
        if N != REF_FULL_N or T != REF_FULL_T_B32 * args.curriculum_passes:
            _fail(f"expected full OPSD N={REF_FULL_N}, T={REF_FULL_T_B32 * args.curriculum_passes}; got N={N}, T={T}")
    elif args.expect_universe == "none":
        # subset universes (e.g., q4) — skip absolute N/T check; diff/ours equality is still enforced above.
        print(f"[verify-manifest-once]      universe check skipped (--expect_universe none); N={N} T={T}", flush=True)

    print(
        f"[verify-manifest-once] OK  diff/ours same universe: N={N} T={T} "
        f"B_glob={args.B_glob} passes={args.curriculum_passes}",
        flush=True,
    )
    print("[verify-manifest-once] ALL CHECKS PASSED ✓", flush=True)


if __name__ == "__main__":
    main()