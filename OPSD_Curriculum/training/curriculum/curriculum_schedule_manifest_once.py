#!/usr/bin/env python3
"""
curriculum_schedule_manifest_once.py
====================================

Manifest-based OPSD curriculum schedule builder for the 2026-06-22 wave-1
fair 1-pass experiment.

This file is intentionally separate from `curriculum_schedule.py` so the v1
legacy schedule code remains untouched.  The source of truth here is
`stages[*].problem_ids` in a stage manifest such as:

  * stages_cond2_diff.json
  * stages_cond3_ours_C2.json

Design invariants:
  * stage order is preserved: order_index, fallback stage_index;
  * within-stage order may be deterministic shuffle or manifest order;
  * each problem_id appears exactly once per pass;
  * no cycling, no padding, no drop-last at schedule construction time;
  * boundary optimizer steps may contain multiple stages and are verified by an
    expected per-step Counter.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from curriculum_schedule import OUT_DIR


def load_stage_manifest(stages_json_path: Path) -> dict:
    """Load and minimally validate a stages[*].problem_ids manifest."""
    p = Path(stages_json_path)
    if not p.exists():
        raise FileNotFoundError(f"[manifest] stages JSON missing: {p}")

    data = json.loads(p.read_text())
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"[manifest] {p} must contain a non-empty list key 'stages'")

    seen_stage_indices: set[int] = set()
    for i, st in enumerate(stages):
        if "stage_index" not in st:
            raise ValueError(f"[manifest] stage #{i} missing stage_index")
        sidx = int(st["stage_index"])
        if sidx in seen_stage_indices:
            raise ValueError(f"[manifest] duplicated stage_index={sidx}")
        seen_stage_indices.add(sidx)

        ids = st.get("problem_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"[manifest] stage_index={sidx} has no problem_ids")
        if "n" in st and int(st["n"]) != len(ids):
            raise ValueError(
                f"[manifest] stage_index={sidx} n={st['n']} "
                f"!= len(problem_ids)={len(ids)}"
            )

    if "n_stages" in data and int(data["n_stages"]) != len(stages):
        raise ValueError(f"[manifest] n_stages={data['n_stages']} != len(stages)={len(stages)}")
    return data


def _stage_sort_key(st: dict) -> tuple[int, int]:
    return (int(st.get("order_index", st.get("stage_index"))), int(st["stage_index"]))


def expected_counters_from_stage_per_pos(stage_per_pos: list[int], B_glob: int) -> list[dict[str, int]]:
    """Return one JSON-stable Counter per optimizer step.

    The final partial batch is included.  Keys are strings so the schedule meta
    can be JSON-serialized deterministically; the monitor converts them back to
    int before comparing with actual Counter(_stage_window).
    """
    B_glob = int(B_glob)
    if B_glob <= 0:
        raise ValueError(f"B_glob must be positive, got {B_glob}")

    out: list[dict[str, int]] = []
    for start in range(0, len(stage_per_pos), B_glob):
        cnt = Counter(int(x) for x in stage_per_pos[start:start + B_glob])
        out.append({str(k): int(v) for k, v in sorted(cnt.items())})
    return out


def build_schedule_from_stage_manifest(
    stages_json_path: Path,
    row_table_path: Path,
    B_glob: int = 32,
    seed: int = 42,
    within_stage_order: str = "shuffle",
    tail_policy: str = "partial",
    curriculum_passes: int = 1,
):
    """Build a fair OPSD schedule from stages[*].problem_ids.

    Returns:
      (schedule_indices, stage_per_pos, expected_stage_counters, meta)
    """
    B_glob = int(B_glob)
    seed = int(seed)
    curriculum_passes = int(curriculum_passes)
    if within_stage_order not in {"shuffle", "manifest"}:
        raise ValueError("within_stage_order must be 'shuffle' or 'manifest'")
    if tail_policy != "partial":
        raise ValueError("only tail_policy='partial' is currently supported")
    if curriculum_passes < 1:
        raise ValueError(f"curriculum_passes must be >=1, got {curriculum_passes}")

    manifest = load_stage_manifest(Path(stages_json_path))
    stages = sorted(manifest["stages"], key=_stage_sort_key)

    rows = pd.read_parquet(row_table_path, columns=["problem_id", "opsd_index", "in_setA"])
    if rows["problem_id"].duplicated().any():
        dup = rows.loc[rows["problem_id"].duplicated(), "problem_id"].head(5).tolist()
        raise ValueError(f"[manifest] row_table has duplicated problem_id values, e.g. {dup}")
    by_pid = rows.set_index("problem_id")

    base_schedule: list[int] = []
    base_stage_per_pos: list[int] = []
    all_ids: list[str] = []
    meta_stages: list[dict] = []
    pos = 0

    for st in stages:
        sidx = int(st["stage_index"])
        order_index = int(st.get("order_index", sidx))
        ids = [str(x) for x in st["problem_ids"]]
        all_ids.extend(ids)

        missing = [pid for pid in ids if pid not in by_pid.index]
        if missing:
            raise ValueError(
                f"[manifest] stage_index={sidx} has {len(missing)} missing problem_ids; "
                f"examples={missing[:5]}"
            )

        sub = by_pid.loc[ids]
        if manifest.get("include_other") is False:
            bad = sub.loc[~sub["in_setA"].astype(bool)]
            if len(bad):
                raise ValueError(
                    f"[manifest] include_other=false but stage_index={sidx} contains "
                    f"{len(bad)} non-Set-A rows; examples={list(bad.index[:5])}"
                )

        opsd_indices = [int(x) for x in sub["opsd_index"].tolist()]
        if within_stage_order == "shuffle":
            # Stage order is preserved.  Only the within-stage order is shuffled.
            rng = np.random.default_rng((seed ^ (sidx * 1_000_003)) + order_index)
            perm = rng.permutation(len(opsd_indices))
            opsd_indices = [opsd_indices[int(i)] for i in perm]

        first_pos = pos
        base_schedule.extend(opsd_indices)
        base_stage_per_pos.extend([sidx] * len(opsd_indices))
        pos += len(opsd_indices)
        last_pos = pos - 1
        meta_stages.append({
            "stage_index": sidx,
            "order_index": order_index,
            "pool_size": len(ids),
            "slots_per_pass": len(ids),
            "first_pos": first_pos,
            "last_pos": last_pos,
            "first_step": first_pos // B_glob,
            "last_step": last_pos // B_glob,
            "n_unique_problem_ids": len(set(ids)),
        })

    dup_count = len(all_ids) - len(set(all_ids))
    if dup_count:
        dups = [pid for pid, c in Counter(all_ids).items() if c > 1][:5]
        raise ValueError(f"[manifest] duplicate problem_ids within arm: {dup_count}; examples={dups}")

    schedule: list[int] = []
    stage_per_pos: list[int] = []
    for _ in range(curriculum_passes):
        schedule.extend(base_schedule)
        stage_per_pos.extend(base_stage_per_pos)

    T = int(math.ceil(len(schedule) / B_glob))
    expected_stage_counters = expected_counters_from_stage_per_pos(stage_per_pos, B_glob)
    assert len(expected_stage_counters) == T, (len(expected_stage_counters), T)
    assert len(schedule) == len(stage_per_pos), (len(schedule), len(stage_per_pos))

    tail_size = len(schedule) % B_glob
    meta = {
        "schedule_mode": "manifest_once",
        "spec": manifest.get("spec"),
        "include_other": manifest.get("include_other"),
        "stages_json": str(stages_json_path),
        "row_table": str(row_table_path),
        "n_examples_per_pass": len(base_schedule),
        "n_examples": len(schedule),
        "B_glob": B_glob,
        "T": T,
        "tail_policy": tail_policy,
        "tail_size": int(tail_size),
        "within_stage_order": within_stage_order,
        "seed": seed,
        "curriculum_passes": curriculum_passes,
        "duplicates": int(dup_count),
        "missing_problem_ids": 0,
        "num_stages": len(stages),
        "schedule_len": len(schedule),
        "context_per_stage": manifest.get("context_per_stage"),
        "stages": meta_stages,
        "expected_stage_counters": expected_stage_counters,
    }
    return schedule, stage_per_pos, expected_stage_counters, meta


def save_manifest_schedule_meta(meta: dict, run_name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"schedule_meta_manifest_once_{run_name}.json"
    p.write_text(json.dumps(meta, indent=2))
    print(f"[schedule-manifest-once] wrote {p}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(description="Build/inspect manifest_once OPSD schedule")
    ap.add_argument("--stages_json", required=True)
    ap.add_argument("--row_table", default=str(OUT_DIR / "join_setA_rows.parquet"))
    ap.add_argument("--B_glob", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--within_stage_order", choices=["shuffle", "manifest"], default="shuffle")
    ap.add_argument("--tail_policy", choices=["partial"], default="partial")
    ap.add_argument("--curriculum_passes", type=int, default=1)
    ap.add_argument("--run_name", default="debug")
    args = ap.parse_args()

    schedule, stage_per_pos, expected, meta = build_schedule_from_stage_manifest(
        Path(args.stages_json),
        Path(args.row_table),
        B_glob=args.B_glob,
        seed=args.seed,
        within_stage_order=args.within_stage_order,
        tail_policy=args.tail_policy,
        curriculum_passes=args.curriculum_passes,
    )
    save_manifest_schedule_meta(meta, args.run_name)
    print(
        f"[schedule-manifest-once] spec={meta.get('spec')} N={len(schedule)} "
        f"T={meta['T']} tail={meta['tail_size']} stages={meta['num_stages']} "
        f"expected_counters={len(expected)} stage_per_pos={len(stage_per_pos)}",
        flush=True,
    )


if __name__ == "__main__":
    main()