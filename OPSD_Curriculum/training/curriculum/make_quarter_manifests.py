#!/usr/bin/env python3
"""
make_quarter_manifests.py
=========================
Build 1/4-scale manifests for fair A/B (cond2_diff vs cond3_ours_C2).

CRITICAL fairness constraint
----------------------------
Both source manifests share the SAME universe of 28,771 problem_ids
(only the stage assignment differs).  `verify_schedule_manifest_once.py`
enforces `diff_set == ours_set`.  Therefore we must NOT sample each arm
independently; we sample a SINGLE common subset once and intersect each
arm's stage_ids with that subset.

Procedure
---------
1. Load both source manifests; assert their problem_id sets are identical.
2. Sort the universe deterministically and pick frac (=0.25) of ids using
   numpy seeded RNG (seed=42).
3. For each arm, for each stage, keep only ids in selected_ids (preserving
   stage_index, order_index, and stage metadata).
4. Write `*_q4.json` for both arms.  Both arms now share the SAME set of
   ~7,193 problems; only the stage groupings differ.

Result
------
- N (per arm)   ≈ 7,193  (depends on hash distribution; both arms equal)
- T (per arm)   ≈ ceil(N / 32) ≈ 225 optimizer steps
- diff_set == ours_set                          ✓ verify check 1
- universe is a proper subset of Set-A (28,771) ✓ no missing rows
"""
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
STAGES_DIR = REPO / "training" / "stages_tiered_20260622"


def union_ids(manifest: dict) -> set[str]:
    s: set[str] = set()
    for st in manifest["stages"]:
        s.update(str(x) for x in st["problem_ids"])
    return s


def filter_manifest(manifest: dict, keep: set[str]) -> dict:
    new_stages = []
    total = 0
    for st in manifest["stages"]:
        ids = [str(x) for x in st["problem_ids"] if str(x) in keep]
        new_st = dict(st)
        new_st["problem_ids"] = ids
        new_st["n"] = len(ids)
        new_st["source_n"] = len(st["problem_ids"])
        new_stages.append(new_st)
        total += len(ids)
    out = dict(manifest)
    out["stages"] = new_stages
    out["n_stages"] = len(new_stages)
    out["spec"] = manifest.get("spec", "") + "_q4"
    return out, total


def main():
    diff_src = STAGES_DIR / "stages_cond2_diff.json"
    ours_src = STAGES_DIR / "stages_cond3_ours_C2.json"
    diff_dst = STAGES_DIR / "stages_cond2_diff_q4.json"
    ours_dst = STAGES_DIR / "stages_cond3_ours_C2_q4.json"
    frac = 0.25
    seed = 42

    diff = json.loads(diff_src.read_text())
    ours = json.loads(ours_src.read_text())

    diff_ids = union_ids(diff)
    ours_ids = union_ids(ours)
    if diff_ids != ours_ids:
        only_diff = len(diff_ids - ours_ids)
        only_ours = len(ours_ids - diff_ids)
        raise SystemExit(
            f"[quarter] source manifests have different universes: "
            f"diff_only={only_diff}, ours_only={only_ours}; cannot build fair q4"
        )
    universe_sorted = sorted(diff_ids)  # deterministic
    N = len(universe_sorted)
    n_target = int(round(N * frac))
    print(f"[quarter] universe N={N}, target N_q4={n_target} (frac={frac}, seed={seed})")

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_target, replace=False)
    idx.sort()
    selected: set[str] = {universe_sorted[i] for i in idx}
    assert len(selected) == n_target

    diff_q4, diff_total = filter_manifest(diff, selected)
    ours_q4, ours_total = filter_manifest(ours, selected)
    diff_q4_ids = union_ids(diff_q4)
    ours_q4_ids = union_ids(ours_q4)
    assert diff_q4_ids == selected, (
        f"diff_q4 universe mismatch: missing={len(selected - diff_q4_ids)} extra={len(diff_q4_ids - selected)}"
    )
    assert ours_q4_ids == selected, (
        f"ours_q4 universe mismatch: missing={len(selected - ours_q4_ids)} extra={len(ours_q4_ids - selected)}"
    )
    assert diff_q4_ids == ours_q4_ids, "diff_q4 and ours_q4 universes differ (must be identical)"
    assert diff_total == ours_total == n_target

    # metadata
    for m in (diff_q4, ours_q4):
        m["subsample_frac"] = frac
        m["subsample_seed"] = seed
        m["source_universe_n"] = N
        m["sampled_n"] = n_target

    diff_q4["source_manifest"] = str(diff_src)
    ours_q4["source_manifest"] = str(ours_src)

    diff_dst.write_text(json.dumps(diff_q4, indent=2))
    ours_dst.write_text(json.dumps(ours_q4, indent=2))

    print(f"[quarter] wrote {diff_dst.name}  total={diff_total}  T~{int(np.ceil(diff_total/32))}")
    for st in diff_q4["stages"]:
        print(f"  diff stage{st['stage_index']} order={st['order_index']}: {st['source_n']} -> {st['n']}")
    print(f"[quarter] wrote {ours_dst.name}  total={ours_total}  T~{int(np.ceil(ours_total/32))}")
    for st in ours_q4["stages"]:
        print(f"  ours stage{st['stage_index']} order={st['order_index']}: {st['source_n']} -> {st['n']}")

    print(f"[quarter] OK: diff/ours share identical universe of {len(diff_q4_ids)} problems")


if __name__ == "__main__":
    main()
